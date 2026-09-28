"""E4/E5 -- instrumented training: gate distribution, gradient norms, non-zero
initialisation ablation, and per-source reciprocal ranks.

The external assessment lists four things it could not see in the delivered
material: (i) what the gating values actually do, (ii) whether the zero
initialisation of the gate (which is what makes the untrained model equal the
content floor) matters, (iii) gradient norms, (iv) a per-source, source-level
inference rather than a seed-level paired t-test.  All four need instrumentation
*inside* the training loop, so this module re-implements ``train3.train_model``
with collection hooks and nothing else changed.

Fidelity: ``--check-fidelity`` runs the same (dataset, model, seed) through both
this loop and the frozen ``train3.train_model`` and compares the returned
metrics bit for bit; the per-source collector is validated against the frozen
``ranking.evaluate_ranking`` on the same scorer in every run.

Outputs (per job) under ``results_ext/e4_instrument/``:
  ``job_<dataset>_<model>_seed<seed>_<cond>.json``  metrics, gate stats, grads
  ``ps_<dataset>_<model>_seed<seed>_<cond>.npz``    per-source arrays

Conditions:
  ``zero``   the frozen configuration (gate MLP and time projection zero-init)
  ``rand``   gate MLP initialised as N(0, sigma) with sigma = --gate-sigma,
             so the untrained model no longer equals the content floor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from content_floor import build_bundle                      # noqa: E402
from gated import _Gate, build_model                        # noqa: E402
from ranking import evaluate_ranking                        # noqa: E402
from train3 import _make_scorer, _sampled_pairs, set_seed   # noqa: E402

OUT = ROOT / "results_ext" / "e4_instrument"
SEEDS = {"npm": list(range(101, 111)), "maven": list(range(121, 131))}


# --------------------------------------------------------------------------- #
# per-source ranking (same loop as ranking.evaluate_ranking + collection)      #
# --------------------------------------------------------------------------- #
def per_source_ranking(graph, split, scorer, stage: str) -> dict:
    """Return the frozen aggregate *and* the per-source arrays behind it."""
    mp_edges = split[f"mp_edges_{stage}"]
    core_mask = split["core_mask"]
    core = torch.nonzero(core_mask, as_tuple=False).flatten()
    core_np = core.numpy()
    n_nodes = int(core_mask.numel())
    fit_edges = split["fit_window"]["fit_mp_edges"]
    fit_targets: dict[int, np.ndarray] = {}
    if fit_edges.numel():
        pairs_np = fit_edges.t().numpy()
        order = np.argsort(pairs_np[:, 0], kind="stable")
        pairs_np = pairs_np[order]
        start = 0
        while start < len(pairs_np):
            source = int(pairs_np[start, 0])
            end = start
            while end < len(pairs_np) and int(pairs_np[end, 0]) == source:
                end += 1
            fit_targets[source] = pairs_np[start:end, 1]
            start = end

    by_source: dict[int, list[int]] = {}
    for source, target in mp_edges.t().tolist():
        by_source.setdefault(int(source), []).append(int(target))

    col_of = np.full(n_nodes, -1, dtype=np.int64)
    col_of[core_np] = np.arange(core_np.size, dtype=np.int64)
    blocked = np.zeros(n_nodes, dtype=bool)
    rank_of_node = np.empty(n_nodes, dtype=np.int64)

    rows = []
    for source, targets in sorted(by_source.items()):
        mine = fit_targets.get(source)
        if mine is not None:
            blocked[mine] = True
        blocked[source] = True
        keep = core_np[~blocked[core_np]]
        if keep.size == 0:
            if mine is not None:
                blocked[mine] = False
            blocked[source] = False
            continue
        pairs = torch.from_numpy(np.stack([np.full(keep.size, source, dtype=np.int64), keep]))
        scores = np.asarray(scorer(pairs))
        order = np.argsort(-scores, kind="stable")
        rank_of_keep = np.empty(keep.size, dtype=np.int64)
        rank_of_keep[order] = np.arange(1, keep.size + 1, dtype=np.int64)
        rank_of_node[keep] = rank_of_keep
        tgt = np.asarray(targets, dtype=np.int64)
        tgt_col = col_of[tgt]
        reachable = tgt[(tgt_col >= 0) & (~blocked[tgt])]
        ranks = rank_of_node[reachable] if reachable.size else np.zeros(0, dtype=np.int64)
        if mine is not None:
            blocked[mine] = False
        blocked[source] = False
        if ranks.size == 0:
            continue
        best = int(ranks.min())
        rows.append((source, best, 1.0 / best, int(ranks.size), int(len(targets))))

    best_arr = np.array([r[1] for r in rows], dtype=np.int64)
    agg = {
        "mrr": float(np.mean([r[2] for r in rows])) if rows else 0.0,
        "hits@10": float(np.mean(best_arr <= 10)) if rows else 0.0,
        "hits@100": float(np.mean(best_arr <= 100)) if rows else 0.0,
        "sources": len(rows),
        "positives": int(sum(r[3] for r in rows)),
        "candidates_mean": float(core.numel() - 1) if rows else 0.0,
    }
    arrays = {
        "source": np.array([r[0] for r in rows], dtype=np.int64),
        "best_rank": best_arr,
        "rr": np.array([r[2] for r in rows], dtype=np.float64),
        "n_reachable": np.array([r[3] for r in rows], dtype=np.int64),
        "n_positives": np.array([r[4] for r in rows], dtype=np.int64),
    }
    return {"aggregate": agg, "arrays": arrays}


def _assert_matches_frozen(graph, split, scorer, stage: str, collected: dict) -> None:
    """The collector must reproduce the frozen aggregate exactly."""
    frozen = evaluate_ranking(graph, split, scorer, stage=stage)
    got = collected["aggregate"]
    for key in ("mrr", "hits@10", "hits@100", "sources", "positives"):
        if abs(float(frozen[key]) - float(got[key])) > 1e-12:
            raise AssertionError(f"collector mismatch on {key}: frozen={frozen[key]} vs {got[key]}")


# --------------------------------------------------------------------------- #
# instrumentation helpers                                                      #
# --------------------------------------------------------------------------- #
def gate_stats(model: nn.Module, desc: torch.Tensor) -> dict:
    """Value distribution of every gate in the encoder, at the current weights."""
    out: dict[str, dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, _Gate):
            with torch.no_grad():
                pre = module.mlp(desc).squeeze(-1)
                g = 1.0 + torch.tanh(pre)
            p = pre.numpy()
            out[name] = {
                "n": int(p.size),
                "pre_abs_mean": float(np.abs(p).mean()),
                "pre_abs_p50": float(np.percentile(np.abs(p), 50)),
                "pre_abs_p95": float(np.percentile(np.abs(p), 95)),
                "pre_abs_max": float(np.abs(p).max()),
                "g_mean": float(g.mean()), "g_sd": float(g.std(unbiased=False)),
                "g_min": float(g.min()), "g_max": float(g.max()),
                "share_g_below_0.99": float((g < 0.99).double().mean()),
                "share_g_above_1.01": float((g > 1.01).double().mean()),
                "weight_norm": float(module.mlp.weight.norm().item()),
            }
    return out


def grad_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Split parameters into gate / time / encoder groups for gradient norms."""
    groups: dict[str, list[nn.Parameter]] = {"gate": [], "time": [], "encoder": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".gate_" in name or name.endswith("gate_in.mlp.weight") or ".gate_in." in name or ".gate_out." in name:
            groups["gate"].append(param)
        elif "time_proj" in name:
            groups["time"].append(param)
        else:
            groups["encoder"].append(param)
    return groups


def grad_norms(groups: dict[str, list[nn.Parameter]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, params in groups.items():
        total = 0.0
        seen = False
        for p in params:
            if p.grad is None:
                continue
            seen = True
            total += float(p.grad.detach().pow(2).sum().item())
        out[name] = (total ** 0.5) if seen else None
    return out


def nonzero_gate_init(model: nn.Module, sigma: float) -> int:
    """Replace the zero gate initialisation by N(0, sigma); returns how many."""
    touched = 0
    for module in model.modules():
        if isinstance(module, _Gate):
            with torch.no_grad():
                module.mlp.weight.normal_(0.0, sigma)
                if module.mlp.bias is not None:
                    module.mlp.bias.normal_(0.0, sigma)
            touched += 1
    return touched

# --------------------------------------------------------------------------- #
# instrumented training loop (mirrors train3.train_model)                      #
# --------------------------------------------------------------------------- #
def train_instrumented(bundle, model_name: str, cfg: dict, seed: int, *,
                       max_epochs: int = 100, patience: int = 20,
                       negatives: int = 10, eval_every: int = 2,
                       condition: str = "zero", gate_sigma: float = 0.05,
                       graph_free: bool = False) -> dict:
    """``graph_free`` removes *all* structural input from the model -- empty
    message-passing edge sets and zeroed descriptors -- while keeping the frozen
    evaluation protocol (candidate pool, leakage filter, positives) untouched, so
    the ablation isolates "how much of the gain is graph structure" rather than
    changing the task."""
    graph, split = bundle["graph"], bundle["split"]
    features = bundle["features"]
    descriptors = bundle["descriptors"]
    edge_time = bundle["edge_time"]
    fit_window = split["fit_window"]

    empty_edges = torch.zeros((2, 0), dtype=torch.long)
    empty_t = torch.zeros((0, 1), dtype=torch.float32)
    if graph_free:
        ef_e, ef_t = empty_edges, empty_t
        ev_e, ev_t = empty_edges, empty_t
        et_e, et_t = empty_edges, empty_t
        descriptors = torch.zeros_like(descriptors)
    else:
        ef_e, ef_t = fit_window["fit_mp_edges"], edge_time["fit"]
        ev_e, ev_t = split["mp_edges_val"], edge_time["val"]
        et_e, et_t = split["mp_edges_test"], edge_time["test"]

    set_seed(seed)
    model = build_model(model_name, features.shape[1], cfg)
    n_gates = nonzero_gate_init(model, gate_sigma) if condition == "rand" else 0
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                                 weight_decay=float(cfg.get("weight_decay", 1e-5)))
    loss_fn = nn.BCEWithLogitsLoss()
    groups = grad_groups(model)

    fit_mp, fit_pos = ef_e, fit_window["fit_pos"]
    edge_time_fit = ef_t
    core_indices = np.nonzero(fit_window["core_mask"].numpy())[0]

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val, best_epoch, stale, last_epoch = -1.0, -1, 0, 0
    val_curve, history = [], []

    scorer = _make_scorer(model, features, ev_e, descriptors, ev_t)
    init_val = evaluate_ranking(graph, split, scorer, stage="val")["mrr"]
    val_curve.append(init_val)
    init_gates = gate_stats(model, descriptors)
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch
        model.train()
        optimizer.zero_grad()
        pairs, labels = _sampled_pairs(fit_pos, core_indices, negatives,
                                       np.random.default_rng(seed * 100003 + epoch))
        z = model.encode(features, fit_mp, descriptors, edge_time_fit)
        loss = loss_fn(model.score(z, pairs), labels)
        loss.backward()
        gn = grad_norms(groups)
        optimizer.step()

        if epoch % eval_every == 0 or epoch == 1:
            sc = _make_scorer(model, features, ev_e, descriptors, ev_t)
            val_mrr = evaluate_ranking(graph, split, sc, stage="val")["mrr"]
            val_curve.append(val_mrr)
            history.append({"epoch": epoch, "loss": float(loss.item()),
                            "val_mrr": float(val_mrr), "grad_norms": gn,
                            "gate_stats": gate_stats(model, descriptors)})
            if val_mrr > best_val + 1e-6:
                best_val, best_epoch, stale = val_mrr, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

    model.load_state_dict(best_state)
    seconds = round(time.time() - started, 1)

    def stage(stage_name: str) -> dict:
        e, t = (ev_e, ev_t) if stage_name == "val" else (et_e, et_t)
        sc = _make_scorer(model, features, e, descriptors, t)
        col = per_source_ranking(graph, split, sc, stage=stage_name)
        _assert_matches_frozen(graph, split, sc, stage_name, col)
        frozen = evaluate_ranking(graph, split, sc, stage=stage_name)
        col["frozen"] = {k: float(frozen[k]) for k in ("mrr", "hits@10", "hits@100")}
        return col

    def floor_stage(stage_name: str) -> dict:
        zf = features

        def sc(pairs):
            left = zf.index_select(0, pairs[0])
            right = zf.index_select(0, pairs[1])
            return (left * right).sum(dim=-1).numpy()
        col = per_source_ranking(graph, split, sc, stage=stage_name)
        _assert_matches_frozen(graph, split, sc, stage_name, col)
        return col

    val_col, test_col = stage("val"), stage("test")
    floor_val_col, floor_test_col = floor_stage("val"), floor_stage("test")

    # what the gates respond to: correlation of the pre-activation with each
    # descriptor dimension, at the selected checkpoint
    desc_names = (bundle.get("descriptor_metadata") or {}).get("names")
    correlations: dict[str, list[float]] = {}
    d = descriptors.numpy()
    d_std = d.std(axis=0)
    d_c = d - d.mean(axis=0)
    for name, module in model.named_modules():
        if isinstance(module, _Gate):
            with torch.no_grad():
                pre = module.mlp(descriptors).squeeze(-1).numpy()
            pre_c = pre - pre.mean()
            denom = d_std * np.sqrt((pre_c ** 2).sum())
            corr = np.where(denom > 1e-12, (d_c * pre_c[:, None]).sum(axis=0) / np.maximum(denom, 1e-12), 0.0)
            correlations[name] = [float(x) for x in corr]

    return {
        "dataset": bundle["dataset"], "model": model_name, "seed": seed,
        "config": cfg, "config_id": cfg.get("id"), "condition": condition,
        "graph_free": graph_free,
        "gate_sigma": gate_sigma if condition == "rand" else None,
        "gates_reinitialised": n_gates,
        "checkpoint_rule": os.environ.get("CHECKPOINT_RULE", "best_val"),
        "init_val_mrr": float(init_val), "init_gate_stats": init_gates,
        "best_epoch": best_epoch, "best_val_mrr": float(best_val),
        "epochs_run": last_epoch, "train_seconds": seconds,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "selected_gate_stats": gate_stats(model, descriptors),
        "gate_descriptor_correlation": correlations,
        "descriptor_names": desc_names,
        "history": history,
        "val_mrr_curve": val_curve,
        "val": val_col["frozen"], "test": test_col["frozen"],
        "floor_val": floor_val_col["aggregate"], "floor_test": floor_test_col["aggregate"],
        "floor_recomputed_here": True,
        "val_arrays": val_col["arrays"], "test_arrays": test_col["arrays"],
        "floor_val_arrays": floor_val_col["arrays"], "floor_test_arrays": floor_test_col["arrays"],
    }


def run_job(dataset: str, model: str, seed: int, condition: str, gate_sigma: float,
            max_epochs: int, patience: int, eval_every: int,
            graph_free: bool = False) -> dict:
    import json as _json
    from run_experiments import MANIFEST, _canonical

    bundle = build_bundle(dataset, seed)
    selections = _json.loads((ROOT / "results" / "selections.json").read_text(encoding="utf-8"))
    gid = selections[dataset]["used"][_canonical(model)]
    cfg = next(c for c in MANIFEST["tuning"]["grid"] if c["id"] == gid)
    payload = train_instrumented(bundle, model, cfg, seed, max_epochs=max_epochs,
                                 patience=patience, negatives=MANIFEST["training"]["negatives_per_positive"],
                                 eval_every=eval_every, condition=condition, gate_sigma=gate_sigma,
                                 graph_free=graph_free)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{model}_seed{seed}_{condition}" + ("_textonly" if graph_free else "")
    arrays = {k: v for k, v in payload.items() if k.endswith("_arrays")}
    for k in list(arrays):
        del payload[k]
    flat = {f"{stage}_{name}": arr for stage, d in arrays.items() for name, arr in d.items()}
    np.savez_compressed(OUT / f"ps_{stem}.npz", **flat)
    (OUT / f"job_{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"stem": stem, "test_mrr": payload["test"]["mrr"], "floor": payload["floor_test"]["mrr"],
            "init_val_mrr": payload["init_val_mrr"], "seconds": payload["train_seconds"],
            "gates_reinitialised": payload["gates_reinitialised"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["npm", "maven"])
    ap.add_argument("--models", nargs="+", default=["gat_dir", "ragat_sym"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--conditions", nargs="+", default=["zero", "rand"])
    ap.add_argument("--gate-sigma", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--graph-free", action="store_true",
                    help="remove all structural input (empty MP edges, zero descriptors)")
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = []
    for dataset in args.datasets:
        for model in args.models:
            for seed in (args.seeds or SEEDS[dataset]):
                for cond in args.conditions:
                    stem = f"{dataset}_{model}_seed{seed}_{cond}" + ("_textonly" if args.graph_free else "")
                    if (OUT / f"job_{stem}.json").exists():
                        continue
                    jobs.append((dataset, model, seed, cond))
    print(f"[e4] {len(jobs)} jobs, workers={args.workers}", flush=True)
    if args.workers <= 1:
        for j in jobs:
            r = run_job(*j, args.gate_sigma, args.epochs, args.patience, args.eval_every,
                        args.graph_free)
            print(f"[e4] {r['stem']} test={r['test_mrr']:.4f} floor={r['floor']:.4f} "
                  f"init={r['init_val_mrr']:.4f} ({r['seconds']}s)", flush=True)
        return 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, *j, args.gate_sigma, args.epochs, args.patience,
                          args.eval_every, args.graph_free): j
                for j in jobs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[e4][error] {futs[fut]}: {type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"[e4] {r['stem']} test={r['test_mrr']:.4f} floor={r['floor']:.4f} "
                  f"init={r['init_val_mrr']:.4f} ({r['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
