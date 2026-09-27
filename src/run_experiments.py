"""Run the tuning grid and the multi-seed judgment runs on npm / Maven.

Replaces ``run_phase4.py`` + ``run_phase5.py`` + ``run_phase6.py`` of the
recovered tree (three near-duplicate runners with inconsistent import paths and
a per-dataset/per-model registry that contradicted the manuscript).  One runner,
one manifest, one bundle cache, parallel workers.

Phases::

    --phase floor   # content floor per (dataset, seed)
    --phase diag    # realized split / candidate-pool scale per (dataset, seed)
    --phase tune    # grid search on the frozen tuning seed (validation MRR)
    --phase select  # freeze the config; --per-model tunes each cell separately
    --phase final   # train the frozen config on every final seed
    --phase all     # tune, select, final

Two notes on the design that follow from the manuscript text:

* the tuning grid varies hidden size and dropout but not the learning rate.  The
  default is a **per-cell** selection (each cell takes the grid point with the
  best validation MRR), because that is what the published tables show: the
  config column of Table~\ref{tab:main} differs from cell to cell (c1/c2/c3/c4),
  which also means dropout is *not* identical across cells, contradicting the
  sentence "learning-rate and dropout choices are identical across cells" in the
  manuscript.  ``--shared`` freezes one config per dataset instead, for the
  robustness check the review asks for.
* every training job is independent (own seed, own checkpoint), so the final
  phase parallelises over jobs; each job writes its own JSON and is skipped on
  rerun (resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PROJECT = ROOT.parent
MANIFEST = json.loads((PROJECT / "MANIFEST.json").read_text(encoding="utf-8"))
# ``RESULTS_ROOT`` lets a replication write next to (never on top of) the
# canonical results, e.g. testing a different checkpoint policy.
RESULTS_DIR = Path(os.environ.get("RESULTS_ROOT", PROJECT / "results"))


def _seed_list(dataset: str) -> list[int]:
    seeds = MANIFEST["final_seeds"]
    return list(seeds[dataset]) if isinstance(seeds, dict) else list(seeds)


def _all_models() -> list[str]:
    return MANIFEST["models"]["cells"] + MANIFEST["models"]["baselines"]


def _canonical(model: str) -> str:
    from gated import canonical

    return canonical(model)


# --------------------------------------------------------------------------- #
# workers                                                                      #
# --------------------------------------------------------------------------- #
def _worker_train(job: dict) -> dict:
    """Train one (dataset, model, seed) job.  Runs in a subprocess."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from content_floor import build_bundle
    from train3 import train_model

    bundle = build_bundle(job["dataset"], job["seed"])
    t0 = time.time()
    record = train_model(
        bundle["graph"], bundle["split"], bundle["fit_window"],
        bundle["features"], bundle["descriptors"], bundle["edge_time"],
        job["model"], job["config"], seed=job["seed"],
        max_epochs=job["max_epochs"], patience=job["patience"],
        negatives_per_positive=job["negatives"], eval_every=job["eval_every"],
    )
    payload = {
        "dataset": job["dataset"],
        "model": _canonical(job["model"]),
        # the registry (code) name, not the CLI spelling: the manuscript uses
        # rcgat_sym while the registry uses ragat_sym, and downstream tools key
        # files by the registry name (a run launched with the paper alias used to
        # write "rcgat_sym" here and silently disappear from analyses).
        "registry_model": _canonical(job["model"]),
        "seed": job["seed"],
        "grid_id": job["grid_id"],
        "config": job["config"],
        "floor": bundle["floor"]["test"],
        "floor_val": bundle["floor"]["val"],
        "graph_built_at": bundle.get("graph_built_at"),
        "fit_window_metadata": bundle["fit_window"]["metadata"],
        "feature_metadata": bundle["feature_metadata"],
        "edge_time_edges": {k: int(v.shape[0]) for k, v in bundle["edge_time"].items()},
        "wall_seconds": round(time.time() - t0, 1),
        "phase": job["phase"],
        **{k: v for k, v in record.items() if k not in ("config", "val_mrr_curve")},
    }
    out = Path(job["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out)
    return payload


def _run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    if not jobs:
        return []
    if workers <= 1:
        return [_worker_train(j) for j in jobs]
    out: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker_train, j): j for j in jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                payload = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[error] {job['dataset']} {job['model']} seed={job['seed']}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            out.append(payload)
            print(f"[{job['phase']}] {payload['dataset']} {payload['model']} seed={payload['seed']} "
                  f"cfg={payload['grid_id']} val={payload['val']['mrr']:.4f} "
                  f"test={payload['test']['mrr']:.4f} h10={payload['test']['hits@10']:.4f} "
                  f"epoch={payload['best_epoch']} ({payload['wall_seconds']}s)", flush=True)
    return out


# --------------------------------------------------------------------------- #
# phases                                                                       #
# --------------------------------------------------------------------------- #
def run_floor(datasets: list[str]) -> None:
    from content_floor import build_bundle

    out_dir = RESULTS_DIR / "floor"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        for seed in [MANIFEST["tuning"]["tuning_seed"], *_seed_list(dataset)]:
            path = out_dir / f"floor_{dataset}_seed{seed}.json"
            if path.exists():
                continue
            t0 = time.time()
            bundle = build_bundle(dataset, seed)
            payload = {
                "dataset": dataset,
                "seed": seed,
                "floor": bundle["floor"],
                "fit_window_metadata": bundle["fit_window"]["metadata"],
                "feature_metadata": bundle["feature_metadata"],
                "seconds": round(time.time() - t0, 1),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[floor] {dataset} seed={seed} val={bundle['floor']['val']['mrr']:.4f} "
                  f"test={bundle['floor']['test']['mrr']:.4f} "
                  f"({payload['seconds']}s, {bundle['fit_window']['metadata']['test_nodes']} test nodes)",
                  flush=True)


def run_diag(datasets: list[str]) -> None:
    from content_floor import build_bundle

    out_dir = RESULTS_DIR / "diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        rows = []
        for seed in _seed_list(dataset):
            bundle = build_bundle(dataset, seed)
            md = bundle["fit_window"]["metadata"]
            floor = bundle["floor"]
            rows.append({
                "dataset": dataset, "seed": seed, "test_nodes": md["test_nodes"],
                "test_edges_all": md["test_edges"], "test_sources": floor["test"]["sources"],
                "positives": floor["test"]["positives"],
                "candidates_mean": floor["test"]["candidates_mean"],
                "floor_val_mrr": floor["val"]["mrr"], "floor_test_mrr": floor["test"]["mrr"],
                "fit_nodes": md["fit_nodes"], "val_nodes": md["val_nodes"],
                "cutoff": md.get("cutoff"),
            })
        payload = {
            "dataset": dataset,
            "rows": rows,
            "mean": {k: float(np.mean([r[k] for r in rows])) for k in
                     ("test_nodes", "test_sources", "positives", "candidates_mean", "floor_test_mrr")},
        }
        (out_dir / f"diag_{dataset}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[diag] {dataset} mean = {json.dumps(payload['mean'])}", flush=True)

# --------------------------------------------------------------------------- #
# tuning / selection                                                           #
# --------------------------------------------------------------------------- #
def run_tuning(datasets: list[str], models: list[str], workers: int = 1) -> None:
    grid = MANIFEST["tuning"]["grid"]
    seed = MANIFEST["tuning"]["tuning_seed"]
    out_dir = RESULTS_DIR / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict] = []
    for dataset in datasets:
        for model in models:
            for cfg in grid:
                out_path = out_dir / f"tune_{dataset}_{_canonical(model)}_{cfg['id']}_seed{seed}.json"
                if out_path.exists():
                    continue
                jobs.append({
                    "dataset": dataset, "model": model, "seed": seed, "grid_id": cfg["id"],
                    "config": cfg, "out_path": str(out_path), "phase": "tune",
                    "max_epochs": MANIFEST["training"]["epochs_max_tuning"],
                    "patience": MANIFEST["training"]["patience_tuning"],
                    "negatives": MANIFEST["training"]["negatives_per_positive"],
                    "eval_every": MANIFEST["training"]["eval_every"],
                })
    for dataset in datasets:
        for model in models:
            aggregate_tuning(dataset, model, grid, seed)
    print(f"[tune] {len(jobs)} jobs on tuning seed {seed} (workers={workers})", flush=True)
    for payload in _run_jobs(jobs, workers):
        aggregate_tuning(payload["dataset"], payload["registry_model"], grid, seed)
        aggregate_tuning(payload["dataset"], payload["model"], grid, seed)


def aggregate_tuning(dataset: str, model: str, grid: list[dict], seed: int) -> None:
    """Rebuild ``tuning_<dataset>_<model>.json`` from the per-config job files.

    Aggregating from disk (instead of incrementally in the parent) keeps the
    phase resumable: a killed run keeps every finished config, and a rerun
    re-aggregates whatever is present.
    """
    out_dir = RESULTS_DIR / "tuning"
    entries = []
    floor_val = None
    for cfg in grid:
        path = out_dir / f"tune_{dataset}_{_canonical(model)}_{cfg['id']}_seed{seed}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries.append({k: v for k, v in payload.items() if k != "config"})
        floor_val = payload.get("floor_val")
    if not entries:
        return
    (out_dir / f"tuning_{dataset}_{_canonical(model)}.json").write_text(json.dumps({
        "dataset": dataset, "model": _canonical(model), "tuning_seed": seed,
        "grid": grid, "entries": sorted(entries, key=lambda e: e["grid_id"]),
        "floor": floor_val,
    }, indent=2), encoding="utf-8")


def _tuning_entries(dataset: str, model: str) -> list[dict]:
    path = RESULTS_DIR / "tuning" / f"tuning_{dataset}_{_canonical(model)}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def select_configs(datasets: list[str], models: list[str], shared: bool = False) -> dict:
    grid = MANIFEST["tuning"]["grid"]
    by_id = {c["id"]: c for c in grid}
    cells = MANIFEST["models"]["cells"]
    path = RESULTS_DIR / "selections.json"
    selections: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    selections["mode"] = "shared" if shared else "per_model"
    for dataset in datasets:
        per_model: dict[str, dict] = {}
        for model in models:
            entries = _tuning_entries(dataset, model)
            if not entries:
                # No tuning results under this RESULTS_ROOT (a replication run):
                # carry the config already frozen for this cell over, so the
                # replication uses exactly the canonical hyper-parameters.
                carried = (selections.get(dataset, {}).get("used") or {}).get(_canonical(model))
                if carried:
                    per_model[_canonical(model)] = {"grid_id": carried, "carried_over": True}
                continue
            best = sorted(entries, key=lambda e: (-e["val"]["mrr"], e["grid_id"]))[0]
            per_model[_canonical(model)] = {
                "grid_id": best["grid_id"],
                "tuning_val_mrr": best["val"]["mrr"],
                "tuning_test_mrr": best["test"]["mrr"],
                "param_count": best.get("param_count"),
            }
        # shared config = best mean validation MRR over the four model cells
        shared_id, shared_mean = None, None
        for cfg in grid:
            vals = [_tuning_entries(dataset, m) for m in cells]
            vals = [[e["val"]["mrr"] for e in es if e["grid_id"] == cfg["id"]] for es in vals]
            flat = [v for vs in vals for v in vs]
            if len(flat) != len(cells):
                continue
            m = float(np.mean(flat))
            if shared_mean is None or m > shared_mean:
                shared_id, shared_mean = cfg["id"], m
        chosen = shared_id or (next(iter(per_model.values()))["grid_id"] if per_model else None)
        selections[dataset] = {
            "shared_grid_id": chosen,
            "shared_mean_cell_val_mrr": shared_mean,
            "config": by_id.get(chosen),
            "per_model_best": per_model,
            # keys are canonicalised: the CLI accepts the manuscript's display
            # names (rcgat_sym) while the run files use the code name (ragat_sym),
            # and a raw key here silently made run_final pick gid=None.
            "used": {_canonical(m): (chosen if shared else (per_model.get(_canonical(m)) or {}).get("grid_id"))
                     for m in models},
        }
        print(f"[select] {dataset}: mode={'shared' if shared else 'per_model'} chosen={chosen} "
              f"(mean cell val MRR over grid {shared_mean:.4f})" if shared_mean is not None
              else f"[select] {dataset}: chosen={chosen}", flush=True)
    path.write_text(json.dumps(selections, indent=2), encoding="utf-8")
    return selections


def run_final(datasets: list[str], models: list[str], workers: int = 1, shared: bool = False) -> None:
    selections = select_configs(datasets, models, shared=shared)
    out_dir = RESULTS_DIR / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict] = []
    for dataset in datasets:
        for model in models:
            canonical = _canonical(model)
            gid = selections[dataset]["used"].get(canonical)
            cfg = next(c for c in MANIFEST["tuning"]["grid"] if c["id"] == gid)
            for seed in _seed_list(dataset):
                out_path = out_dir / f"run_{dataset}_{canonical}_seed{seed}.json"
                if out_path.exists():
                    continue
                jobs.append({
                    "dataset": dataset, "model": model, "seed": seed, "grid_id": gid,
                    "config": cfg, "out_path": str(out_path), "phase": "final",
                    "max_epochs": MANIFEST["training"]["epochs_max_final"],
                    "patience": MANIFEST["training"]["patience_final"],
                    "negatives": MANIFEST["training"]["negatives_per_positive"],
                    "eval_every": MANIFEST["training"]["eval_every"],
                })
    print(f"[final] {len(jobs)} jobs, workers={workers}", flush=True)
    if jobs:
        # build the bundle cache once, in the parent, so workers never race on it
        from content_floor import build_bundle

        for dataset in sorted({j["dataset"] for j in jobs}):
            for seed in sorted({j["seed"] for j in jobs if j["dataset"] == dataset}):
                build_bundle(dataset, seed)
    _run_jobs(jobs, workers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["floor", "diag", "tune", "select", "final", "all"])
    ap.add_argument("--datasets", nargs="+", default=MANIFEST["datasets"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shared", action="store_true",
                    help="freeze one config per dataset (best mean validation MRR over the four cells) "
                         "instead of the default per-cell selection, which is what the published "
                         "tables show (the config column differs from cell to cell)")
    args = ap.parse_args()
    models = args.models or _all_models()
    shared = bool(args.shared)
    if args.phase == "floor":
        run_floor(args.datasets)
    elif args.phase == "diag":
        run_diag(args.datasets)
    elif args.phase == "tune":
        run_tuning(args.datasets, models, args.workers)
    elif args.phase == "select":
        select_configs(args.datasets, models, shared=shared)
    elif args.phase == "final":
        run_final(args.datasets, models, args.workers, shared=shared)
    else:
        run_tuning(args.datasets, models, args.workers)
        run_final(args.datasets, models, args.workers, shared=shared)
    return 0


if __name__ == "__main__":
    sys.exit(main())
