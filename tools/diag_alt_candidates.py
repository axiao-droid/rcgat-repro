"""DIAGNOSTIC -- which candidate-pool reading produces "models below the floor"?

Not a result: it evaluates the test split to pick the checkpoint and is used only
to locate a discrepancy between the manuscript and the rebuilt pipeline.

``tools/diag_epoch_curve.py`` rules the checkpoint rule and the learned
temperature *out* as the cause of the manuscript's "every non-time cell sits
below the content floor on npm" (the curve never dwells below the floor, and the
temperature moves by 0.7 over 100 epochs).  That leaves the data/split layer, and
there is exactly one place in the protocol where the manuscript contradicts
itself:

  * its candidate count (4,614 = 4,962 - 347 - 1) requires the whole test window
    to be excluded from the pool -> the intra-window out-edges of the test
    sources are then *unrankable*, and the reported truth set (234 sources /
    686 positives ~= the 205 / 689 intra-window edges we measure) cannot be the
    ranked one (README sections 3.2 / 5.2);

  * but if the pool is instead "every node except the source" (reading B), those
    test-window targets *are* rankable -- and they are exactly the nodes the
    structural encoder cannot see: they enter the graph with content features
    only, no fit-window edges, so a GNN can only add noise to their ranking while
    a content-only ranker is unaffected.  That is a mechanism that can push a
    trained model *below* the content floor, which is what the manuscript
    reports.

So this tool trains the three npm cells under the production protocol (training
edges unchanged, checkpoint chosen on the reading-A validation split) and scores
the test split under both readings.  Findings are reported; the tool itself never
produces a number for the paper.

Usage::

    OMP_NUM_THREADS=1 python tools/diag_alt_candidates.py --dataset npm --seed 101 --epochs 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from content_floor import build_bundle  # noqa: E402
from gated import build_model  # noqa: E402
from ranking import evaluate_ranking  # noqa: E402
from train3 import _make_scorer, _sampled_pairs, set_seed  # noqa: E402

MANIFEST = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
GRID = {c["id"]: c for c in MANIFEST["tuning"]["grid"]}


def _cosine_scorer(features: torch.Tensor):
    def score(pairs: torch.Tensor) -> np.ndarray:
        left = features.index_select(0, pairs[0])
        right = features.index_select(0, pairs[1])
        return (left * right).sum(dim=-1).numpy()
    return score


def reading_b(split: dict) -> dict:
    """Copy of the split whose candidate pool is 'every node except the source'."""
    return {**split, "core_mask": torch.ones_like(split["core_mask"])}


def run(dataset: str, seed: int, model: str, epochs: int) -> dict:
    bundle = build_bundle(dataset, seed)
    graph, split_a, fw = bundle["graph"], bundle["split"], bundle["fit_window"]
    split_b = reading_b(split_a)
    feats, desc, etime = bundle["features"], bundle["descriptors"], bundle["edge_time"]
    grid_id = json.loads((ROOT / "results" / "selections.json").read_text(encoding="utf-8"))[dataset][
        "per_model_best"][model]["grid_id"]
    cfg = GRID[grid_id]
    eval_every = MANIFEST["training"]["eval_every"]

    cos = _cosine_scorer(feats)
    floors = {}
    for tag, sp in (("A", split_a), ("B", split_b)):
        floors[tag] = {stage: evaluate_ranking(graph, sp, cos, stage=stage) for stage in ("val", "test")}

    set_seed(seed)
    net = build_model(model, feats.shape[1], cfg)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]),
                           weight_decay=float(cfg.get("weight_decay", 1e-5)))
    loss_fn = nn.BCEWithLogitsLoss()
    core_idx = np.nonzero(fw["core_mask"].numpy())[0]
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    best_val, best_epoch = -1.0, -1

    for epoch in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        pairs, labels = _sampled_pairs(fw["fit_pos"], core_idx,
                                       MANIFEST["training"]["negatives_per_positive"],
                                       np.random.default_rng(seed * 100003 + epoch))
        z = net.encode(feats, fw["fit_mp_edges"], desc, etime["fit"])
        loss = loss_fn(net.score(z, pairs), labels)
        loss.backward()
        opt.step()
        if epoch % eval_every == 0 or epoch == 1:
            vm = evaluate_ranking(graph, split_a, _make_scorer(net, feats, split_a["mp_edges_val"],
                                                              desc, etime["val"]), stage="val")["mrr"]
            if vm > best_val + 1e-6:
                best_val, best_epoch = vm, epoch
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    out = {"dataset": dataset, "seed": seed, "model": model, "grid_id": grid_id,
           "epochs": epochs, "best_val_epoch": best_epoch, "floor": floors, "model_metrics": {}}
    for tag, sp in (("A", split_a), ("B", split_b)):
        out["model_metrics"][tag] = {
            stage: evaluate_ranking(graph, sp, _make_scorer(net, feats, sp["mp_edges_" + stage], desc, etime[stage]),
                                    stage=stage)
            for stage in ("val", "test")}
    for tag in ("A", "B"):
        f = floors[tag]["test"]
        m = out["model_metrics"][tag]["test"]
        print(f"  reading {tag}: candidates/pool rule={'window-excluded' if tag == 'A' else 'all-but-source'}  "
              f"floor={f['mrr']:.4f} ({f['sources']} src / {f['positives']} pos)  "
              f"model={m['mrr']:.4f}  diff={m['mrr'] - f['mrr']:+.4f}  "
              f"{'BELOW FLOOR' if m['mrr'] < f['mrr'] else 'above floor'}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="npm")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--models", nargs="+", default=["gat_dir", "ragat_sym", "gat_time"])
    ap.add_argument("--out", default=str(ROOT / "logs" / "diag_alt_candidates.json"))
    args = ap.parse_args()

    out = []
    for model in args.models:
        print(f"[{args.dataset} seed={args.seed} {args.epochs}ep] {model}", flush=True)
        out.append(run(args.dataset, args.seed, model, args.epochs))
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten {args.out}")
    print("\n=== summary (test MRR) ===")
    print("| model | floor A (window-excluded) | model A | ΔA | floor B (all-but-source) | model B | ΔB |")
    print("|---|---|---|---|---|---|---|")
    for r in out:
        fa, ma = r["floor"]["A"]["test"], r["model_metrics"]["A"]["test"]
        fb, mb = r["floor"]["B"]["test"], r["model_metrics"]["B"]["test"]
        print(f"| {r['model']} | {fa['mrr']:.4f} ({fa['sources']}/{fa['positives']}) | {ma['mrr']:.4f} | "
              f"{ma['mrr'] - fa['mrr']:+.4f} | {fb['mrr']:.4f} ({fb['sources']}/{fb['positives']}) | "
              f"{mb['mrr']:.4f} | {mb['mrr'] - fb['mrr']:+.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
