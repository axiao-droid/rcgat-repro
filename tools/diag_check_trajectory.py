"""Sanity check: does tools/diag_epoch_curve.py follow the same trajectory as
the production trainer (``train3.train_model``)?

The diagnostic re-implements the training loop inline so that it can log the
*test* split every second epoch.  If the two trajectories differ, the diagnostic
is measuring a different model and its conclusion ("the curve never dwells below
the content floor") would not transfer to the reported runs.  This script trains
the same (dataset, seed, model, grid point) with the production trainer for the
same number of epochs and compares the validation curves.

Usage::

    OMP_NUM_THREADS=1 python tools/diag_check_trajectory.py --dataset npm --seed 101 --model gat_dir --epochs 74
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

from content_floor import build_bundle  # noqa: E402
from train3 import train_model  # noqa: E402

MANIFEST = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
GRID = {c["id"]: c for c in MANIFEST["tuning"]["grid"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="npm")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--model", default="gat_dir")
    ap.add_argument("--grid-id", default=None)
    ap.add_argument("--epochs", type=int, default=74)
    ap.add_argument("--diag-json", default=str(ROOT / "logs" / "diag_epoch.json"))
    args = ap.parse_args()

    sel = json.loads((ROOT / "results" / "selections.json").read_text(encoding="utf-8"))
    gid = args.grid_id or sel[args.dataset]["per_model_best"][args.model]["grid_id"]
    cfg = GRID[gid]
    eval_every = MANIFEST["training"]["eval_every"]
    bundle = build_bundle(args.dataset, args.seed)
    out = train_model(
        bundle["graph"], bundle["split"], bundle["fit_window"], bundle["features"],
        bundle["descriptors"], bundle["edge_time"], args.model, cfg, args.seed,
        max_epochs=args.epochs, patience=args.epochs + 1,
        negatives_per_positive=MANIFEST["training"]["negatives_per_positive"],
        eval_every=eval_every,
    )
    # ``val_mrr_curve`` is [epoch 0 (the floor anchor), epoch 1, then every
    # ``eval_every``-th epoch]; an off-by-one here silently compares epoch k with
    # k+1 and reports a spurious "DIFFERENT".
    seen = [0, 1] + list(range(2, args.epochs + 1, eval_every))
    assert len(seen) == len(out["val_mrr_curve"]), (len(seen), len(out["val_mrr_curve"]))
    curve = dict(zip(seen, out["val_mrr_curve"]))
    floor_test = bundle["floor"]["test"]["mrr"]

    print(f"grid={gid}  trainer val: " +
          "  ".join(f"ep{e}={curve[e]:.4f}" for e in (1, 20, 40, 100) if e in curve))
    print(f"trainer  val={out['val']['mrr']:.4f}  test={out['test']['mrr']:.4f}  "
          f"floor={floor_test:.4f}  best_epoch={out['best_epoch']}  epochs_run={out['epochs_run']}")

    diag_path = Path(args.diag_json)
    if not diag_path.exists():
        print(f"{diag_path} missing; only the trainer side was run")
        return 0
    rec = next((r for r in json.loads(diag_path.read_text(encoding="utf-8"))
                if r["model"] == args.model), None)
    if rec is None:
        print(f"no {args.model} record in {diag_path}; only the trainer side was run")
        return 0
    dcurve = {r["epoch"]: r["val"] for r in rec["rows"]}
    shared = sorted(set(curve) & set(dcurve))
    diffs = [abs(curve[e] - dcurve[e]) for e in shared]
    print(f"compared {len(shared)} epochs  max|Δval| = {max(diffs):.6g}  "
          f"trainer ep{shared[-1]}={curve[shared[-1]]:.6f}  diag={dcurve[shared[-1]]:.6f}")
    print("IDENTICAL" if max(diffs) < 1e-9
          else "DIFFERENT -- the diagnostic does not follow the production trainer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
