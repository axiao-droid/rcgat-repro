"""DIAGNOSTIC -- is the content floor computed under the same pool as the models?

Not a result.  It computes the content floor under *both* candidate-pool readings
for every seed and cross-tabulates them against the ten-seed model runs that are
already on disk, because the manuscript's pattern ("every non-time cell sits
below the content floor on npm") has a mechanical explanation that costs nothing
to test:

  * the published floor (npm 0.1565) matches our floor under reading A
    (pool = every node outside the test window, 0.1559) to 0.0006;
  * but the published *model* numbers are 0.035 BELOW that floor.

A structure-using model can only be *below* a content-only ranker if the two were
not ranked in the same pool.  Reading B (pool = every node except the source)
includes the intra-window targets, which for content are easy but for a GNN that
has never seen those nodes in the fit graph are uninformative; the floor rises to
0.169 on npm.  Scoring the models under A while crediting the floor under B puts
two of the three npm cells below the floor on this seed -- the published sign
pattern.

Output: ``logs/floor_pool_readings.json`` plus a printed cross-table.

Usage::

    OMP_NUM_THREADS=1 python tools/diag_floor_pool_readings.py --dataset npm
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

from content_floor import build_bundle  # noqa: E402
from ranking import evaluate_ranking  # noqa: E402

MANIFEST = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))


def cosine_scorer(features: torch.Tensor):
    def score(pairs: torch.Tensor) -> np.ndarray:
        left = features.index_select(0, pairs[0])
        right = features.index_select(0, pairs[1])
        return (left * right).sum(dim=-1).numpy()
    return score


def floor_both(dataset: str, seed: int) -> dict:
    bundle = build_bundle(dataset, seed)
    cos = cosine_scorer(bundle["features"])
    split = bundle["split"]
    split_b = {**split, "core_mask": torch.ones_like(split["core_mask"])}
    out = {}
    for tag, sp in (("A", split), ("B", split_b)):
        out[tag] = {stage: evaluate_ranking(bundle["graph"], sp, cos, stage=stage)
                    for stage in ("val", "test")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="npm")
    ap.add_argument("--out", default=str(ROOT / "logs" / "floor_pool_readings.json"))
    args = ap.parse_args()

    seeds = MANIFEST["final_seeds"][args.dataset]
    rows = []
    for seed in seeds:
        floors = floor_both(args.dataset, seed)
        rec = {"seed": seed, "floor": floors}
        run_dir = ROOT / "results" / "runs"
        for model in ("gat_dir", "ragat_sym", "gat_time", "ragat_time"):
            path = run_dir / f"run_{args.dataset}_{model}_seed{seed}.json"
            if path.exists():
                rec.setdefault("model_A", {})[model] = json.loads(path.read_text(encoding="utf-8"))["test"]["mrr"]
        rows.append(rec)
        fa, fb = floors["A"]["test"], floors["B"]["test"]
        print(f"seed {seed}: floor A={fa['mrr']:.4f} ({fa['sources']}/{fa['positives']})  "
              f"floor B={fb['mrr']:.4f} ({fb['sources']}/{fb['positives']})  "
              f"cands {fa['candidates_mean']:.0f} / {fb['candidates_mean']:.0f}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fa = np.array([r["floor"]["A"]["test"]["mrr"] for r in rows])
    fb = np.array([r["floor"]["B"]["test"]["mrr"] for r in rows])
    print(f"\nfloor A: {fa.mean():.4f} ± {fa.std(ddof=1):.4f}   floor B: {fb.mean():.4f} ± {fb.std(ddof=1):.4f}")

    print("\n=== matched reading (models and floor both under A) ===")
    print("| model | mean model A | mean floor A | mean Δ | seeds below floor |")
    print("|---|---|---|---|---|")
    best_val = {}
    for model in ("gat_dir", "ragat_sym", "gat_time", "ragat_time"):
        m = np.array([r["model_A"][model] for r in rows if model in r.get("model_A", {})])
        if m.size == 0:
            continue
        d = m - fa[:m.size]
        best_val[model] = float(d.mean())
        print(f"| {model} | {m.mean():.4f} | {fa[:m.size].mean():.4f} | {d.mean():+.4f} | "
              f"{int((d < 0).sum())}/{d.size} |")

    print("\n=== mixed reading (models under A, floor credited under B) ===")
    print("| model | mean model A | mean floor B | mean Δ | seeds below floor |")
    print("|---|---|---|---|---|")
    mixed = {}
    for model in ("gat_dir", "ragat_sym", "gat_time", "ragat_time"):
        m = np.array([r["model_A"][model] for r in rows if model in r.get("model_A", {})])
        if m.size == 0:
            continue
        d = m - fb[:m.size]
        mixed[model] = float(d.mean())
        print(f"| {model} | {m.mean():.4f} | {fb[:m.size].mean():.4f} | {d.mean():+.4f} | "
              f"{int((d < 0).sum())}/{d.size} |")
    print("\nThe mixed row is the diagnostic point: it is the only combination tried here that "
          "reproduces the manuscript's sign (cells below a content floor) on most seeds.")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
