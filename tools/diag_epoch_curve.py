"""DIAGNOSTIC -- why are the manuscript's models *below* the content floor?

This is not a result and must never be reported as one: it evaluates the test
split at every second epoch, which is test-set model selection and therefore
leaks.  It exists only to answer one question that decides how the paper can be
revised:

    In the manuscript every cell except the two time cells sits BELOW the
    content floor on npm (gat_dir 0.1214 vs a floor of 0.1565, i.e. training
    *costs* 0.035 MRR).  In the rebuilt pipeline the same model, same loss, same
    budget lands 0.008 ABOVE the floor.  A 0.05 swing in the sign of the effect
    has to come from somewhere, and there are only three candidates:

      1. the checkpoint rule  -- if the original code selected the *last* epoch,
         or selected on a metric that does not track MRR, training would be
         allowed to run past its optimum and could end up below the floor;
      2. the data/split        -- if the original test targets were mostly nodes
         absent from the frozen graph, a structure-consuming encoder is worse
         than a content-only ranker, which is exactly what "below the floor"
         means;
      3. the scorer/temperature -- a diverging learned temperature flattens the
         ranking.

    (1) and (3) are visible in a per-epoch curve; if the curve is monotone and
    never crosses the floor, only (2) is left.

Output: ``logs/diag_epoch.json`` plus a printed table.

Usage::

    python tools/diag_epoch_curve.py --dataset npm --seed 101 --epochs 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
SELECTIONS = json.loads((ROOT / "results" / "selections.json").read_text(encoding="utf-8"))
GRID = {c["id"]: c for c in MANIFEST["tuning"]["grid"]}


def curve(dataset: str, seed: int, model: str, epochs: int, eval_every: int = 2) -> dict:
    bundle = build_bundle(dataset, seed)
    graph, split, fw = bundle["graph"], bundle["split"], bundle["fit_window"]
    feats, desc, etime = bundle["features"], bundle["descriptors"], bundle["edge_time"]
    gid = SELECTIONS[dataset]["per_model_best"][model]["grid_id"]
    cfg = GRID[gid]
    floor_test = bundle["floor"]["test"]["mrr"]
    floor_val = bundle["floor"]["val"]["mrr"]

    set_seed(seed)
    model_obj = build_model(model, feats.shape[1], cfg)
    opt = torch.optim.Adam(model_obj.parameters(), lr=float(cfg["lr"]),
                           weight_decay=float(cfg.get("weight_decay", 1e-5)))
    loss_fn = nn.BCEWithLogitsLoss()
    core_idx = np.nonzero(fw["core_mask"].numpy())[0]

    def ev(stage: str, epoch: int) -> dict:
        scorer = _make_scorer(model_obj, feats, split["mp_edges_" + stage], desc, etime[stage])
        return evaluate_ranking(graph, split, scorer, stage=stage)

    rows = [{"epoch": 0, "val": floor_val, "test": floor_test,
             "temperature": (model_obj.scale().item() if hasattr(model_obj, "scale") else None)}]
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model_obj.train()
        opt.zero_grad()
        pairs, labels = _sampled_pairs(fw["fit_pos"], core_idx,
                                       MANIFEST["training"]["negatives_per_positive"],
                                       np.random.default_rng(seed * 100003 + epoch))
        z = model_obj.encode(feats, fw["fit_mp_edges"], desc, etime["fit"])
        loss = loss_fn(model_obj.score(z, pairs), labels)
        loss.backward()
        opt.step()
        if epoch % eval_every == 0:
            v = ev("val", epoch)
            t = ev("test", epoch)
            rows.append({"epoch": epoch, "val": v["mrr"], "test": t["mrr"],
                         "temperature": (model_obj.scale().item() if hasattr(model_obj, "scale") else None),
                         "loss": float(loss.item())})
            print(f"    ep{epoch:3d} loss={loss.item():.4f} val={v['mrr']:.4f} "
                  f"test={t['mrr']:.4f} gain_vs_floor={t['mrr'] - floor_test:+.4f}", flush=True)

    tests = [r["test"] for r in rows[1:]]
    vals = [r["val"] for r in rows[1:]]
    best_i = int(np.argmax(vals))
    last_i = int(np.argmin(tests))
    return {
        "dataset": dataset, "seed": seed, "model": model, "grid_id": gid,
        "floor_val": floor_val, "floor_test": floor_test,
        "rows": rows,
        "best_val_test": tests[best_i], "best_val_epoch": rows[1 + best_i]["epoch"],
        "last_epoch_test": tests[-1],
        "min_test": tests[last_i], "min_test_epoch": rows[1 + last_i]["epoch"],
        "epochs_below_floor": int(sum(1 for t in tests if t < floor_test)),
        "epochs_above_floor": int(sum(1 for t in tests if t > floor_test)),
        "temperature_first": rows[1]["temperature"], "temperature_last": rows[-1]["temperature"],
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="npm")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--models", nargs="+", default=["gat_dir", "ragat_sym", "gat_time"])
    args = ap.parse_args()

    out = []
    for model in args.models:
        print(f"[{args.dataset} seed={args.seed}] {model}", flush=True)
        rec = curve(args.dataset, args.seed, model, args.epochs)
        out.append(rec)
        print(f"  floor={rec['floor_test']:.4f} | best-val(ep{rec['best_val_epoch']}) test="
              f"{rec['best_val_test']:.4f} | last-epoch test={rec['last_epoch_test']:.4f} | "
              f"min test={rec['min_test']:.4f} @ep{rec['min_test_epoch']} | "
              f"epochs below floor: {rec['epochs_below_floor']}/{len(rec['rows']) - 1}\n", flush=True)

    path = ROOT / "logs" / "diag_epoch.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written {path}")
    print("\n=== summary ===")
    print("| model | floor | ep0 | best-val test | last test | min test | #ep<floor |")
    print("|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['model']} | {r['floor_test']:.4f} | {r['rows'][0]['test']:.4f} | "
              f"{r['best_val_test']:.4f} (ep{r['best_val_epoch']}) | {r['last_epoch_test']:.4f} | "
              f"{r['min_test']:.4f} (ep{r['min_test_epoch']}) | {r['epochs_below_floor']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
