"""Gate effect *within a fixed config* (tuning-seed check).

The published gate contrast pairs ``rcgat_sym`` against ``gat_dir`` across seeds,
but -- because the manuscript selects the hyper-parameters per cell -- the two
members of a pair often use *different* grid points, so the contrast mixes the
gate's effect with the config choice.  The tuning phase trains both models at all
four grid points on the tuning seed, which gives a config-matched view for free:
for each grid point, ``rcgat_sym`` minus ``gat_dir`` (and likewise the time pair).

Usage::

    python tools/gate_within_config.py [results_dir]

``results_dir`` defaults to ``results`` (pass ``results_alt_fixedwindow`` to look
at the previous build).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    tuning = results / "tuning"
    for dataset in ("npm", "maven"):
        for a, b in (("ragat_sym", "gat_dir"), ("ragat_time", "gat_time")):
            fa, fb = tuning / f"tuning_{dataset}_{a}.json", tuning / f"tuning_{dataset}_{b}.json"
            if not (fa.exists() and fb.exists()):
                continue
            ea = {e["grid_id"]: e for e in json.loads(fa.read_text(encoding="utf-8"))["entries"]}
            eb = {e["grid_id"]: e for e in json.loads(fb.read_text(encoding="utf-8"))["entries"]}
            print(f"## {dataset}: {a} - {b} (tuning seed, identical config on both sides)")
            print(f"| grid | val {a} | val {b} | d val | test {a} | test {b} | d test |")
            print("|---|---|---|---|---|---|---|")
            diffs_v, diffs_t = [], []
            for gid in sorted(set(ea) & set(eb)):
                va, vb = ea[gid]["val"]["mrr"], eb[gid]["val"]["mrr"]
                ta, tb = ea[gid]["test"]["mrr"], eb[gid]["test"]["mrr"]
                diffs_v.append(va - vb)
                diffs_t.append(ta - tb)
                print(f"| {gid.split('_')[0]} | {va:.4f} | {vb:.4f} | {va - vb:+.4f} | "
                      f"{ta:.4f} | {tb:.4f} | {ta - tb:+.4f} |")
            if diffs_v:
                print(f"\nmean delta val = {sum(diffs_v) / len(diffs_v):+.4f} "
                      f"(positive in {sum(1 for d in diffs_v if d > 0)}/{len(diffs_v)} grid points); "
                      f"mean delta test = {sum(diffs_t) / len(diffs_t):+.4f} "
                      f"(positive in {sum(1 for d in diffs_t if d > 0)}/{len(diffs_t)})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
