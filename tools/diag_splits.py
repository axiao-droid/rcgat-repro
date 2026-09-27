"""Diagnose the node-date split against the published dataset statistics.

The published numbers that constrain the split are, for npm,

    4,962 nodes / 13,596 edges / avg degree 2.740024183796856
    max in/out degree 253 / 167, reciprocity 0.001
    test sources 234, positives 686, mean candidates 4,614
    node dates 2010-12-19 .. 2026-09-17, content floor 0.1565

and for Maven 1,484 / 5,482 / 3.6940700808625335 / 216 / 87 / 0.004,
94 sources, 352 positives, 1,376 candidates, 2021-04-19 .. 2026-09-21, floor 0.3377.

Mean candidates pins the split: with ``F_TEST = 0.07`` the test window holds
``round(0.07*4962) = 347`` nodes, and candidates = all nodes minus the test
window minus the source itself = 4,962 - 347 - 1 = 4,614, exactly the published
value.  This tool checks the rest of the constrained quantities under several
readings of "source" and of the node date, so that the reconstruction can be
judged rather than assumed.

Usage::

    python tools/diag_splits.py npm maven
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "data" / "graphs"
F_TEST, F_VAL, JIT = 0.07, 0.08, 0.05
SEED = 101

PUBLISHED = {
    "npm": dict(nodes=4962, edges=13596, avg=2.740024183796856, max_in=253, max_out=167,
                sources=234, positives=686, cand=4614, dmin="2010-12-19", dmax="2026-09-17", floor=0.1565),
    "maven": dict(nodes=1484, edges=5482, avg=3.6940700808625335, max_in=216, max_out=87,
                  sources=94, positives=352, cand=1376, dmin="2021-04-19", dmax="2026-09-21", floor=0.3377),
}


def load(dataset: str) -> tuple[list[dict], np.ndarray]:
    with gzip.open(GRAPHS / f"{dataset}_graph.json.gz", "rb") as fh:
        payload = json.loads(fh.read().decode())
    return payload["nodes"], np.array(payload["edges"], dtype=np.int64)


def days(nodes: list[dict], key: str) -> np.ndarray:
    d0 = np.datetime64("1970-01-01")
    out = []
    for n in nodes:
        raw = n.get(key) or n.get("date")
        out.append(float((np.datetime64(raw[:10]) - d0) // np.timedelta64(1, "D")))
    return np.array(out, dtype=float)


def split(order: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_test = max(1, round(F_TEST * n))
    rng = np.random.default_rng(seed * 7919 + 13)
    n_val = max(1, round(F_VAL * (1.0 + JIT * (2 * rng.random() - 1)) * n))
    start = n - n_test
    return order[: max(0, start - n_val)], order[max(0, start - n_val): start], order[start:]


def report(dataset: str) -> None:
    nodes, edges = load(dataset)
    n = len(nodes)
    src, tgt = edges[:, 0], edges[:, 1]
    print(f"\n=== {dataset} (ours) ===")
    print(f"nodes={n} edges={len(edges)} avg_out={len(edges)/n:.6f} "
          f"max_out={np.bincount(src, minlength=n).max()} max_in={np.bincount(tgt, minlength=n).max()}")
    pub = PUBLISHED[dataset]
    print(f"published: nodes={pub['nodes']} edges={pub['edges']} avg={pub['avg']:.6f} "
          f"max_out={pub['max_out']} max_in={pub['max_in']} "
          f"sources={pub['sources']} positives={pub['positives']} cand={pub['cand']}")

    for key, label in (("date", "node date"), ("date_first", "first publish (Maven only)")):
        if key == "date_first" and not any("date_first" in x for x in nodes):
            continue
        d = days(nodes, key)
        print(f"  {label}: span {np.datetime64(int(d.min()),'D')} .. {np.datetime64(int(d.max()),'D')}")
    for key, label in (("date", "node date"), ("date_first", "date_first")):
        if key == "date_first" and not any("date_first" in x for x in nodes):
            continue
        d = days(nodes, key)
        for seed in (101, 127):
            order = np.argsort(d, kind="stable")
            fit, val, test = split(order, n, seed)
            test_mask = np.zeros(n, bool)
            test_mask[test] = True
            # (a) sources = test nodes with >=1 out-edge at all
            any_out = np.zeros(n, bool)
            any_out[src] = True
            a = int((any_out & test_mask).sum())
            # (b) sources = test nodes with >=1 target inside the test window
            in_win = test_mask[tgt]
            out_win = np.zeros(n, bool)
            out_win[src[in_win]] = True
            b = int((out_win & test_mask).sum())
            # (c) positives = number of distinct (source, target) pairs with both in test window
            pairs = set(zip(src[in_win].tolist(), tgt[in_win].tolist()))
            cand = n - len(test) - 1
            print(f"  [{label} seed {seed}] fit={len(fit)} val={len(val)} test={len(test)} | "
                  f"sources_any_out={a} sources_target_in_window={b} "
                  f"positives_all_targets={int((out_win & test_mask).sum() and in_win.sum())} "
                  f"positives_window={len(pairs)} cand={cand}")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["npm"]:
        report(ds)
