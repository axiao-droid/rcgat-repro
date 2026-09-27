"""Check that the vectorised ``evaluate_ranking`` is numerically identical to the
original per-element loop (``tools/ranking_loop_reference.py``).

Usage:  python3 tools/test_ranking_equiv.py [dataset] [seed]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from content_floor import build_bundle  # noqa: E402
from ranking import evaluate_ranking  # noqa: E402
from ranking_loop_reference import evaluate_ranking as reference  # noqa: E402

dataset = sys.argv[1] if len(sys.argv) > 1 else "npm"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
bundle = build_bundle(dataset, seed)
z = bundle["features"]
rng = np.random.default_rng(0)
# a scorer with genuinely many ties (the floor-like cosine of a random projection)
w = torch.tensor(rng.normal(size=(z.shape[1], 32)).astype(np.float32))


def scorer(pairs):
    left = z.index_select(0, pairs[0]) @ w
    right = z.index_select(0, pairs[1]) @ w
    return (left * right).sum(dim=-1).numpy()


ok = True
for stage in ("val", "test"):
    a = evaluate_ranking(bundle["graph"], bundle["split"], scorer, stage=stage)
    b = reference(bundle["graph"], bundle["split"], scorer, stage=stage)
    same = all(abs(a[k] - b[k]) < 1e-12 for k in a)
    ok &= same
    print(f"{dataset} seed={seed} {stage}: fast={a}")
    print(f"{dataset} seed={seed} {stage}: ref ={b}")
    print(f"  identical: {same}")
print("EQUIVALENT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
