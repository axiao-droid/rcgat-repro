"""Profile a few training epochs to find the loop's real hot spots.

Usage:  OMP_NUM_THREADS=1 python3 tools/profile_train.py [dataset] [model] [epochs]
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("PROFILE_THREADS", "1")))

from content_floor import build_bundle  # noqa: E402
from train3 import train_model  # noqa: E402

dataset = sys.argv[1] if len(sys.argv) > 1 else "npm"
model = sys.argv[2] if len(sys.argv) > 2 else "ragat_sym"
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 6

bundle = build_bundle(dataset, 11)
cfg = {"id": "prof", "hidden": 64, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-5, "heads": 4}

pr = cProfile.Profile()
t0 = time.time()
pr.enable()
rec = train_model(bundle["graph"], bundle["split"], bundle["fit_window"], bundle["features"],
                  bundle["descriptors"], bundle["edge_time"], model, cfg, seed=11,
                  max_epochs=epochs, patience=99, negatives_per_positive=10, eval_every=2)
pr.disable()
wall = time.time() - t0
print(f"\n{dataset}/{model}: {rec['epochs_run']} epochs in {wall:.1f}s "
      f"= {wall / max(1, rec['epochs_run']):.2f}s/epoch")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
print(s.getvalue())
