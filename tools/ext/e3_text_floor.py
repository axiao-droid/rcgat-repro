"""E3a -- is the content floor weak because of its text representation?

The manuscript's strongest claim is that a *content-only* ranker is competitive
with graph structure.  That claim is only as good as the content representation
behind it, so this script re-computes the floor under stronger text pipelines,
holding the protocol fixed (same splits, same candidate pool, same leakage
filter, same per-source MRR):

  ``baseline``   the frozen recipe: word TF-IDF (20k features, english stopwords,
                 sublinear) -> SVD 300 -> L2, fitted on the fit window only
  ``char35``     char_wb 3-5 gram TF-IDF (50k features) -> SVD 300
  ``union``      word (20k) hstacked with char_wb 3-5 (30k) -> SVD 300
  ``big_word``   word TF-IDF, 60k features, no stopword list, min_df 1 -> SVD 300

Training-free: it is the same cosine ranking the floor uses, only the features
change.  Results are written per (dataset, seed, recipe).

Usage:
    python3 tools/ext/e3_text_floor.py --datasets npm --seeds 101 102 103 104 105
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from content_floor import TFIDF_TOKEN_PATTERN, build_bundle, load_graph   # noqa: E402
from ranking import evaluate_ranking                                     # noqa: E402

OUT = ROOT / "results_ext" / "e3_text_floor"
DIM = 300

RECIPES = {
    "baseline": [
        ("word", dict(max_features=20000, sublinear_tf=True, stop_words="english",
                      token_pattern=TFIDF_TOKEN_PATTERN, min_df=1)),
    ],
    "char35": [
        ("char", dict(analyzer="char_wb", ngram_range=(3, 5), max_features=50000,
                      sublinear_tf=True, min_df=1)),
    ],
    "union": [
        ("word", dict(max_features=20000, sublinear_tf=True, stop_words="english",
                      token_pattern=TFIDF_TOKEN_PATTERN, min_df=1)),
        ("char", dict(analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
                      sublinear_tf=True, min_df=1)),
    ],
    "big_word": [
        ("word", dict(max_features=60000, sublinear_tf=True, min_df=1,
                      token_pattern=TFIDF_TOKEN_PATTERN)),
    ],
}


def embed(recipe: str, fit_texts: list[str], all_texts: list[str]) -> torch.Tensor:
    blocks_fit, blocks_all = [], []
    for _, kwargs in RECIPES[recipe]:
        vec = TfidfVectorizer(**kwargs)
        blocks_fit.append(vec.fit_transform(fit_texts))
        blocks_all.append(vec.transform(all_texts))
    x_fit = sp.hstack(blocks_fit).tocsr() if len(blocks_fit) > 1 else blocks_fit[0]
    x_all = sp.hstack(blocks_all).tocsr() if len(blocks_all) > 1 else blocks_all[0]
    svd = TruncatedSVD(n_components=DIM, n_iter=10, random_state=42)
    svd.fit(x_fit)
    z = svd.transform(x_all).astype(np.float32)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    return torch.tensor(z, dtype=torch.float32)


def run(dataset: str, seed: int, recipe: str, test_only: bool = False) -> dict:
    bundle = build_bundle(dataset, seed)
    graph, split = bundle["graph"], bundle["split"]
    raw = load_graph(dataset)
    texts = [t if t else " " for t in raw["texts"]]
    fit_mask = split["fit_window"]["core_mask"].numpy()
    t0 = time.time()
    z = embed(recipe, [t for t, m in zip(texts, fit_mask) if m], texts)

    def scorer(pairs):
        left = z.index_select(0, pairs[0])
        right = z.index_select(0, pairs[1])
        return (left * right).sum(dim=-1).numpy()

    stages = ("test",) if test_only else ("val", "test")
    out = {stage: evaluate_ranking(graph, split, scorer, stage=stage) for stage in stages}
    out["recipe"] = recipe
    out["dataset"] = dataset
    out["seed"] = seed
    out["dim"] = DIM
    out["seconds"] = round(time.time() - t0, 1)
    out["frozen_floor_test"] = float(bundle["floor"]["test"]["mrr"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["npm", "maven"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--recipes", nargs="+", default=list(RECIPES))
    ap.add_argument("--test-only", action="store_true",
                    help="skip the validation-stage evaluation (the floor is not tuned)")
    args = ap.parse_args()
    default_seeds = {"npm": [101, 102, 103, 104, 105], "maven": [121, 122, 123, 124, 125]}

    OUT.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for seed in (args.seeds or default_seeds[dataset]):
            for recipe in args.recipes:
                path = OUT / f"cell_{dataset}_seed{seed}_{recipe}.json"
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                else:
                    payload = run(dataset, seed, recipe, args.test_only)
                    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"[e3] {dataset} seed={seed} {recipe:9s} "
                      f"test={payload['test']['mrr']:.4f} "
                      f"(frozen {payload['frozen_floor_test']:.4f}) "
                      f"val={payload.get('val', {}).get('mrr', float('nan')):.4f} "
                      f"{payload['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
