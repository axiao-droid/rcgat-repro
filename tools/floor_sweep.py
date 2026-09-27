"""Calibrate the content-floor text recipe against the published floor values.

The manuscript fixes the floor only up to "TF-IDF fitted on training texts,
truncated SVD to 300 dimensions, L2-normalized", with the weight formula
``w = f * log(N / (1 + df))``.  That is not enough to reproduce a number: the
token pattern, stop-word list, feature cap, sublinear term frequency and text
fields all move the floor by several points.  This tool therefore evaluates a
grid of recipes on the frozen split and reports each floor next to the published
values (npm 0.1565, Maven 0.3377), so the recipe used by the pipeline can be
chosen on evidence instead of guessed.

    python tools/floor_sweep.py npm            # all variants, 3 seeds
    python tools/floor_sweep.py npm --seeds 101 105 109 --full
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
F_TEST, F_VAL, JIT = 0.07, 0.08, 0.05
PUBLISHED = {"npm": 0.1565, "maven": 0.3377}
_CLEAN = re.compile(r"\s+")

VARIANTS: dict[str, dict] = {
    "v1_cur_pipe":      dict(readme=1500, tok=r"(?u)\b\w+\b", stop="english", feat=20000, sub=True, mindf=1),
    "v2_sklearn_tok":   dict(readme=1500, tok=r"(?u)\b\w\w+\b", stop="english", feat=20000, sub=True, mindf=1),
    "v3_readme500":     dict(readme=500, tok=r"(?u)\b\w\w+\b", stop="english", feat=20000, sub=True, mindf=1),
    "v4_no_readme":     dict(readme=0, tok=r"(?u)\b\w\w+\b", stop="english", feat=20000, sub=True, mindf=1),
    "v5_plain_tf":      dict(readme=1500, tok=r"(?u)\b\w\w+\b", stop="english", feat=20000, sub=False, mindf=1),
    "v6_no_stopwords":  dict(readme=1500, tok=r"(?u)\b\w+\b", stop=None, feat=20000, sub=True, mindf=1),
    "v7_feat5000":      dict(readme=1500, tok=r"(?u)\b\w+\b", stop="english", feat=5000, sub=True, mindf=1),
    "v8_mindf2":        dict(readme=1500, tok=r"(?u)\b\w+\b", stop="english", feat=20000, sub=True, mindf=2),
    "v9_no_smooth_idf": dict(readme=1500, tok=r"(?u)\b\w+\b", stop="english", feat=20000, sub=True, mindf=1, smooth=False),
    "v10_desc_only":    dict(readme=-1, tok=r"(?u)\b\w+\b", stop="english", feat=20000, sub=True, mindf=1),
    "v11_svd100":       dict(readme=1500, tok=r"(?u)\b\w+\b", stop="english", feat=20000, sub=True, mindf=1, k=100),
    "v12_min_df1_nostop_tok": dict(readme=1500, tok=r"(?u)[a-zA-Z][a-zA-Z0-9_+#.-]*", stop="english", feat=20000, sub=True, mindf=1),
}


def load_graph(dataset: str):
    with gzip.open(ROOT / "data" / "graphs" / f"{dataset}_graph.json.gz", "rb") as fh:
        payload = json.loads(fh.read().decode())
    return payload["nodes"], np.array(payload["edges"], np.int64)


def npm_texts(raw_dir: Path) -> dict[str, dict]:
    """name / description / keywords / readme for every fetched package."""
    out: dict[str, dict] = {}
    for path in raw_dir.glob("*.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        name = doc.get("name")
        if not name:
            continue
        kw = doc.get("keywords") or []
        if isinstance(kw, str):
            kw = [kw]
        out[name] = {
            "name": name,
            "desc": _CLEAN.sub(" ", str(doc.get("description") or "")).strip(),
            "kw": " ".join(str(k) for k in kw)[:400],
            "readme": _CLEAN.sub(" ", str(doc.get("readme") or "")),
        }
    return out


def build_text(rec: dict, readme: int) -> str:
    if readme == -1:  # description only
        return rec["desc"]
    parts = [rec["name"], rec["desc"], rec["kw"]]
    if readme > 0:
        parts.append(rec["readme"][:readme])
    return " ".join(p for p in parts if p)


def split(days: np.ndarray, seed: int):
    n = len(days)
    order = np.argsort(days, kind="stable")
    n_test = max(1, round(F_TEST * n))
    rng = np.random.default_rng(seed * 7919 + 13)
    n_val = max(1, round(F_VAL * (1.0 + JIT * (2 * rng.random() - 1)) * n))
    start = n - n_test
    return order[: max(0, start - n_val)], order[max(0, start - n_val): start], order[start:]


def floor_for(Z: np.ndarray, src: np.ndarray, tgt: np.ndarray, test: np.ndarray, n: int) -> tuple[float, int, int]:
    tmask = np.zeros(n, bool)
    tmask[test] = True
    sel = tmask[src]
    pool = np.setdiff1d(np.arange(n), test)          # leakage-safe candidate pool
    pos: dict[int, list[int]] = {}
    for a, b in zip(src[sel].tolist(), tgt[sel].tolist()):
        if not tmask[b]:                              # only rankable targets count
            pos.setdefault(a, []).append(b)
    rr, npos = [], 0
    for a, targets in pos.items():
        keep = pool[pool != a]
        sc = Z[keep] @ Z[a]
        order = np.argsort(-sc, kind="stable")
        rank = {int(keep[i]): r + 1 for r, i in enumerate(order)}
        reach = [rank[t] for t in targets if t in rank]
        if reach:
            rr.append(1.0 / min(reach))
            npos += len(reach)
    return (float(np.mean(rr)) if rr else 0.0), len(rr), npos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="npm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 105, 109])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()

    dataset = args.dataset
    nodes, edges = load_graph(dataset)
    n = len(nodes)
    names = [x["name"] for x in nodes]
    d0 = np.datetime64("1970-01-01")
    days = np.array([float((np.datetime64(x["date"][:10]) - d0) // np.timedelta64(1, "D")) for x in nodes])
    src, tgt = edges[:, 0], edges[:, 1]
    if dataset == "npm":
        raw = npm_texts(ROOT / "data" / "raw_npm")
        missing = [x for x in names if x not in raw]
        print(f"graph nodes={n} edges={len(edges)} raw texts available={n - len(missing)}")
    else:
        print(f"graph nodes={n} edges={len(edges)} (texts taken from the graph payload)")
        raw = None

    print(f"published {dataset} floor = {PUBLISHED[dataset]}")
    for label in args.variants:
        cfg = VARIANTS[label]
        k = cfg.get("k", 300)
        t0 = time.time()
        vals, srcs, poss = [], [], []
        for seed in args.seeds:
            fit, _val, test = split(days, seed)
            if dataset == "npm":
                texts = [build_text(raw.get(x, {"name": x, "desc": "", "kw": "", "readme": ""}), cfg["readme"])
                         for x in names]
            else:
                texts = [x.get("text", "") for x in nodes]
            tfidf = TfidfVectorizer(max_features=cfg["feat"], sublinear_tf=cfg["sub"],
                                    stop_words=cfg["stop"], token_pattern=cfg["tok"],
                                    min_df=cfg["mindf"], smooth_idf=cfg.get("smooth", True))
            Xf = tfidf.fit_transform([texts[i] for i in np.sort(fit)])
            svd = TruncatedSVD(n_components=k, random_state=42, n_iter=10)
            svd.fit(Xf)
            Z = svd.transform(tfidf.transform(texts)).astype(np.float32)
            Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-8)
            m, s, p = floor_for(Z, src, tgt, test, n)
            vals.append(m)
            srcs.append(s)
            poss.append(p)
        print(f"  {label:24s} floor={np.mean(vals):.4f}  per-seed={np.round(vals, 4).tolist()} "
              f"sources={int(np.mean(srcs))} positives={int(np.mean(poss))} dims={k} vocab<={cfg['feat']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
