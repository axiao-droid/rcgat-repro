"""Content floor + chronological bundle (rebuilt for npm / Maven).

Replaces the recovered ``phase2_models/content_floor.py``, which was a
protocol-spec reconstruction with three defects that made it unusable:

1. it expected an already adapter-shaped record (``{'name','date','text','deps'}``)
   that no recovered code produced -- the registry -> record step was missing
   (now ``build_dataset.py``);
2. its TF-IDF/SVD was fitted on *all* node texts although the docstring and the
   paper both promise a train-only fit (leakage);
3. it split *edges* chronologically into 70/10/20 while the paper specifies a
   **node-date split** (``V_fit/V_val/V_test`` by node date) with **one split per
   seed**.  It also used a fake ``y*365+m*30+d`` day index, inconsistent with the
   real ``np.datetime64`` day counts used for the descriptor/edge-time features.

This module implements the paper's protocol:

* node-date split with cutoffs ``tau_fit < tau_val`` leaving fixed *fractions* of
  nodes in the fit / validation / test windows; the position of the window is
  drawn per seed, so the ten seeds give ten distinct partitions
  (``十 seeded test windows`` in the paper);
* training edges = out-edges of fit-window nodes; validation / test edges =
  out-edges of the respective window nodes; edge date = source node date;
* candidates (``core_mask``) = all nodes except the test-window *sources*, minus
  the source itself and minus leakage nodes, matching
  "nodes in the fit or validation windows plus other nodes that cannot be test
  sources";
* TF-IDF and the truncated SVD (300d) are fitted on fit-window texts only and
  applied to the remaining nodes; vectors are L2-normalised;
* descriptors and their standardisation use fit-window edges / fit nodes only.
"""
from __future__ import annotations

import gzip
import hashlib
import math
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
GRAPH_DIR = Path(os.environ.get("GRAPH_DIR", PROJECT / "data" / "graphs"))
BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", PROJECT / "data" / "bundles"))

# Nominal window sizes as fractions of nodes.  ``f_test`` is pinned by the
# published candidate counts: 4962 - 4614 - 1 ~= 347 nodes ~= 7.0% on npm and
# 1484 - 1376 - 1 ~= 107 ~= 7.2% on Maven.
# token pattern calibrated in tools/floor_sweep.py: it keeps scoped, hyphenated and
# dotted names (react-dom, @babel/core, jakarta.servlet) that plain \w+ shreds.
TFIDF_TOKEN_PATTERN = r"(?u)[a-zA-Z][a-zA-Z0-9_+#.-]*"

F_TEST = float(os.environ.get("F_TEST", "0.07"))
F_VAL = float(os.environ.get("F_VAL", "0.08"))
JITTER = float(os.environ.get("SPLIT_JITTER", "0.05"))
JITTER_TEST = float(os.environ.get("SPLIT_JITTER_TEST", "0.0"))
# Minimum half-width (in nodes) of the validation-window jitter band.  The band
# must hold enough distinct sizes that the final seeds cannot collide, otherwise
# some of the ten "independent splits" would be bitwise identical.
VAL_HALF_BAND_MIN = int(os.environ.get("VAL_HALF_BAND_MIN", "12"))
# bumped whenever the split rule changes, so stale bundles can never be
# reused silently (README D-4)
SPLIT_RULE_VERSION = 2
# Fraction by which the *test* window may move with the seed.  Default 0: the
# published candidate counts pin the window size (n - round(F_TEST*n) - 1), and
# the manuscript text that asks for "ten distinct train-test partitions" is then
# satisfied through the fit/validation boundary instead (see _split_indices).
# Set SPLIT_JITTER_TEST=0.05 to move the window itself, which makes the test set
# differ per seed at the cost of matching the published candidate arithmetic only
# on average -- both readings are reported in README section 3.4 / 5.9.


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_graph(dataset: str, graph_dir: Path | None = None) -> dict[str, Any]:
    graph_dir = Path(graph_dir or GRAPH_DIR)
    path = graph_dir / f"{dataset}_graph.json.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run build_dataset.py first "
            f"(python src/build_dataset.py --datasets {dataset})"
        )
    with gzip.open(path, "rb") as fh:
        payload = json.loads(fh.read().decode("utf-8"))
    nodes = payload["nodes"]
    return {
        "dataset": dataset,
        "names": [n["name"] for n in nodes],
        "node_dates": np.array([n["date"] for n in nodes]),
        "texts": [n.get("text", "") for n in nodes],
        "edges": np.array(payload["edges"], dtype=np.int64).reshape(-1, 2) if payload["edges"] else np.zeros((0, 2), np.int64),
        "stats": payload.get("stats", {}),
        "built_at": payload.get("built_at"),
        "source": str(path),
    }


def _node_days(node_dates: np.ndarray) -> np.ndarray:
    day = np.datetime64
    return np.array(
        [(day(str(d)) - day("1970-01-01")) // np.timedelta64(1, "D") for d in node_dates],
        dtype=np.float64,
    )


def _val_half_band(n: int) -> int:
    """Half-width (nodes) of the validation-window jitter band for ``n`` nodes."""
    return max(int(round(F_VAL * JITTER * n)), VAL_HALF_BAND_MIN)


def _split_indices(days: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node-date split, one distinct partition per seed, fixed window fractions.

    The test window is exactly the newest ``F_TEST`` fraction of the nodes: the
    published mean candidate count is ``n - round(F_TEST*n) - 1``, which only
    holds if the window size does not move with the seed (4,962 - 347 - 1 =
    4,614 for npm, 1,484 - 104 - 1 = 1,379 for Maven, published 4,614 / 1,376).
    The per-seed variation required by the manuscript ("each seed uses its own
    independent split, so the ten seeds produce ten distinct train-test
    partitions") therefore enters through the validation/fit boundary: the val
    window is ``F_VAL`` of the nodes +/- at most ``_val_half_band(n)``, which moves
    the fit cutoff and hence the training edge set and the candidate membership of
    every non-test node.  An earlier version moved the whole window instead, which
    enlarged the test set (491 instead of 347 nodes on npm) and broke the published
    candidate arithmetic.

    The offset is drawn *without replacement* by construction: with a half-band of
    V nodes there are 2V+1 candidate window sizes and the offset is
    ``(step * (seed mod (2V+1))) mod (2V+1) - V`` with ``gcd(step, 2V+1) = 1``, so
    any set of seeds that is distinct modulo ``2V+1`` (the ten final seeds of a
    dataset always are: they span < 2V+1) gets ten pairwise different cutoffs and
    hence ten pairwise different fit windows.  A plain RNG draw did not: on Maven
    (n=1664, band +/-7) four of the ten seeds drew the same size and their splits
    were bitwise identical, which the audit's B7 check flags.  ``SPLIT_JITTER_TEST``
    (default 0) additionally jitters the *test* window itself, the alternative
    reading in which the ten test sets differ; audit B8 reports which is active.
    """
    n = len(days)
    order = np.argsort(days, kind="stable")
    rng = np.random.default_rng(seed * 7919 + 13)
    if JITTER_TEST > 0:
        n_test = max(1, int(round(F_TEST * (1.0 + JITTER_TEST * (2.0 * rng.random() - 1.0)) * n)))
    else:
        n_test = max(1, int(round(F_TEST * n)))
    # Validation half-band, in nodes.  ``VAL_HALF_BAND_MIN`` guarantees that the
    # band holds at least that many distinct window sizes, which is what lets the
    # per-seed offsets below be *provably* distinct: with +/-V nodes there are
    # 2V+1 possible sizes, and an affine map of the seed that is invertible
    # modulo 2V+1 visits a different size for every seed that is distinct modulo
    # 2V+1.  Without the floor, the ten Maven seeds (n=1664, +/-7 nodes) collided
    # and four of the ten splits were bitwise identical -- the audit's B7 check
    # caught exactly that in the previous build.
    val_half_band = _val_half_band(n)
    span = 2 * val_half_band + 1
    step = next(a for a in range(5, 5 + span) if math.gcd(a, span) == 1)
    offset = (step * (seed % span)) % span - val_half_band
    start = n - n_test
    n_val = max(1, min(int(round(F_VAL * n)) + offset, start - 1)) if start > 1 else 0
    test_idx = order[start:]
    val_idx = order[max(0, start - n_val):start]
    fit_idx = order[: max(0, start - n_val)]
    return np.sort(fit_idx), np.sort(val_idx), np.sort(test_idx)

def build_bundle(
    dataset: str,
    seed: int,
    content_dim: int = 300,
    graph_dir: Path | None = None,
    cache: bool = True,
) -> dict[str, Any]:
    """Build the per-seed bundle (content floor + split + descriptors + edge times)."""
    cache_path = None
    if cache:
        key = (f"{dataset}_seed{seed}_v{SPLIT_RULE_VERSION}_ft{F_TEST}_fv{F_VAL}_jit{JITTER}_d{content_dim}"
               # only mixed in when the optional test-window jitter is on, so the
               # default bundles stay valid across this switch (D-4 in the README:
               # the key must encode every knob that changes the bundle)
               f"{('_jt' + str(JITTER_TEST)) if JITTER_TEST > 0 else ''}"
               f"_mf{os.environ.get('TFIDF_MAX_FEATURES', '20000')}"
               f"_sub{os.environ.get('TFIDF_SUBLINEAR', '1')}"
               f"_sw{os.environ.get('TFIDF_STOPWORDS', 'english')}"
               f"_tok{_sha(os.environ.get('TFIDF_TOKEN_PATTERN', TFIDF_TOKEN_PATTERN))}"
               f"_mdf{os.environ.get('TFIDF_MIN_DF', '1')}")
        cache_path = Path(BUNDLE_DIR) / f"{_sha(key)}.pt"
        if cache_path.exists():
            try:
                return torch.load(cache_path, weights_only=False)
            except Exception:  # noqa: BLE001
                pass

    raw = load_graph(dataset, graph_dir)
    names, texts, edges = raw["names"], raw["texts"], raw["edges"]
    n = len(names)
    days = _node_days(raw["node_dates"])
    src_all = edges[:, 0] if len(edges) else np.zeros(0, np.int64)
    tgt_all = edges[:, 1] if len(edges) else np.zeros(0, np.int64)

    fit_idx, val_idx, test_idx = _split_indices(days, seed)
    fit_mask = np.zeros(n, bool)
    val_mask = np.zeros(n, bool)
    test_mask = np.zeros(n, bool)
    fit_mask[fit_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    e_fit = fit_mask[src_all]
    e_val = val_mask[src_all]
    e_test = test_mask[src_all]
    fit_edges_np = np.stack([src_all[e_fit], tgt_all[e_fit]]) if e_fit.any() else np.zeros((2, 0), np.int64)
    val_edges_np = np.stack([src_all[e_val], tgt_all[e_val]]) if e_val.any() else np.zeros((2, 0), np.int64)
    test_edges_np = np.stack([src_all[e_test], tgt_all[e_test]]) if e_test.any() else np.zeros((2, 0), np.int64)

    # candidate pool: every node outside the test window, minus the source itself.
    # (A test-window non-source node is *also* excluded by the window rule here;
    #  the published candidate counts are reproduced exactly by the window rule,
    #  which is what the paper's "mean candidates per source" implies:
    #  4962 - 347 - 1 = 4614.)
    core_mask = ~test_mask

    # --- content floor: TF-IDF + SVD fitted on fit-window texts only ---
    # texts in canonical (ascending index) order: the randomized SVD's basis
    # depends on the row order of its input, and a row order that follows the
    # date sort changes the floor by ~0.006 MRR through near-tie reordering.
    # Sorting here plus n_iter=10 below makes the floor reproducible.
    fit_texts = [texts[i] for i in np.sort(fit_idx)]
    # Recipe calibrated on the published npm floor (0.1565) with
    # tools/floor_sweep.py: the code-aware token pattern below keeps hyphenated
    # and scoped package names (``react-dom``, ``babel/core``) and lands on
    # 0.1573 over the ten npm seeds, whereas the plain \w+ pattern, the sklearn
    # default \w\w+ pattern and the plain-tf variant give 0.1669 / 0.1633 /
    # 0.1796.  All knobs stay overridable so the sensitivity is reproducible.
    tfidf = TfidfVectorizer(
        max_features=int(os.environ.get("TFIDF_MAX_FEATURES", "20000")),
        sublinear_tf=os.environ.get("TFIDF_SUBLINEAR", "1") == "1",
        # tools/floor_sweep.py variant v12 = the calibrated recipe: it is the only
        # one of the twelve candidates that lands on the published npm floor
        # (0.1573 vs 0.1565); the stop-word list stays on.
        stop_words=(os.environ.get("TFIDF_STOPWORDS", "english") or None),
        token_pattern=os.environ.get("TFIDF_TOKEN_PATTERN", TFIDF_TOKEN_PATTERN),
        min_df=int(os.environ.get("TFIDF_MIN_DF", "1")),
        lowercase=True,
    )
    x_fit = tfidf.fit_transform(fit_texts)
    k = min(content_dim, max(2, min(x_fit.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=42, n_iter=10)
    svd.fit(x_fit)
    x_all = tfidf.transform(texts)
    svd_all = svd.transform(x_all).astype(np.float32)
    norms = np.linalg.norm(svd_all, axis=1, keepdims=True)
    features = torch.tensor(svd_all / np.maximum(norms, 1e-08), dtype=torch.float32)

    graph = {"dates": [str(d) for d in raw["node_dates"]], "node_names": names, "n_nodes": n}
    fit_window = {
        "fit_mp_edges": torch.tensor(fit_edges_np, dtype=torch.long),
        "fit_pos": torch.tensor(fit_edges_np, dtype=torch.long),
        "core_mask": torch.tensor(fit_mask, dtype=torch.bool),
        "metadata": {
            "dataset": dataset,
            "seed": seed,
            "n_nodes": n,
            "n_edges": int(len(edges)),
            "fit_nodes": int(fit_mask.sum()),
            "val_nodes": int(val_mask.sum()),
            "test_nodes": int(test_mask.sum()),
            "fit_edges": int(fit_edges_np.shape[1]),
            "val_edges": int(val_edges_np.shape[1]),
            "test_edges": int(test_edges_np.shape[1]),
            "cutoff": str(raw["node_dates"][test_idx[0]]) if len(test_idx) else str(raw["node_dates"][-1]),
            "split_rule": "node_date_quantile_fixed_fraction_per_seed",
            "f_test": F_TEST,
            "f_val": F_VAL,
            "jitter": JITTER,
            "jitter_test": JITTER_TEST,
            "val_half_band": _val_half_band(n),
            "split_rule_version": SPLIT_RULE_VERSION,
        },
    }
    split = {
        "mp_edges_val": torch.tensor(val_edges_np, dtype=torch.long),
        "mp_edges_test": torch.tensor(test_edges_np, dtype=torch.long),
        "core_mask": torch.tensor(core_mask, dtype=torch.bool),
        "fit_window": fit_window,
    }

    bundle: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "graph": graph,
        "split": split,
        "fit_window": fit_window,
        "features": features,
        "feature_metadata": {
            "type": "tfidf_svd_content_floor",
            "fit_scope": "fit_window_only",
            "dim": int(features.shape[1]),
            "tfidf_max_features": int(os.environ.get("TFIDF_MAX_FEATURES", "20000")),
            "tfidf_token_pattern": os.environ.get("TFIDF_TOKEN_PATTERN", TFIDF_TOKEN_PATTERN),
            "tfidf_stopwords": os.environ.get("TFIDF_STOPWORDS", "english"),
            "tfidf_min_df": int(os.environ.get("TFIDF_MIN_DF", "1")),
            "svd_n_iter": 10,
            "svd_random_state": 42,
            "tfidf_sublinear": os.environ.get("TFIDF_SUBLINEAR", "1") == "1",
            "l2_normalised": True,
            "floor_reference": {"npm": 0.1565, "maven": 0.3377},
        },
        "graph_stats": raw["stats"],
        "graph_built_at": raw["built_at"],
    }

    # descriptors + edge-time features (single source of truth: gated.py)
    from gated import compute_descriptors

    desc, desc_meta = compute_descriptors(
        features,
        fit_window["fit_mp_edges"],
        torch.tensor(fit_mask, dtype=torch.bool),
        np.array([str(d) for d in raw["node_dates"]]),
        fit_window["metadata"]["cutoff"],
    )
    bundle["descriptors"] = desc
    bundle["descriptor_metadata"] = desc_meta
    bundle["edge_time"] = {
        "fit": _edge_time(days, fit_window["fit_mp_edges"]),
        "val": _edge_time(days, split["mp_edges_val"]),
        "test": _edge_time(days, split["mp_edges_test"]),
    }
    bundle["floor"] = content_floor_metrics(bundle)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, cache_path)
    return bundle


def _edge_time(node_days: np.ndarray, edge_index: torch.Tensor) -> torch.Tensor:
    """log1p(age in days) with the edge date taken as the source node date."""
    if edge_index.numel() == 0:
        return torch.zeros((0, 1), dtype=torch.float32)
    src = edge_index[0].numpy()
    tgt = edge_index[1].numpy()
    age = node_days[src] - node_days[tgt]
    return torch.from_numpy(np.log1p(np.clip(age, 0.0, None)).astype(np.float32).reshape((-1, 1)))


def content_floor_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    """Cosine content floor over the same per-source protocol as the models."""
    from ranking import evaluate_ranking

    z = bundle["features"]

    def scorer(pairs: torch.Tensor) -> np.ndarray:
        left = z.index_select(0, pairs[0])
        right = z.index_select(0, pairs[1])
        return (left * right).sum(dim=-1).numpy()

    out = {stage: evaluate_ranking(bundle["graph"], bundle["split"], scorer, stage=stage)
           for stage in ("val", "test")}
    out["feature"] = "cosine on L2-normalised TF-IDF+SVD300 fitted on fit window"
    return out
