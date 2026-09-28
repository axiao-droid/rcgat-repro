"""E2 -- structural (structure-only) baselines under the frozen protocol.

The assessment asks for NCN / BUDDY-style structural baselines: if the claim is
"graph neighbourhood information does not add much over the text content floor",
the evidence has to be structure-only methods evaluated under *exactly* the
reported protocol (same splits, same candidate pools, same leakage filter, same
metrics).

BUDDY's official code and its pre-computed ELPH features are not vendored in
this repository and cannot be fetched here, so instead of claiming numbers we do
not have, this script implements the reproducible structural family ourselves:

    random      a fixed random score per node (floor sanity check)
    degree      popularity: score = in+out degree of the candidate
    pa          preferential attachment: deg(u) * deg(v)
    cn          common neighbours
    ncn         normalised common neighbours: cn / sqrt(deg(u) deg(v))
    aa          Adamic-Adar: sum over shared neighbours of 1/log(deg)
    ra          resource allocation: sum over shared neighbours of 1/deg
    ppr         personalised PageRank (alpha=0.15, 20 power iterations)
    spectral    rank-128 SVD of the symmetrised adjacency (a training-free
                structural embedding, the closest thing to what the GNNs learn)

All heuristics run on the *symmetrised* observed graph, which is the standard
link-prediction convention and the most favourable choice for the baseline.
Two observed-graph definitions are reported per heuristic:

    ``obs``  = fit-window edges + validation-window edges, i.e. everything
               observed strictly before the test window. This is the honest
               inductive setting and gives the baselines strictly less
               information at test time than the GNNs have;
    ``test`` = the message-passing graph the reported GNN numbers use
               (``mp_edges_test``), for a like-for-like comparison.

Usage:
    python3 tools/ext/e2_structural_baselines.py --workers 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from content_floor import build_bundle          # noqa: E402
from ranking import evaluate_ranking            # noqa: E402

OUT = ROOT / "results_ext" / "e2_structural"
SEEDS = {"npm": list(range(101, 111)), "maven": list(range(121, 131))}
HEURISTICS = ["random", "degree", "pa", "cn", "ncn", "aa", "ra", "ppr", "spectral"]
PPR_ALPHA = 0.15
PPR_ITERS = 20
SPECTRAL_DIM = 128


def _sym_adj(edges: np.ndarray, n: int) -> sp.csr_matrix:
    """Binary symmetric adjacency from a directed edge list."""
    if edges.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    r, c = edges[0], edges[1]
    data = np.ones(2 * len(r), dtype=np.float32)
    m = sp.coo_matrix((data, (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n))
    m = (m + m.T).tocsr()
    m.data[:] = 1.0
    return m


def _ppr_matrix(A: sp.csr_matrix) -> np.ndarray:
    """(1-alpha) (I - alpha P)^{-1} by power iteration; P row-stochastic."""
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    P = sp.diags(inv.astype(np.float32)) @ A
    X = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(X, 1.0 - PPR_ALPHA)
    term = np.eye(n, dtype=np.float32)
    for _ in range(PPR_ITERS):
        term = PPR_ALPHA * (P @ term)
        X += (1.0 - PPR_ALPHA) * term
    return X


def _spectral_embedding(A: sp.csr_matrix, dim: int) -> np.ndarray:
    n = A.shape[0]
    k = min(dim, n - 2)
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    N = sp.diags(inv.astype(np.float32)) @ A @ sp.diags(inv.astype(np.float32))
    u, s, _ = svds(N.tocsr().astype(np.float32), k=k)
    order = np.argsort(-s)
    return (u[:, order] * s[order]).astype(np.float32)


def _make_scorers(A: sp.csr_matrix, n: int, seed_tag: int) -> dict:
    """One scorer per heuristic; each takes pairs (2,M) and returns M scores."""
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    logd = np.log(np.maximum(deg, 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        aa_w = np.where(deg > 1, 1.0 / logd, 0.0).astype(np.float32)
        ra_w = np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0).astype(np.float32)
    Aw = A.copy()
    Aa = sp.diags(aa_w) @ A
    Ar = sp.diags(ra_w) @ A
    rnd = np.random.default_rng(1000003 + seed_tag).random(n).astype(np.float32)
    PPR = _ppr_matrix(A)
    Z = _spectral_embedding(A, SPECTRAL_DIM)

    # evaluate_ranking calls the scorer once per source with that source's whole
    # candidate pool, so the fast path is a single row slice; the per-pair
    # fallback keeps the function correct for any other call pattern.
    def _row_scorer(mat: sp.spmatrix):
        row_cache: dict[int, np.ndarray] = {}
        col_cache: dict[int, np.ndarray] = {}

        def sc(pairs) -> np.ndarray:
            p = np.asarray(pairs)
            u, v = p[0], p[1]
            if np.all(u == u[0]):
                src = int(u[0])
                row = row_cache.get(src)
                if row is None:
                    row = np.asarray(mat[src].todense()).ravel().astype(np.float32)
                    row_cache[src] = row
                return row[v]
            return np.array([mat[int(a), int(b)] for a, b in zip(u, v)], dtype=np.float32)
        return sc

    cn_of, aa_of, ra_of = _row_scorer(A), _row_scorer(Aa), _row_scorer(Ar)

    def sc_random(pairs):
        return rnd[np.asarray(pairs)[1]]

    def sc_degree(pairs):
        return deg[np.asarray(pairs)[1]]

    def sc_pa(pairs):
        p = np.asarray(pairs)
        return deg[p[0]] * deg[p[1]]

    def sc_ncn(pairs):
        p = np.asarray(pairs)
        u, v = p[0], p[1]
        c = cn_of(p)
        return c / np.maximum(np.sqrt(deg[u] * deg[v]), 1.0)

    def sc_ppr(pairs):
        p = np.asarray(pairs)
        return PPR[p[0], p[1]]

    def sc_spectral(pairs):
        p = np.asarray(pairs)
        return (Z[p[0]] * Z[p[1]]).sum(axis=1)

    return {"random": sc_random, "degree": sc_degree, "pa": sc_pa, "cn": cn_of,
            "ncn": sc_ncn, "aa": aa_of, "ra": ra_of, "ppr": sc_ppr,
            "spectral": sc_spectral}


def run_cell(dataset: str, seed: int) -> dict:
    bundle = build_bundle(dataset, seed)
    graph, split = bundle["graph"], bundle["split"]
    n = int(split["core_mask"].numel())
    fit = split["fit_window"]["fit_mp_edges"].numpy()
    val = split["mp_edges_val"].numpy()
    test_mp = split["mp_edges_test"].numpy()
    graphs = {
        "obs": np.concatenate([fit, val], axis=1) if val.size else fit,
        "test": test_mp,
    }
    out = {"dataset": dataset, "seed": seed, "n_nodes": n,
           "edge_counts": {"fit": int(fit.shape[1]), "val": int(val.shape[1]),
                           "test": int(test_mp.shape[1])},
           "heuristics": {}}
    for gname, edges in graphs.items():
        A = _sym_adj(edges, n)
        scorers = _make_scorers(A, n, seed + (7000 if gname == "test" else 0))
        for h in HEURISTICS:
            t0 = time.time()
            m = evaluate_ranking(graph, split, scorers[h], stage="test")
            m["seconds"] = round(time.time() - t0, 1)
            out["heuristics"].setdefault(h, {})[gname] = m
    out["floor"] = {k: bundle["floor"]["test"][k] for k in ("mrr", "hits@10", "hits@100", "sources", "positives")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["npm", "maven"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--graph", default="both", choices=["obs", "test", "both"])
    ap.add_argument("--tag", default="e2_structural")
    args = ap.parse_args()

    rows = []
    for dataset in args.datasets:
        seeds = args.seeds or SEEDS[dataset]
        for seed in seeds:
            path = OUT / f"cell_{dataset}_seed{seed}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                payload = run_cell(dataset, seed)
                OUT.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            rows.append(payload)
            best = max((h for h in HEURISTICS),
                       key=lambda h: payload["heuristics"][h]["obs"]["mrr"])
            print(f"[e2] {dataset} seed={seed} floor={payload['floor']['mrr']:.4f} "
                  f"best_obs={best} {payload['heuristics'][best]['obs']['mrr']:.4f} "
                  f"best_test={max((h for h in HEURISTICS), key=lambda h: payload['heuristics'][h]['test']['mrr'])} "
                  f"{max(payload['heuristics'][h]['test']['mrr'] for h in HEURISTICS):.4f}", flush=True)
    (OUT / f"{args.tag}_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
