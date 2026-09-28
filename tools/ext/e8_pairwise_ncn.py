"""E8 -- a learned pairwise structural ranker (NCN-style) under the frozen protocol.

``e2_structural_baselines.py`` covers the training-free structural family (common
neighbours, Adamic-Adar, resource allocation, personalised PageRank, a spectral
embedding).  The literature the assessment points at -- NCN (neural common
neighbour) and BUDDY -- is *learned*: a pairwise scorer over neighbourhood
features, optionally with precomputed multi-hop subgraph sketches.  This script
implements that family at this repository's scale:

    features(u,v) = [ A[u,v], (A^2)[u,v], AdamicAdar[u,v], ResourceAlloc[u,v],
                      PPR[u,v], <z_u, z_v> (rank-128 spectral),
                      log1p(deg_u), log1p(deg_v) ]
    score(u,v)    = MLP(features)            (8 -> 64 -> 64 -> 1)

and reports a one-hop-only variant (``A`` plus the degrees) so the contribution
of the multi-hop information that BUDDY precomputes as sketches is visible.
What is *not* implemented is BUDDY's ELPH sketch pipeline itself: its code and
precomputed features are not vendored here and cannot be fetched in this
environment, so the multi-hop ablation stands in for it and the boundary is
stated in the paper rather than papered over with numbers we do not have.

Protocol identical to every other model in the paper: the scorer is fitted on
the fit and validation windows only (positives = observed edges, ten sampled
negatives per positive), and it is then ranked over the frozen test candidate
pool of each source, under both observed-graph definitions used by E2:

    ``obs``  fit+val edges -- honest: fewer edges at test time than the GNNs get;
    ``test`` the GNNs' ``mp_edges_test`` -- like-for-like, and it makes the
             protocol artefact of E2 visible for a learned model as well.

Usage:
    python3 tools/ext/e8_pairwise_ncn.py --workers 2
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
import torch.nn as nn

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TOOLS))

from content_floor import build_bundle                      # noqa: E402
from ranking import evaluate_ranking                        # noqa: E402
from e2_structural_baselines import (_ppr_matrix, _spectral_embedding,  # noqa: E402
                                     _sym_adj)

OUT = ROOT / "results_ext" / "e8_pairwise"
SEEDS = {"npm": [101, 102, 103, 104, 105], "maven": [121, 122, 123, 124, 125]}
VARIANTS = ["onehop", "full"]
NEG_PER_POS = 10
EPOCHS = 40
BATCH = 4096
HIDDEN = 64


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _features(A, A2, Aa, Ar, PPR, Z, deg, variant):
    """Return a function pairs -> feature matrix, with per-source row caches."""
    rows: dict[str, dict[int, np.ndarray]] = {}

    def row_of(mat, src, dense=False):
        cache = rows.setdefault(id(mat), {})
        r = cache.get(src)
        if r is None:
            r = (np.asarray(mat[src].todense()).ravel() if not dense else mat[src]).astype(np.float32)
            cache[src] = r
        return r

    def build(pairs) -> np.ndarray:
        p = np.asarray(pairs)
        u, v = p[0].astype(np.int64), p[1].astype(np.int64)
        single = bool(np.all(u == u[0]))
        if single:
            s = int(u[0])
            a1, a2 = row_of(A, s), row_of(A2, s)
            aa, ar, pp = row_of(Aa, s), row_of(Ar, s), row_of(PPR, s, dense=True)
            zs = Z[s]
            cols = [a1[v], a2[v], aa[v], ar[v], pp[v], (zs[None, :] * Z[v]).sum(axis=1),
                    np.log1p(deg[u]), np.log1p(deg[v])]
        else:
            cols = [np.array([A[i, j] for i, j in zip(u, v)], dtype=np.float32),
                    np.array([A2[i, j] for i, j in zip(u, v)], dtype=np.float32),
                    np.array([Aa[i, j] for i, j in zip(u, v)], dtype=np.float32),
                    np.array([Ar[i, j] for i, j in zip(u, v)], dtype=np.float32),
                    np.array([PPR[i, j] for i, j in zip(u, v)], dtype=np.float32),
                    np.array([float(Z[i] @ Z[j]) for i, j in zip(u, v)], dtype=np.float32),
                    np.log1p(deg[u]), np.log1p(deg[v])]
        X = np.stack(cols, axis=1).astype(np.float32)
        return X[:, :2 + 4 + 2] if variant == "full" else X[:, [0, 6, 7]]

    return build


def _graph_tensors(edges: np.ndarray, n: int):
    A = _sym_adj(edges, n)
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    A2 = (A @ A).tocsr()
    with np.errstate(divide="ignore", invalid="ignore"):
        aa_w = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2.0)), 0.0).astype(np.float32)
        ra_w = np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0).astype(np.float32)
    Aa = (sp.diags(aa_w) @ A).tocsr()
    Ar = (sp.diags(ra_w) @ A).tocsr()
    PPR = _ppr_matrix(A)
    Z = _spectral_embedding(A, 128)
    return A, A2, Aa, Ar, PPR, Z, deg


def _train(A, A2, Aa, Ar, PPR, Z, deg, edges, variant, seed):
    build = _features(A, A2, Aa, Ar, PPR, Z, deg, variant)
    pos = edges.T.astype(np.int64) if edges.size else np.zeros((0, 2), np.int64)
    if pos.shape[0] > 60000:
        pos = pos[np.random.default_rng(seed).choice(pos.shape[0], 60000, replace=False)]
    rng = np.random.default_rng(seed + 17)
    n_nodes = int(deg.size)
    src_col = np.repeat(pos[:, 0], NEG_PER_POS)
    dst_col = rng.integers(0, n_nodes, size=src_col.size)
    neg = np.stack([src_col, dst_col], axis=1).astype(np.int64)
    keep = np.asarray(A[neg[:, 0], neg[:, 1]]).ravel() == 0
    neg = neg[keep]
    Xp = build(pos.T)
    Xn = build(neg.T)
    both = np.concatenate([Xp, Xn], axis=0)
    mu, sd = both.mean(axis=0), both.std(axis=0) + 1e-6
    # A *ranking* objective, not pair classification: the frozen protocol scores
    # each source against its whole candidate pool, and a saturated
    # positive-vs-random-negative classifier ranks the pool arbitrarily.  BPR
    # (negative log-sigmoid of the score margin) optimises the ordering itself,
    # which is what the protocol measures.
    Xp_t = torch.from_numpy(((Xp - mu) / sd).astype(np.float32))
    Xn_t = torch.from_numpy(((Xn - mu) / sd).astype(np.float32))
    k = NEG_PER_POS
    torch.manual_seed(seed)
    model = MLP(Xp_t.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n_pos = Xp_t.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(n_pos, generator=g)
        for i in range(0, n_pos, BATCH):
            idx = perm[i:i + BATCH]
            pos = model(Xp_t[idx])
            neg_idx = (idx.repeat(k) * k + torch.arange(k).repeat(len(idx)))
            neg_idx = neg_idx.clamp(max=Xn_t.shape[0] - 1)
            neg = model(Xn_t[neg_idx]).view(len(idx), k)
            loss = -torch.nn.functional.logsigmoid(pos[:, None] - neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, build, mu, sd


def run_cell(dataset: str, seed: int, graph_defs=("obs", "test")) -> dict:
    bundle = build_bundle(dataset, seed)
    graph, split = bundle["graph"], bundle["split"]
    n = int(split["core_mask"].numel())
    fit = split["fit_window"]["fit_mp_edges"].numpy()
    val = split["mp_edges_val"].numpy()
    obs = np.concatenate([fit, val], axis=1) if val.size else fit
    test_mp = split["mp_edges_test"].numpy()

    A, A2, Aa, Ar, PPR, Z, deg = _graph_tensors(obs, n)
    out = {"dataset": dataset, "seed": seed, "n_nodes": n,
           "train_positives": int(obs.shape[1]), "variants": {}}
    for variant in VARIANTS:
        t0 = time.time()
        model, _, mu, sd = _train(A, A2, Aa, Ar, PPR, Z, deg, obs, variant, seed)
        out["variants"].setdefault(variant, {})["train_seconds"] = round(time.time() - t0, 1)
        for gname in graph_defs:
            if gname == "obs":
                A_, A2_, Aa_, Ar_, PPR_, Z_, deg_ = A, A2, Aa, Ar, PPR, Z, deg
            else:
                A_, A2_, Aa_, Ar_, PPR_, Z_, deg_ = _graph_tensors(test_mp, n)

            def scorer(pairs, A_=A_, A2_=A2_, Aa_=Aa_, Ar_=Ar_, PPR_=PPR_, Z_=Z_, deg_=deg_):
                X = _features(A_, A2_, Aa_, Ar_, PPR_, Z_, deg_, variant)(pairs)
                with torch.no_grad():
                    return model(torch.from_numpy(((X - mu) / sd).astype(np.float32))).numpy()

            m = evaluate_ranking(graph, split, scorer, stage="test")
            m["seconds"] = round(time.time() - t0, 1)
            out["variants"][variant][gname] = m
            del A_, A2_, Aa_, Ar_, PPR_, Z_, deg_
    out["floor"] = {k: bundle["floor"]["test"][k]
                    for k in ("mrr", "hits@10", "hits@100", "sources", "positives")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["npm", "maven"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = []
    for dataset in args.datasets:
        for seed in (args.seeds or SEEDS[dataset]):
            path = OUT / f"cell_{dataset}_seed{seed}.json"
            if not path.exists():
                jobs.append((dataset, seed))

    if args.workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_cell, d, s): (d, s) for d, s in jobs}
            for fut in as_completed(futs):
                d, s = futs[fut]
                payload = fut.result()
                (OUT / f"cell_{d}_seed{s}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                _print_cell(payload)
    else:
        for d, s in jobs:
            payload = run_cell(d, s)
            (OUT / f"cell_{d}_seed{s}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _print_cell(payload)
    return 0


def _print_cell(payload: dict) -> None:
    bits = [f"[e8] {payload['dataset']} seed={payload['seed']} floor={payload['floor']['mrr']:.4f}"]
    for variant, gd in payload["variants"].items():
        bits.append(f"{variant}: " + "/".join(f"{g}={gd[g]['mrr']:.4f}" for g in ("obs", "test") if g in gd))
    print(" | ".join(bits), flush=True)


if __name__ == "__main__":
    sys.exit(main())
