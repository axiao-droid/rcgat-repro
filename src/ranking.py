"""Per-source ranking evaluation (full-filtered candidate pool).

Rebuilt from the recovered ``phase1/ranking.py`` with three fixes:

1. the recovered version computed ``best = min(rank_of.get(t) for t in targets if
   t in rank_of)`` and then tested ``if best is None`` -- ``min()`` over an empty
   generator raises ``ValueError`` instead of returning ``None``, so any source
   whose positives are all outside the candidate pool crashed the run;
2. the source itself was left inside its own candidate pool, although the paper
   excludes it ("the candidate pool is filtered to exclude the source itself");
3. candidates are now the paper's pool: every node outside the test window minus
   the source, minus leakage nodes, instead of "fit nodes that have at least one
   fit edge".

Metrics: MRR (primary), hits@10, hits@100, computed per source and averaged over
the sources that have at least one rankable positive.

Aggregation order and tie-breaking are stable (``kind='stable'`` argsort over the
ascending candidate list), so the vectorised implementation below returns exactly
the numbers the earlier per-element Python loop returned -- it is a pure speed-up
(the loop cost ~5.5 s per evaluation on npm, i.e. ~75% of the training wall time).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


def leakage_nodes(fit_edges: torch.Tensor, source: int) -> set[int]:
    """Candidates that would put a training edge and a test edge in one pair."""
    if fit_edges.numel() == 0:
        return set()
    row = fit_edges[0] == source
    return set(int(x) for x in fit_edges[1][row].tolist())


def _fit_targets_by_source(fit_edges: torch.Tensor) -> dict[int, np.ndarray]:
    """``{source: targets of its fit-window edges}`` built in one pass."""
    if fit_edges.numel() == 0:
        return {}
    pairs = fit_edges.t().numpy()
    order = np.argsort(pairs[:, 0], kind="stable")
    pairs = pairs[order]
    out: dict[int, np.ndarray] = {}
    start = 0
    while start < len(pairs):
        source = int(pairs[start, 0])
        end = start
        while end < len(pairs) and int(pairs[end, 0]) == source:
            end += 1
        out[source] = pairs[start:end, 1]
        start = end
    return out


def positives_by_source(mp_edges: torch.Tensor) -> dict[int, list[int]]:
    """``{source: [targets in edge order]}`` for one stage."""
    out: dict[int, list[int]] = {}
    for source, target in mp_edges.t().tolist():
        out.setdefault(int(source), []).append(int(target))
    return out


def evaluate_ranking(
    graph: dict[str, Any],
    split: dict[str, Any],
    scorer: Callable[[torch.Tensor], np.ndarray],
    stage: str,
    ks: tuple[int, ...] = (10, 100),
) -> dict[str, float]:
    """Return {'mrr', 'hits@10', 'hits@100', 'sources', 'positives'} for one stage."""
    mp_edges = split[f"mp_edges_{stage}"]
    core_mask = split["core_mask"]
    core = torch.nonzero(core_mask, as_tuple=False).flatten()
    core_np = core.numpy()
    n_nodes = int(core_mask.numel())
    fit_edges = split.get("fit_window", {}).get("fit_mp_edges") if isinstance(split.get("fit_window"), dict) else None
    if fit_edges is None:
        fit_edges = torch.zeros((2, 0), dtype=torch.long)
    fit_targets = _fit_targets_by_source(fit_edges)

    # column index of every candidate inside the pool (per source the pool is the
    # same ascending list of core nodes minus that source's leakage nodes and the
    # source itself)
    col_of = np.full(n_nodes, -1, dtype=np.int64)
    col_of[core_np] = np.arange(core_np.size, dtype=np.int64)

    rr: list[float] = []
    hits = {int(k): 0 for k in ks}
    positives = 0
    blocked = np.zeros(n_nodes, dtype=bool)
    # rank of a node inside the source's pool, written only for the pool members
    # (a node that is in the pool of *some* source is always rewritten before it
    # is read, because the read is masked by that source's pool membership)
    rank_of_node = np.empty(n_nodes, dtype=np.int64)
    for source, targets in sorted(positives_by_source(mp_edges).items()):
        mine = fit_targets.get(source)
        if mine is not None:
            blocked[mine] = True
        blocked[source] = True
        keep = core_np[~blocked[core_np]]
        if keep.size == 0:
            if mine is not None:
                blocked[mine] = False
            blocked[source] = False
            continue
        pairs = torch.from_numpy(np.stack([np.full(keep.size, source, dtype=np.int64), keep]))
        scores = np.asarray(scorer(pairs))
        order = np.argsort(-scores, kind="stable")
        rank_of_keep = np.empty(keep.size, dtype=np.int64)
        rank_of_keep[order] = np.arange(1, keep.size + 1, dtype=np.int64)
        rank_of_node[keep] = rank_of_keep
        tgt = np.asarray(targets, dtype=np.int64)
        tgt_col = col_of[tgt]
        reachable = rank_of_node[tgt[(tgt_col >= 0) & (~blocked[tgt])]]
        if mine is not None:
            blocked[mine] = False
        blocked[source] = False
        if reachable.size == 0:
            continue
        best = int(reachable.min())
        rr.append(1.0 / best)
        positives += int(reachable.size)
        for k in hits:
            hits[k] += int(best <= k)

    count = len(rr)
    result = {"mrr": float(np.mean(rr)) if count else 0.0}
    result.update({f"hits@{k}": float(hits[k] / count) if count else 0.0 for k in ks})
    result["sources"] = count
    result["positives"] = positives
    result["candidates_mean"] = float(core.numel() - 1) if count else 0.0
    return result
