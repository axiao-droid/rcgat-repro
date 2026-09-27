"""Reference (pre-optimisation) per-source loop, kept only for the equivalence test.

This is the exact Python-loop implementation that ``ranking.evaluate_ranking``
replaced; ``tools/test_ranking_equiv.py`` checks that the vectorised version
returns identical numbers on a real bundle.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


def leakage_nodes(fit_edges, source):
    if fit_edges.numel() == 0:
        return set()
    row = fit_edges[0] == source
    return set(int(x) for x in fit_edges[1][row].tolist())


def evaluate_ranking(graph, split, scorer, stage, ks=(10, 100)):
    mp_edges = split[f"mp_edges_{stage}"]
    core_mask = split["core_mask"]
    core = torch.nonzero(core_mask, as_tuple=False).flatten()
    fit_edges = split.get("fit_window", {}).get("fit_mp_edges") if isinstance(split.get("fit_window"), dict) else None
    if fit_edges is None:
        fit_edges = torch.zeros((2, 0), dtype=torch.long)

    positives_by_source = {}
    for source, target in mp_edges.t().tolist():
        positives_by_source.setdefault(int(source), []).append(int(target))

    rr = []
    hits = {int(k): 0 for k in ks}
    positives = 0
    for source, targets in sorted(positives_by_source.items()):
        blocked = leakage_nodes(fit_edges, source)
        blocked.add(source)
        keep = torch.tensor([int(c) for c in core.tolist() if int(c) not in blocked], dtype=torch.long)
        if keep.numel() == 0:
            continue
        pairs = torch.stack([torch.full_like(keep, int(source)), keep], dim=0)
        scores = scorer(pairs)
        order = np.argsort(-np.asarray(scores), kind="stable")
        rank_of = {int(keep[i]): rank + 1 for rank, i in enumerate(order)}
        reachable = [rank_of[t] for t in targets if t in rank_of]
        if not reachable:
            continue
        best = min(reachable)
        rr.append(1.0 / best)
        positives += sum(1 for t in targets if t in rank_of)
        for k in hits:
            hits[k] += int(best <= k)

    count = len(rr)
    result = {"mrr": float(np.mean(rr)) if count else 0.0}
    result.update({f"hits@{k}": float(hits[k] / count) if count else 0.0 for k in ks})
    result["sources"] = count
    result["positives"] = positives
    result["candidates_mean"] = float(core.numel() - 1) if count else 0.0
    return result
