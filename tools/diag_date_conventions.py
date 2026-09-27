"""Which node-date convention reproduces the published test-window statistics?

Global graph statistics (nodes / edges / degree extremes / date span) do not
identify the node date used for the split, because the same edge list is produced
either way.  The split does depend on it, and the published test-window numbers
do constrain it:

    npm   : 234 test sources, 686 positives, mean 4,614 candidates
    Maven :  94 test sources, 352 positives, mean 1,376 candidates

For a fixed test fraction (7% of nodes, F_TEST=0.07) the candidate pool size is
``n - n_test - 1`` regardless of which nodes are in the window, so only the
source/positive counts discriminate.  This tool rebuilds the graph under every
available date convention and reports those counts.

Usage::

    python tools/diag_date_conventions.py npm maven
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F_TEST = 0.07
PUBLISHED = {"npm": (4962, 13596, 234, 686, 4614), "maven": (1484, 5482, 94, 352, 1376)}


def _day(stamp: str) -> str:
    return str(stamp)[:10]


def npm_graph(raw_dir: Path, date_key: str, versions: str = "latest"):
    """versions='latest' uses dist-tags.latest deps; 'all' unions every version."""
    nodes, edges = {}, set()
    for path in sorted(raw_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = rec.get("name")
        if not name:
            continue
        time = rec.get("time") or {}
        vs = rec.get("versions") or {}
        latest = (rec.get("dist-tags") or {}).get("latest")
        if date_key == "created":
            stamp = time.get("created")
        elif date_key == "modified":
            stamp = time.get("modified")
        elif date_key == "latest":
            stamp = time.get(latest) or time.get("modified") or time.get("created")
        else:
            raise ValueError(date_key)
        if not stamp:
            continue
        nodes[name] = _day(stamp)
        if versions == "latest":
            deps = (vs.get(latest) or {}).get("dependencies") or {}
        else:
            deps = {}
            for v in vs.values():
                deps.update(v.get("dependencies") or {})
        for dep in deps:
            if isinstance(dep, str) and not dep.startswith("node:"):
                edges.add((name, dep))
    return nodes, edges


def maven_graph(raw_dir: Path, date_key: str):
    nodes, edges = {}, set()
    for path in sorted(raw_dir.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ga = rec.get("ga")
        stamp = rec.get(date_key)
        if not ga or not stamp:
            continue
        nodes[ga] = _day(stamp)
        for dep in (rec.get("deps") or {}):
            edges.add((ga, dep))
    return nodes, edges


def split_stats(names: list[str], dates: list[str], edges: set) -> dict:
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    src, tgt = [], []
    for a, b in edges:
        if a in idx and b in idx:
            src.append(idx[a])
            tgt.append(idx[b])
    src, tgt = np.array(src, np.int64), np.array(tgt, np.int64)
    order = np.argsort(np.array(dates, dtype="datetime64[D]"), kind="stable")
    n_test = max(1, round(0.07 * n))
    test = order[n - n_test:]
    tmask = np.zeros(n, bool)
    tmask[test] = True
    out_any = np.zeros(n, bool)
    out_any[src] = True
    in_win = tmask[tgt]
    t2t = {(int(a), int(b)) for a, b in zip(src[in_win].tolist(), tgt[in_win].tolist())}
    t2n = {(int(a), int(b)) for a, b in zip(src[~in_win].tolist(), tgt[~in_win].tolist())}
    return dict(
        nodes=n, edges=len(edges), n_test=int(n_test), cand=n - int(n_test) - 1,
        max_out=int(np.bincount(src, minlength=n).max()) if len(src) else 0,
        max_in=int(np.bincount(tgt, minlength=n).max()) if len(tgt) else 0,
        dmin=min(dates), dmax=max(dates),
        src_any_out=int((out_any & tmask).sum()),
        src_nontest_target=len({a for a, _ in t2n}),
        pairs_t2t=len(t2t), pairs_t2n=len(t2n),
    )


def main(argv: list[str]) -> int:
    for ds in argv or ["npm"]:
        pub = PUBLISHED[ds]
        print(f"\n=== {ds} ===  published: nodes={pub[0]} edges={pub[1]} "
              f"sources={pub[2]} positives={pub[3]} cand={pub[4]}")
        if ds == "npm":
            raw = ROOT / "data" / "raw_npm"
            variants = [("created/latest-deps", "created", "latest"),
                        ("modified/latest-deps", "modified", "latest"),
                        ("latest/latest-deps", "latest", "latest"),
                        ("created/all-versions-deps", "created", "all")]
            for label, dk, vk in variants:
                nodes, edges = npm_graph(raw, dk, vk)
                names = sorted(nodes)
                st = split_stats(names, [nodes[k] for k in names], edges)
                print(f"  [{label:26s}] {json.dumps(st)}")
        else:
            raw = ROOT / "data" / "raw_maven"
            for dk in ("latest_date", "first_date"):
                nodes, edges = maven_graph(raw, dk)
                names = sorted(nodes)
                st = split_stats(names, [nodes[k] for k in names], edges)
                print(f"  [{dk:26s}] {json.dumps(st)}")
                recent = {k: v for k, v in nodes.items() if v >= "2021-04-19"}
                en = {e for e in edges if e[0] in recent}
                names2 = sorted(recent)
                st2 = split_stats(names2, [recent[k] for k in names2], en)
                print(f"  [{dk + ' >=2021-04-19':26s}] {json.dumps(st2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
