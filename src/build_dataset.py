"""Build the frozen npm / Maven dependency-graph snapshots from the raw fetches.

This module is the missing middle layer of the recovered pipeline: the
reconstructed ``content_floor.build_bundle`` expected an already-adapter-shaped
record (``{'name','date','text','deps'}``) and no code in the recovered set ever
produced it from the registry documents.  It also implements the filtering rule
the paper documents ("retain only nodes that participate in at least one
directed edge").

npm
---
* node          = package name
* node date     = ``time.created`` from the packument (first publish)
* edge ``u->v`` = ``versions[dist-tags.latest].dependencies`` of ``u``
* text          = ``description`` + ``keywords`` + README prefix

Maven
-----
* node          = ``groupId:artifactId``
* node date     = latest non-prerelease release date of the artifact; the
                  first-publish date is stored alongside it (``date_first``) so
                  the convention can be switched without refetching.
                  Validated against the published Maven table: with the latest
                  release date and a 2021-04-19 recency filter the realized
                  span starts 2021-04-2x (published 2021-04-19) and max out
                  degree is 87 (published 87, exact); under the first-publish
                  date no filter reproduces those two numbers (span from 2005,
                  max out 90).
* node filter   = artifacts whose node date is on/after ``MAVEN_MIN_DATE``
                  (default 2021-04-19), because the published Maven graph is
                  "concentrated after 2021": without it ~150 dead artifacts
                  enter the graph and stretch the node-date span back to 2005
* edge ``u->v`` = compile-scope dependencies of the latest release POM

Output: one gzipped snapshot per dataset under ``data/graphs/``::

    {"dataset": "npm", "built_at": "...", "node_date_field": "...",
     "nodes": [{"name": ..., "date": ..., "date_latest": ..., "text": ...}, ...],
     "edges": [[src_idx, tgt_idx], ...], "stats": {...}}

Usage::

    RAW_NPM=.../data/raw_npm RAW_MAVEN=.../data/raw_maven \
        python build_dataset.py --datasets npm maven --out-dir data/graphs
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

READMELIMIT = int(os.environ.get("README_PREFIX_CHARS", "1500"))
SYSTEM_ROOTS = {"node", "npm", "js", "python", "java", "net", "util", "fs", "path", "os"}


def _ws(text: str) -> str:
    """Collapse whitespace (the calibrated npm text recipe keeps markup as-is)."""
    return re.sub(r"\s+", " ", text or "").strip()


def _clean(text: str, limit: int | None = None) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] if limit else text


# --------------------------------------------------------------------------- npm

def _npm_record(doc: dict, readme_limit: int = READMELIMIT) -> dict | None:
    name = doc.get("name")
    created = (doc.get("time") or {}).get("created")
    if not name or not created:
        return None
    versions = doc.get("versions") or {}
    if not isinstance(versions, dict):
        return None
    latest = (doc.get("dist-tags") or {}).get("latest")
    deps: list[str] = []
    if latest and isinstance(versions.get(latest), dict):
        raw_deps = versions[latest].get("dependencies") or {}
        deps = [d for d in raw_deps if isinstance(d, str) and not d.startswith("node:")]
    elif versions:
        # no dist-tag: fall back to the newest published version that has deps
        published = [(v.get("time") or "", v) for v in versions.values() if isinstance(v, dict)]
        published.sort(key=lambda kv: kv[0])
        for _, v in reversed(published):
            raw_deps = v.get("dependencies") or {}
            if raw_deps:
                deps = [d for d in raw_deps if isinstance(d, str) and not d.startswith("node:")]
                break
    description = doc.get("description") or ""
    keywords = doc.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    readme = doc.get("readme") or ""

    # Text construction is calibrated, not free: tools/floor_sweep.py evaluates the
    # candidate recipes against the published npm floor (0.1565) on the frozen split.
    # Collapsing whitespace only -- without stripping HTML tags from the README --
    # and truncating the README *before* the fields are joined lands on 0.1573 over
    # the ten npm seeds; stripping HTML first and truncating afterwards gives 0.1624,
    # i.e. 4% off.  See README section 3.3.
    text = " ".join(p for p in [name, _ws(description), _ws(" ".join(str(k) for k in keywords))[:400],
                                _ws(readme)[:readme_limit]] if p)
    return {
        "name": name,
        "date": str(created)[:10],
        "date_latest": str((doc.get("time") or {}).get("modified") or created)[:10],
        "text": text,
        "deps": deps,
        "source_json": doc.get("_id") or name,
    }


def _load_npm(raw_dir: Path) -> list[dict]:
    records = []
    bad = 0
    for path in sorted(raw_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        rec = _npm_record(doc)
        if rec:
            records.append(rec)
        else:
            bad += 1
    print(f"[npm] parsed={len(records)} unparsable={bad}")
    return records


# ------------------------------------------------------------------------- maven

def _maven_record(doc: dict) -> dict | None:
    ga = doc.get("ga")
    if not ga or not doc.get("first_date"):
        return None
    name = doc.get("name") or doc.get("artifact") or ga
    description = doc.get("description") or ""
    text = _clean(f"{name} {description}")
    return {
        "name": ga,
        "date": doc.get("latest_date") or doc.get("first_date"),
        "date_first": doc["first_date"],
        "date_latest": doc.get("latest_date") or doc["first_date"],
        "text": text or _clean(ga),
        "deps": list((doc.get("deps") or {}).keys()),
        "latest": doc.get("latest"),
    }


def _load_maven(raw_dir: Path, node_date: str = "latest") -> list[dict]:
    records = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.endswith(".partial"):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        rec = _maven_record(doc)
        if not rec:
            continue
        if node_date == "first":
            rec["date"] = rec["date_first"]
        records.append(rec)
    print(f"[maven] parsed={len(records)} (node_date={node_date})")
    return records


# ------------------------------------------------------------------- pipeline

def induce_graph(records: list[dict], min_edges: int = 0) -> tuple[list[dict], list[tuple[int, int]]]:
    """Induced directed subgraph.

    ``min_edges=0`` (default) keeps every parsed package/artifact as a node and
    only requires the *edges* to be internal.  This is what the published npm
    numbers imply: 13{,}596 / 4{,}962 = 2.740024183796856, i.e. the node count is
    the number of successfully fetched packages (isolated nodes included).
    ``min_edges=1`` implements the rule stated in the paper text ("retain only
    nodes that participate in at least one directed edge") and yields ~64 fewer
    npm nodes; the two readings are kept switchable so the choice can be stated
    explicitly in the paper.
    """
    index = {r["name"]: i for i, r in enumerate(records)}
    keep: set[int] = set()
    raw_edges: list[tuple[int, int]] = []
    for i, rec in enumerate(records):
        for dep in rec.get("deps") or []:
            j = index.get(dep)
            if j is None or j == i:
                continue
            raw_edges.append((i, j))
            keep.add(i)
            keep.add(j)
    if min_edges <= 0:
        keep = set(range(len(records)))
    order = sorted(keep)
    remap = {old: new for new, old in enumerate(order)}
    nodes = [records[old] for old in order]
    edges = [(remap[s], remap[t]) for s, t in raw_edges]
    # deduplicate parallel edges (a package may list the same dep once only, but
    # Maven POMs can repeat a dependency under different scopes/profiles)
    edges = sorted(set(edges))
    return nodes, edges


def graph_stats(nodes: list[dict], edges: list[tuple[int, int]]) -> dict:
    n = len(nodes)
    out_deg = [0] * n
    in_deg = [0] * n
    edge_set = set(edges)
    for s, t in edges:
        out_deg[s] += 1
        in_deg[t] += 1
    dates = sorted(r["date"][:10] for r in nodes if r.get("date"))
    recip = sum(1 for s, t in edges if (t, s) in edge_set) / len(edges) if edges else 0.0
    return {
        "nodes": n,
        "edges": len(edges),
        "avg_out": len(edges) / n if n else 0.0,
        "avg_in": len(edges) / n if n else 0.0,
        "density": len(edges) / (n * (n - 1)) if n > 1 else 0.0,
        "reciprocity": recip,
        "max_out": max(out_deg) if out_deg else 0,
        "max_in": max(in_deg) if in_deg else 0,
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
        "nodes_with_out": sum(1 for d in out_deg if d > 0),
        "src_count": n,
    }


def build(dataset: str, raw_dir: Path, node_date: str = "latest", min_edges: int = 0,
          maven_min_date: str | None = None, maven_max_date: str | None = None) -> dict:
    if dataset == "npm":
        records = _load_npm(raw_dir)
    elif dataset == "maven":
        records = _load_maven(raw_dir, node_date=node_date)
        if maven_min_date:
            kept = [r for r in records if r["date"][:10] >= maven_min_date]
            if maven_max_date:
                kept = [r for r in kept if r["date"][:10] <= maven_max_date]
                print(f"[maven] span filter <= {maven_max_date}: kept {len(kept)}")
            print(f"[maven] recency filter {maven_min_date}: kept {len(kept)}/{len(records)} "
                  f"artifacts (the published Maven span starts {maven_min_date}, i.e. only "
                  f"recently released artifacts are nodes)")
            records = kept
    else:
        raise ValueError(dataset)
    nodes, edges = induce_graph(records, min_edges=min_edges)
    stats = graph_stats(nodes, edges)
    incident = {i for e in edges for i in e}
    stats["isolated_nodes"] = len(nodes) - len(incident)
    stats["min_edges_rule"] = min_edges
    return {
        "dataset": dataset,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "node_date_field": "time.created" if dataset == "npm" else f"maven:{node_date}_release_date",
        "readme_prefix_chars": READMELIMIT if dataset == "npm" else None,
        "maven_min_date": maven_min_date if dataset == "maven" else None,
        "maven_max_date": maven_max_date if dataset == "maven" else None,
        "raw_dir": str(raw_dir),
        "n_raw_records": len(records),
        "nodes": [
            {
                "name": r["name"],
                "date": r["date"][:10],
                "date_latest": (r.get("date_latest") or r["date"])[:10],
                "text": r.get("text", ""),
            }
            for r in nodes
        ],
        "edges": [[int(s), int(t)] for s, t in edges],
        "stats": stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["npm", "maven"])
    ap.add_argument("--out-dir", default="data/graphs")
    ap.add_argument("--raw-npm", default=os.environ.get("RAW_NPM", "data/raw_npm"))
    ap.add_argument("--raw-maven", default=os.environ.get("RAW_MAVEN", "data/raw_maven"))
    ap.add_argument("--maven-node-date", choices=["latest", "first"], default="latest")
    ap.add_argument("--maven-min-date", default=os.environ.get("MAVEN_MIN_DATE", "2021-04-19"),
                    help="keep only Maven artifacts whose node date is on/after this ISO date "
                         "(the published Maven node span starts 2021-04-19); pass \"\" to disable")
    ap.add_argument("--maven-max-date", default=os.environ.get("MAVEN_MAX_DATE") or None,
                    help="keep only Maven artifacts whose node date is on/before this ISO date")
    ap.add_argument("--min-edges", type=int, default=0,
                    help="0 = keep every fetched node (matches the published npm counts); "
                         "1 = keep only nodes with at least one incident edge")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for dataset in args.datasets:
        raw_dir = Path(args.raw_npm if dataset == "npm" else args.raw_maven)
        payload = build(dataset, raw_dir, node_date=args.maven_node_date, min_edges=args.min_edges,
                        maven_min_date=args.maven_min_date or None,
                        maven_max_date=args.maven_max_date)
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        gz_path = out_dir / f"{dataset}_graph.json.gz"
        with gzip.open(gz_path, "wb", compresslevel=9) as fh:
            fh.write(blob)
        stats_path = out_dir / f"{dataset}_stats.json"
        stats_path.write_text(json.dumps(payload["stats"], indent=2), encoding="utf-8")
        summary[dataset] = {
            **payload["stats"],
            "sha256_gz": hashlib.sha256(gz_path.read_bytes()).hexdigest(),
            "gz_bytes": gz_path.stat().st_size,
            "raw_json_bytes": len(blob),
        }
        print(f"[{dataset}] {json.dumps(summary[dataset], ensure_ascii=False)}")
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
