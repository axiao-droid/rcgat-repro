"""E7 -- citation-network control rebuilt from public sources (HepTh / HepPh).

The frozen manuscript carries a "strong-content control" table on the arXiv
citation networks HepTh and HepPh ("five seeds per graph"), but neither the
graphs nor the node text of that earlier round are in this repository, so the
table is not reproducible from the delivered artefacts.  This script rebuilds
the control from the primary public sources instead of re-stating it:

  edges  SNAP ``cit-HepTh`` / ``cit-HepPh`` directed citation lists
         (snap.stanford.edu/data, downloaded once into data/raw_citation/);
  text   arXiv title + abstract, fetched per paper id from the official arXiv
         Atom API (batches, 3 s apart);
  dates  the arXiv submission date returned by the same API -- the SNAP files
         carry no dates, so this is also what makes the temporal split possible.

Scope: the network is restricted to the papers of the last arXiv-id years
(``--min-yy 99``, i.e. 1999-2003) and to their induced citation edges.  That is
the only way the chain fits the CPU budget here (the full HepTh has 27,770 nodes
and 352,807 edges, roughly 15x the wall time of a single npm run, i.e. several
days for the 63 runs per graph).  The frozen table's full-graph numbers are
therefore *not* reproduced by construction; what this run gives is an
independent, fully documented citation-network control at npm/Maven scale.

Usage:
    python3 tools/ext/e7_citation_control.py --fetch              # metadata only
    python3 tools/ext/e7_citation_control.py --build              # graphs only
    python3 tools/ext/e7_citation_control.py --workers 4          # full chain
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "data" / "raw_citation"
META = ROOT / "data" / "raw_citation" / "meta"
GRAPH_DIR = ROOT / "data" / "graphs_ext"
ATOM = "{http://www.w3.org/2005/Atom}"
CATEGORY = {"hepth": "hep-th", "hepph": "hep-ph"}
SNAP_FILE = {"hepth": "cit-HepTh.txt.gz", "hepph": "cit-HepPh.txt.gz"}
# five seeds per graph, as the frozen table reports
SEEDS = [11, 21, 22, 23, 24]      # tuning seed first, then the four final seeds
CELLS = ["gat_dir", "ragat_sym", "gat_time", "ragat_time"]


# --------------------------------------------------------------------------- #
# 1. edges                                                                      #
# --------------------------------------------------------------------------- #
def pad_id(raw: str) -> str:
    """SNAP writes the pre-2007 arXiv ids with the leading zeros dropped.

    ``0100001`` (January 2001) is stored as ``100001`` and ``0004001`` as
    ``4001``; padding back to seven digits recovers ``YYMMNNN``.
    """
    return raw.strip().zfill(7)


def id_year(padded: str) -> int:
    """Calendar year of a seven-digit arXiv id.

    ``cit-HepTh`` / ``cit-HepPh`` start in 1992, so the two-digit year runs
    ``90``..``99`` -> 1990..1999 and ``00``..``07`` -> 2000..2007."""
    yy = int(padded[:2])
    return 1900 + yy if yy >= 90 else 2000 + yy


def read_edges(source: str, min_year: int) -> tuple[list[str], np.ndarray]:
    """SNAP directed edge list, restricted to papers published from ``min_year``."""
    path = RAW / SNAP_FILE[source]
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            a, b = line.split()[:2]
            a, b = pad_id(a), pad_id(b)
            if len(a) == 7 and len(b) == 7:
                rows.append((a, b))
    ids = sorted({x for pair in rows for x in pair
                  if len(x) == 7 and x.isdigit() and id_year(x) >= min_year})
    index = {x: k for k, x in enumerate(ids)}
    edges = [(index[a], index[b]) for a, b in rows if a in index and b in index]
    return ids, np.array(edges, dtype=np.int64).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# 2. arXiv metadata                                                             #
# --------------------------------------------------------------------------- #
def _meta_path(source: str, arxiv_id: str) -> Path:
    return META / source / f"{arxiv_id.replace('/', '_')}.json"


def fetch_metadata(source: str, ids: list[str], batch: int = 150, delay: float = 3.0) -> dict:
    """Title + abstract + submission date per arXiv id, cached on disk."""
    cat = CATEGORY[source]
    (META / source).mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not _meta_path(source, i).exists()]
    print(f"[e7] {source}: {len(ids)} ids, {len(todo)} to fetch", flush=True)
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        query = ",".join(f"{cat}/{i}" for i in chunk)
        url = ("https://export.arxiv.org/api/query?id_list="
               + urllib.parse.quote(query, safe=",/") + f"&max_results={len(chunk)}")
        payload = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=180) as fh:
                    payload = fh.read()
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[e7] batch {start} attempt {attempt + 1} failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                time.sleep(delay * (attempt + 2))
        if payload is None:
            continue
        if len(chunk) > 1:
            time.sleep(delay)
        root = ET.fromstring(payload)
        seen = set()
        for entry in root.findall(f"{ATOM}entry"):
            eid = (entry.findtext(f"{ATOM}id") or "").strip()
            key = eid.rsplit("/abs/", 1)[-1]
            key = key.split("v")[0] if key else ""
            short = key.split("/")[-1]
            if not short:
                continue
            seen.add(short)
            title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
            summary = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
            published = (entry.findtext(f"{ATOM}published") or "")[:10]
            _meta_path(source, short).write_text(json.dumps(
                {"id": key, "title": title, "abstract": summary, "published": published},
                ensure_ascii=False), encoding="utf-8")
        missing = [c for c in chunk if c not in seen]
        for m in missing:
            _meta_path(source, m).write_text(json.dumps({"id": m, "missing": True}), encoding="utf-8")
        print(f"[e7] {source}: fetched {min(start + batch, len(todo))}/{len(todo)} "
              f"({len(missing)} unresolvable)", flush=True)
        time.sleep(delay)
    out = {}
    for i in ids:
        path = _meta_path(source, i)
        if path.exists():
            out[i] = json.loads(path.read_text(encoding="utf-8"))
    print(f"[e7] {source}: metadata for {len(out)}/{len(ids)} ids "
          f"({len(ids) - len(out)} not resolvable)", flush=True)
    return out


# --------------------------------------------------------------------------- #
# 3. graph assembly                                                             #
# --------------------------------------------------------------------------- #
def build_graph(source: str, min_year: int) -> dict:
    ids, edges = read_edges(source, min_year)
    meta = fetch_metadata(source, ids)
    cat = CATEGORY[source]
    keep = [i for i in ids if not meta[i].get("missing") and meta[i].get("title")
            and meta[i].get("published")]
    pos = {x: k for k, x in enumerate(ids)}
    index = {pos[x]: k for k, x in enumerate(keep)}
    nodes = []
    for i in keep:
        m = meta[i]
        nodes.append({
            "name": f"{cat}/{i}",
            "date": m["published"],
            "text": f"{m['title']} {m['abstract']}".strip(),
            "title": m["title"],
        })
    kept_edges = [[index[int(a)], index[int(b)]] for a, b in edges
                  if int(a) in index and int(b) in index]
    dates = np.array([n["date"] for n in nodes])
    src = np.array([e[0] for e in kept_edges], dtype=np.int64) if kept_edges else np.zeros(0, np.int64)
    tgt = np.array([e[1] for e in kept_edges], dtype=np.int64) if kept_edges else np.zeros(0, np.int64)
    n = len(nodes)
    out_deg = np.bincount(src, minlength=n) if n else np.zeros(0, np.int64)
    in_deg = np.bincount(tgt, minlength=n) if n else np.zeros(0, np.int64)
    mutual = len({(int(a), int(b)) for a, b in zip(src, tgt)}
                 & {(int(b), int(a)) for a, b in zip(src, tgt)}) if len(src) else 0
    stats = {
        "nodes": n, "edges": len(kept_edges),
        "avg_out": float(out_deg.mean()) if n else 0.0,
        "avg_in": float(in_deg.mean()) if n else 0.0,
        "density": float(len(kept_edges) / (n * (n - 1))) if n > 1 else 0.0,
        "reciprocity": float(mutual / max(len(kept_edges), 1)),
        "max_out": int(out_deg.max()) if n else 0,
        "max_in": int(in_deg.max()) if n else 0,
        "date_min": min(dates.tolist()), "date_max": max(dates.tolist()),
        "nodes_with_out": int((out_deg > 0).sum()),
        "src_count": n, "isolated_nodes": int(((out_deg + in_deg) == 0).sum()),
        "min_edges_rule": 0,
        "subsample": {"rule": f"arXiv submission year >= {min_year}, "
                              "induced citation edges only",
                      "source_edges": f"SNAP {SNAP_FILE[source]}",
                      "source_text": "arXiv Atom API (title + abstract)",
                      "full_graph": "27,770 nodes / 352,807 edges (HepTh), "
                                    "34,546 nodes / 421,578 edges (HepPh)"},
    }
    return {"dataset": source, "nodes": nodes, "edges": kept_edges, "stats": stats,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "text_recipe": "title + abstract (arXiv Atom API)"}


def write_graph(dataset: str, payload: dict) -> Path:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / f"{dataset}_graph.json.gz"
    with gzip.open(path, "wb", compresslevel=9) as fh:
        fh.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    (GRAPH_DIR / f"{dataset}_stats.json").write_text(
        json.dumps(payload["stats"], indent=2), encoding="utf-8")
    print(f"[e7] wrote {path.name}: {payload['stats']['nodes']} nodes, "
          f"{payload['stats']['edges']} edges, span "
          f"{payload['stats']['date_min']}..{payload['stats']['date_max']}", flush=True)
    return path


# --------------------------------------------------------------------------- #
# 4. full chain on the citation graph                                           #
# --------------------------------------------------------------------------- #
def run_chain(dataset: str, workers: int, final_seeds: list[int] | None = None) -> None:
    import os

    ext_root = ROOT / "results_ext" / dataset
    ext_root.mkdir(parents=True, exist_ok=True)
    os.environ["RESULTS_ROOT"] = str(ext_root)
    os.environ["GRAPH_DIR"] = str(GRAPH_DIR)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    for mod in [m for m in list(sys.modules) if m in ("content_floor", "run_experiments", "gated", "ranking", "train3")]:
        del sys.modules[mod]
    import run_experiments as RX

    RX.RESULTS_DIR = ext_root
    seeds = list(final_seeds) if final_seeds else SEEDS[1:]
    RX.MANIFEST["final_seeds"][dataset] = list(seeds)
    (ext_root / "manifest_patch.json").write_text(json.dumps({
        "dataset": dataset, "final_seeds": seeds, "tuning_seed": SEEDS[0],
        "cells": CELLS, "note": "citation control: four gated cells only, five seeds "
                                "per graph as in the frozen table",
    }, indent=2), encoding="utf-8")

    from content_floor import build_bundle

    for seed in SEEDS:
        b = build_bundle(dataset, seed)
        print(f"[bundle] {dataset} seed={seed} floor_test={b['floor']['test']['mrr']:.4f} "
              f"sources={b['floor']['test']['sources']} "
              f"candidates={b['floor']['test']['candidates_mean']:.0f}", flush=True)
    RX.run_tuning([dataset], CELLS, workers=workers)
    RX.select_configs([dataset], CELLS, shared=False)
    RX.run_final([dataset], CELLS, workers=workers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["hepth", "hepph"])
    ap.add_argument("--min-year", type=int, default=2001)
    ap.add_argument("--fetch", action="store_true", help="download arXiv metadata only")
    ap.add_argument("--build", action="store_true", help="build the graphs only")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--final-seeds", nargs="+", type=int, default=None,
                    help="override the final seed list (used to top up a single seed)")
    args = ap.parse_args()

    for source in args.sources:
        if args.fetch:
            ids, _ = read_edges(source, args.min_year)
            fetch_metadata(source, ids)
            continue
        path = GRAPH_DIR / f"{source}_graph.json.gz"
        if args.build or not path.exists():
            write_graph(source, build_graph(source, args.min_year))
        if args.build:
            continue
        run_chain(source, args.workers, args.final_seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
