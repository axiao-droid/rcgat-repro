"""E6 -- Maven rebuilt on the FIRST-PUBLICATION date convention, full chain.

Why: the delivered Maven graph places every artifact at its *latest
non-prerelease release* date, which makes the node dates pile up against the
collection cutoff and turns the temporal test window into a few days rather than
a period.  The external assessment asks for the same pipeline under the *first
publication* convention, which the raw Maven records already carry
(``date_first``); the frozen graph builder supports it with ``node_date='first'``
and no refetch.

Two variants are run, because the convention change and the recency filter are
separable and both are informative:

  ``maven_firstpub``       node_date='first' AND the same calendar recency filter
                           (>= 2021-04-19) applied to the first-publication date.
                           This is the strict like-for-like swap: one thing
                           changed, the date convention.
  ``maven_firstpub_full``  node_date='first' and NO recency filter, so the graph
                           keeps every artifact ever crawled and the node dates
                           span two decades.  This is the variant that gives the
                           temporal protocol real room to move.

Nothing canonical is touched: the graphs are written under their own names and
all run files go to ``results_ext/<dataset>`` via ``RESULTS_ROOT``.

Usage:
    python3 tools/ext/e6_maven_firstpublish.py --workers 4
    python3 tools/ext/e6_maven_firstpublish.py --workers 4 --stage build
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]           # repro/
sys.path.insert(0, str(ROOT / "src"))

MIN_DATE = "2021-04-19"           # same calendar rule as the published graph
# dataset -> builder kwargs; None means "no recency filter"
VARIANTS = {
    "maven_firstpub": {"node_date": "first", "maven_min_date": MIN_DATE},
    "maven_firstpub_full": {"node_date": "first", "maven_min_date": None},
}


def build_graph(dataset: str, kwargs: dict) -> dict:
    """Step 1: the first-publication-convention Maven graph for one variant."""
    out = ROOT / "data" / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{dataset}_graph.json.gz"
    if path.exists():
        payload = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
        print(f"[graph] reusing {path.name} ({len(payload['nodes'])} nodes)", flush=True)
    else:
        from build_dataset import build

        payload = build("maven", ROOT / "data" / "raw_maven", **kwargs)
        payload["dataset"] = dataset      # the bundle key is the file name
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with gzip.open(path, "wb", compresslevel=9) as fh:
            fh.write(blob)
        print(f"[graph] wrote {path.name}: {len(payload['nodes'])} nodes, "
              f"{len(payload['edges'])} edges", flush=True)
    stats = payload["stats"]
    (out / f"{dataset}_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def run_variant(dataset: str, kwargs: dict, stage: str, workers: int) -> dict:
    print(f"\n=== {dataset} ===", flush=True)
    stats = build_graph(dataset, kwargs)

    ext_root = ROOT / "results_ext" / dataset
    ext_root.mkdir(parents=True, exist_ok=True)
    (ext_root / "graph_stats.json").write_text(json.dumps({
        "dataset": dataset, "base_dataset": "maven",
        "node_date_field": "first non-prerelease release date",
        "maven_min_date": kwargs.get("maven_min_date"),
        "stats": stats,
    }, indent=2), encoding="utf-8")
    if stage == "build":
        return stats

    os.environ["RESULTS_ROOT"] = str(ext_root)
    import run_experiments as RX

    # RESULTS_DIR is a module constant bound at import time, so a second variant
    # in the same process would keep writing into the first variant's directory.
    RX.RESULTS_DIR = ext_root

    seeds = list(RX.MANIFEST["final_seeds"]["maven"])
    RX.MANIFEST["final_seeds"][dataset] = seeds
    (ext_root / "manifest_patch.json").write_text(json.dumps({
        "dataset": dataset, "final_seeds": seeds,
        "tuning_seed": RX.MANIFEST["tuning"]["tuning_seed"],
        "grid": RX.MANIFEST["tuning"]["grid"],
        "training": RX.MANIFEST["training"],
    }, indent=2), encoding="utf-8")

    models = RX._all_models()
    t0 = time.time()

    # bundles in the parent so the workers never race on the cache
    from content_floor import build_bundle

    for seed in [RX.MANIFEST["tuning"]["tuning_seed"], *seeds]:
        b = build_bundle(dataset, seed)
        print(f"[bundle] seed={seed} floor_test={b['floor']['test']['mrr']:.4f} "
              f"candidates={b['floor']['test']['candidates_mean']:.0f}", flush=True)

    if stage in ("all", "tune"):
        RX.run_tuning([dataset], models, workers=workers)
    if stage in ("all", "final"):
        RX.select_configs([dataset], models, shared=False)
        RX.run_final([dataset], models, workers=workers)
    print(f"[{dataset}] done in {round(time.time() - t0, 1)}s", flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", default="all", choices=["all", "build", "tune", "final"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    for dataset in [v for v in args.variants.split(",") if v]:
        run_variant(dataset, VARIANTS[dataset], args.stage, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
