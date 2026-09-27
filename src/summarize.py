"""Summarize multi-seed runs into tables + paired contrasts.

Differences from the recovered ``summarize6.py``:

* the content floor is **read from the per-seed floor runs** instead of being
  hard-coded (``FLOOR = {'npm': 0.1565, 'maven': 0.3377}``), so the floor and the
  models always come from the same partition -- hard-coding it made the
  ``delta_floor`` column unverifiable and silently wrong after any data change;
* the Student-t critical value comes from ``scipy.stats`` instead of a hand-typed
  table that stopped at 20 degrees of freedom;
* contrasts are paired per seed, and the sign count is reported for every
  contrast (the manuscript's decision rule needs "9 of 10 seeds").

Usage::

    python summarize.py --datasets npm maven
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
MANIFEST = json.loads((PROJECT / "MANIFEST.json").read_text(encoding="utf-8"))
CELLS = MANIFEST["models"]["cells"]
BASELINES = MANIFEST["models"]["baselines"]
MODELS = CELLS + BASELINES

CONTRASTS = [
    ("gat_vs_floor", "gat_dir", None),
    ("gcn_vs_floor", "gcn_dir", None),
    ("sage_vs_floor", "sage_dir", None),
    ("gatv2_vs_floor", "gatv2_dir", None),
    ("rcgat_vs_floor", "ragat_sym", None),
    ("gate", "ragat_sym", "gat_dir"),
    ("time", "gat_time", "gat_dir"),
    ("gate_under_time", "ragat_time", "gat_time"),
    ("full_vs_gatdir", "ragat_time", "gat_dir"),
    ("gcn_vs_gatdir", "gcn_dir", "gat_dir"),
    ("sage_vs_gatdir", "sage_dir", "gat_dir"),
    ("gatv2_vs_gatdir", "gatv2_dir", "gat_dir"),
]


def _runs(dataset: str, model: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted((RESULTS / "runs").glob(f"run_{dataset}_{model}_seed*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        out[int(rec["seed"])] = rec
    return out


def _floors(dataset: str) -> dict[int, float]:
    """Per-seed content floor, restricted to the ten *final* seeds.

    ``run_experiments --phase floor`` also measures the tuning seed (it feeds the
    epoch-0 anchor check in the audit), but the reported floor is a mean over the
    ten final seeds -- including seed 11 would mix a different split into the
    mean and into the paired Delta-vs-floor column.
    """
    final_seeds = set(MANIFEST["final_seeds"][dataset])
    out: dict[int, float] = {}
    for path in sorted((RESULTS / "floor").glob(f"floor_{dataset}_seed*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        seed = int(rec["seed"])
        if seed in final_seeds:
            out[seed] = float(rec["floor"]["test"]["mrr"])
    return out


def paired(diffs: list[float]) -> dict:
    n = len(diffs)
    m = float(mean(diffs))
    if n > 1:
        sd = stdev(diffs)
        sem = sd / math.sqrt(n)
        half = float(stats.t.ppf(0.975, n - 1) * sem)
        p = float(stats.ttest_1samp(diffs, 0.0).pvalue) if any(d != 0 for d in diffs) else 1.0
    else:
        half, p = None, None
    significant = bool(half is not None and ((m - half) > 0 or (m + half) < 0))
    return {
        "mean": m,
        "sd": float(stdev(diffs)) if n > 1 else 0.0,
        "ci95_lo": (m - half) if half is not None else None,
        "ci95_hi": (m + half) if half is not None else None,
        "p_value": p,
        "seeds": n,
        "signs_positive": sum(1 for d in diffs if d > 0),
        "signs_negative": sum(1 for d in diffs if d < 0),
        "significant_95": significant,
        "rule_9of10": abs(sum(1 for d in diffs if d > 0) - n / 2) >= n / 2 - 1 if n == 10 else None,
    }


def summarize(datasets: list[str]) -> dict:
    out: dict = {}
    for dataset in datasets:
        floors = _floors(dataset)
        table = []
        runs_by_model = {m: _runs(dataset, m) for m in MODELS}
        for model in MODELS:
            runs = runs_by_model[model]
            if not runs:
                continue
            seeds = sorted(runs)
            mrr = [runs[s]["test"]["mrr"] for s in seeds]
            h10 = [runs[s]["test"]["hits@10"] for s in seeds]
            h100 = [runs[s]["test"]["hits@100"] for s in seeds]
            floor_paired = [runs[s]["test"]["mrr"] - floors[s] for s in seeds if s in floors]
            table.append({
                "dataset": dataset,
                "model": model,
                "seeds": len(seeds),
                "seed_list": seeds,
                "test_mrr_mean": float(mean(mrr)),
                "test_mrr_sd": float(stdev(mrr)) if len(mrr) > 1 else 0.0,
                "delta_floor_mean": float(mean(floor_paired)) if floor_paired else None,
                "hits@10": float(mean(h10)),
                "hits@100": float(mean(h100)),
                "cfg": runs[seeds[0]].get("grid_id"),
                "params": runs[seeds[0]].get("param_count"),
                "best_epoch_mean": float(mean([runs[s]["best_epoch"] for s in seeds])),
                "train_seconds_mean": float(mean([runs[s]["train_seconds"] for s in seeds])),
            })
        contrasts = []
        for name, a, b in CONTRASTS:
            ra = runs_by_model.get(a) or {}
            if not ra:
                continue
            if b is None:
                diffs = [ra[s]["test"]["mrr"] - floors[s] for s in sorted(ra) if s in floors]
            else:
                rb = runs_by_model.get(b) or {}
                diffs = [ra[s]["test"]["mrr"] - rb[s]["test"]["mrr"] for s in sorted(ra) if s in rb]
            if not diffs:
                continue
            contrasts.append({"dataset": dataset, "contrast": name, "a": a, "b": b or "floor", **paired(diffs)})
        out[dataset] = {
            "table": table,
            "contrasts": contrasts,
            "floor": {
                "mean": float(mean(floors.values())) if floors else None,
                "sd": float(stdev(floors.values())) if len(floors) > 1 else 0.0,
                "seeds": sorted(floors),
                "per_seed": {str(k): v for k, v in sorted(floors.items())},
            },
        }
    return out


def _write_md(summary: dict, path: Path) -> None:
    lines = ["# Multi-seed summary (npm / Maven)", ""]
    for dataset, block in summary.items():
        lines += [f"## {dataset}", "",
                  f"Content floor (per-seed mean over {len(block['floor']['seeds'])} seeds): "
                  f"**{block['floor']['mean']:.4f}** ± {block['floor']['sd']:.4f}", "",
                  "| model | test MRR | Δ vs floor (paired) | H@10 | H@100 | cfg | params | best epoch | s/run |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in block["table"]:
            d = f"{row['delta_floor_mean']:+.4f}" if row["delta_floor_mean"] is not None else "n/a"
            lines.append(f"| {row['model']} | {row['test_mrr_mean']:.4f}±{row['test_mrr_sd']:.4f} | {d} | "
                         f"{row['hits@10']:.4f} | {row['hits@100']:.4f} | {row['cfg']} | {row['params']} | "
                         f"{row['best_epoch_mean']:.1f} | {row['train_seconds_mean']:.0f} |")
        lines += ["", "### Paired contrasts (per-seed paired t)", "",
                  "| contrast | mean | 95% CI | seeds | signs+ | p |", "|---|---|---|---|---|---|"]
        for c in block["contrasts"]:
            ci = f"[{c['ci95_lo']:+.4f},{c['ci95_hi']:+.4f}]" if c["ci95_lo"] is not None else "n/a"
            p = f"{c['p_value']:.4f}" if c["p_value"] is not None else "n/a"
            lines.append(f"| {c['contrast']} | {c['mean']:+.4f} | {ci} | {c['seeds']} | {c['signs_positive']} | {p} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def _write_csv(summary: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "seeds", "test_mrr_mean", "test_mrr_sd", "delta_floor_mean",
                    "hits@10", "hits@100", "cfg", "params", "best_epoch_mean", "train_seconds_mean"])
        for dataset, block in summary.items():
            for r in block["table"]:
                w.writerow([dataset, r["model"], r["seeds"], round(r["test_mrr_mean"], 5),
                            round(r["test_mrr_sd"], 5),
                            None if r["delta_floor_mean"] is None else round(r["delta_floor_mean"], 5),
                            round(r["hits@10"], 5), round(r["hits@100"], 5), r["cfg"], r["params"],
                            round(r["best_epoch_mean"], 1), round(r["train_seconds_mean"], 1)])
        w.writerow([])
        w.writerow(["dataset", "contrast", "a", "b", "mean", "sd", "ci95_lo", "ci95_hi", "p_value",
                    "seeds", "signs_positive"])
        for dataset, block in summary.items():
            for c in block["contrasts"]:
                w.writerow([dataset, c["contrast"], c["a"], c["b"], round(c["mean"], 5), round(c["sd"], 5),
                            None if c["ci95_lo"] is None else round(c["ci95_lo"], 5),
                            None if c["ci95_hi"] is None else round(c["ci95_hi"], 5),
                            None if c["p_value"] is None else round(c["p_value"], 5),
                            c["seeds"], c["signs_positive"]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=MANIFEST["datasets"])
    ap.add_argument("--out-dir", default=str(RESULTS / "summary"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(args.datasets)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_md(summary, out / "summary.md")
    _write_csv(summary, out / "summary.csv")
    print(f"written: {out}/summary.json|.md|.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
