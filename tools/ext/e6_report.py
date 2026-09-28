"""Aggregate the Maven first-publication re-run (E6) and the canonical Maven run.

Reports, for canonical Maven and for the two first-publication variants, the
node-date distribution (the reason the assessment asked for the re-run), the
per-model test MRR with its paired difference against the same-seed content
floor, and the gate contrasts that the manuscript's main table is built on.

Usage:
    python3 tools/ext/e6_report.py
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
OUT = ROOT / "results_ext"
GRAPHS = {
    "maven": ROOT / "data" / "graphs" / "maven_graph.json.gz",
    "maven_firstpub": ROOT / "data" / "graphs" / "maven_firstpub_graph.json.gz",
    "maven_firstpub_full": ROOT / "data" / "graphs" / "maven_firstpub_full_graph.json.gz",
}
RUN_DIRS = [ROOT / "results" / "runs",
            ROOT / "results_ext" / "maven_firstpub" / "runs",
            ROOT / "results_ext" / "maven_firstpub_full" / "runs"]
CONTRASTS = [
    ("gate", "ragat_sym", "gat_dir"),
    ("time", "gat_time", "gat_dir"),
    ("gate_under_time", "ragat_time", "gat_time"),
    ("full_vs_gatdir", "ragat_time", "gat_dir"),
]


def date_profile(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    days = np.array([(np.datetime64(n["date"]) - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
                     for n in payload["nodes"]], dtype=float)
    span = days.max() - days.min()
    return {
        "nodes": len(payload["nodes"]), "edges": len(payload["edges"]),
        "date_min": str(np.datetime64(int(days.min()), "D")),
        "date_max": str(np.datetime64(int(days.max()), "D")),
        "span_days": int(span),
        "share_within_3_months": float(np.mean(days >= days.max() - 91)),
        "share_within_12_months": float(np.mean(days >= days.max() - 365)),
        "share_within_25pct_of_span": float(np.mean(days <= days.min() + 0.25 * span)) if span else 1.0,
    }


def load_runs() -> dict[str, dict[str, dict[int, dict]]]:
    out: dict[str, dict[str, dict[int, dict]]] = {}
    for d in RUN_DIRS:
        for path in sorted(glob.glob(str(d / "run_*.json"))):
            r = json.loads(Path(path).read_text(encoding="utf-8"))
            out.setdefault(r["dataset"], {}).setdefault(
                r.get("registry_model", r["model"]), {})[int(r["seed"])] = r
    return out


def paired(diff: np.ndarray) -> dict:
    n = diff.size
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    return {"mean": float(diff.mean()), "sd": float(diff.std(ddof=1)) if n > 1 else 0.0,
            "ci95": [float(diff.mean() - crit * se), float(diff.mean() + crit * se)],
            "positive_seeds": int((diff > 0).sum()), "n_seeds": int(n),
            "p": float(stats.ttest_rel(diff, np.zeros_like(diff)).pvalue) if n > 1 else None}


def main() -> int:
    runs = load_runs()
    report: dict = {"graphs": {}, "datasets": {}}
    lines = ["# E6 -- Maven under the first-publication date convention", "",
             "## Node-date distribution (the reason for the re-run)", "",
             "| graph | nodes | edges | span | share within 3 months of the newest node | within 12 months |",
             "|---|---|---|---|---|---|"]
    for name, path in GRAPHS.items():
        if not path.exists():
            continue
        prof = date_profile(path)
        report["graphs"][name] = prof
        lines.append(f"| {name} | {prof['nodes']} | {prof['edges']} | "
                     f"{prof['date_min']} .. {prof['date_max']} ({prof['span_days']} d) | "
                     f"{100 * prof['share_within_3_months']:.1f}% | "
                     f"{100 * prof['share_within_12_months']:.1f}% |")

    for dataset in ("maven", "maven_firstpub", "maven_firstpub_full"):
        if dataset not in runs:
            continue
        per_model = runs[dataset]
        floor = {s: next(iter(per_model.values()))[s]["floor"]["mrr"] for s in per_model[next(iter(per_model))]}
        entry = {"models": {}, "floor": {"mean": float(np.mean(list(floor.values()))) if floor else None,
                                         "n_seeds": len(floor)}}
        lines += ["", f"## {dataset}", "",
                  f"Content floor: **{entry['floor']['mean']:.4f}** over {entry['floor']['n_seeds']} seeds", "",
                  "| model | test MRR | Δ vs floor | 95% CI | cfg |", "|---|---|---|---|---|"]
        for model, seeds in sorted(per_model.items()):
            keys = sorted(set(seeds) & set(floor))
            test = np.array([seeds[s]["test"]["mrr"] for s in keys])
            fl = np.array([floor[s] for s in keys])
            st = paired(test - fl)
            entry["models"][model] = {
                "test_mean": float(test.mean()),
                "test_sd": float(test.std(ddof=1)) if len(keys) > 1 else 0.0,
                "paired_vs_floor": st,
                "configs": sorted({seeds[s].get("grid_id") for s in keys}),
            }
            lines.append(f"| {model} | {test.mean():.4f}±"
                         f"{entry['models'][model]['test_sd']:.4f} | {st['mean']:+.4f} | "
                         f"[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] | "
                         f"{','.join(sorted({seeds[s].get('grid_id') for s in keys}))} |")
        contrasts = {}
        for name, a, b in CONTRASTS:
            if a in per_model and b in per_model:
                keys = sorted(set(per_model[a]) & set(per_model[b]))
                d = np.array([per_model[a][s]["test"]["mrr"] - per_model[b][s]["test"]["mrr"] for s in keys])
                contrasts[name] = paired(d)
                entry[name] = contrasts[name]
        report["datasets"][dataset] = entry
        if contrasts:
            lines += ["", "| contrast | mean | 95% CI | signs + | p |", "|---|---|---|---|---|"]
            for name, st in contrasts.items():
                lines.append(f"| {name} | {st['mean']:+.4f} | [{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] | "
                             f"{st['positive_seeds']}/{st['n_seeds']} | {st['p']:.4f} |")

    (OUT / "e6_maven_firstpub_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "e6_maven_firstpub_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
