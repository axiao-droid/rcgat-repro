"""Aggregate the citation-network control (E7) rebuilt from the public sources.

The frozen manuscript carries a HepTh/HepPh control table that is not
reproducible from the delivered artefacts (no graph, no node text, no runs).
``e7_citation_control.py`` rebuilds the control from SNAP edge lists plus arXiv
metadata and re-runs the four gated cells on it; this script turns those runs
into the table the manuscript needs:

  * the node-date profile of each rebuilt graph (the temporal split depends on
    it, and the frozen table never reported it);
  * per-cell test MRR with the paired difference against the same-seed content
    floor -- on a strong-content graph the expectation is that the floor is high
    and the structural increment is small;
  * the four contrasts the paper's main table is built on (gate, time,
    gate_under_time, full_vs_gatdir), so the strong-content regime can be read
    against the weak-content regimes of npm and Maven.

Usage:
    python3 tools/ext/e7_report.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))

from e6_report import date_profile, paired  # noqa: E402

OUT = ROOT / "results_ext"
SOURCES = {
    "hepth": ROOT / "data" / "graphs_ext" / "hepth_graph.json.gz",
    "hepph": ROOT / "data" / "graphs_ext" / "hepph_graph.json.gz",
}
CONTRASTS = [
    ("gate", "ragat_sym", "gat_dir"),
    ("time", "gat_time", "gat_dir"),
    ("gate_under_time", "ragat_time", "gat_time"),
    ("full_vs_gatdir", "ragat_time", "gat_dir"),
]
FROZEN = {  # original/table_citation_control.tex, five seeds per graph
    "hepth": 0.4533,
    "hepph": 0.4209,
}


def load_runs(source: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for path in sorted(glob.glob(str(ROOT / "results_ext" / source / "runs" / "run_*.json"))):
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        out.setdefault(r.get("registry_model", r["model"]), {})[int(r["seed"])] = r
    return out


def main() -> int:
    report: dict = {"graphs": {}, "sources": {}}
    lines = ["# E7 -- citation-network control rebuilt from the public sources", "",
             "SNAP directed citation lists (``cit-HepTh`` / ``cit-HepPh``) restricted to the",
             "arXiv submission years 2002-2003, node text and node dates from the arXiv Atom API",
             "(title + abstract + first submission date), four gated cells, five seeds per graph.",
             "",
             "| graph | nodes | edges | span | within 3 months of newest | within 12 months |",
             "|---|---|---|---|---|---|"]
    for name, path in SOURCES.items():
        if not path.exists():
            continue
        prof = date_profile(path)
        report["graphs"][name] = prof
        lines.append(f"| {name} | {prof['nodes']} | {prof['edges']} | "
                     f"{prof['date_min']} .. {prof['date_max']} ({prof['span_days']} d) | "
                     f"{100 * prof['share_within_3_months']:.1f}% | "
                     f"{100 * prof['share_within_12_months']:.1f}% |")

    for source in SOURCES:
        runs = load_runs(source)
        if not runs:
            continue
        any_model = next(iter(runs.values()))
        floor = {s: any_model[s]["floor"]["mrr"] for s in any_model}
        entry = {"models": {}, "floor": {"mean": float(np.mean(list(floor.values()))),
                                         "n_seeds": len(floor)}}
        lines += ["", f"## {source}", "",
                  f"Content floor: **{entry['floor']['mean']:.4f}** over "
                  f"{entry['floor']['n_seeds']} seeds"
                  + (f" (the frozen table reports {FROZEN[source]:.4f} for the full graph "
                     "under a different build, so the levels are not comparable)"
                     if source in FROZEN else ""),
                  "",
                  "| model | test MRR | Δ vs floor | 95% CI | cfg |", "|---|---|---|---|---|"]
        for model, seeds in sorted(runs.items()):
            keys = sorted(set(seeds) & set(floor))
            test = np.array([seeds[s]["test"]["mrr"] for s in keys])
            fl = np.array([floor[s] for s in keys])
            st = paired(test - fl)
            cfg = sorted({seeds[s].get("grid_id") for s in keys})
            entry["models"][model] = {
                "test_mean": float(test.mean()),
                "test_sd": float(test.std(ddof=1)) if len(keys) > 1 else 0.0,
                "paired_vs_floor": st, "configs": cfg, "seeds": keys,
                "test": {s: float(seeds[s]["test"]["mrr"]) for s in keys},
            }
            lines.append(f"| {model} | {test.mean():.4f}±"
                         f"{entry['models'][model]['test_sd']:.4f} | {st['mean']:+.4f} | "
                         f"[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] | {','.join(cfg)} |")
        contrasts = {}
        for name, a, b in CONTRASTS:
            if a in runs and b in runs:
                keys = sorted(set(runs[a]) & set(runs[b]))
                d = np.array([runs[a][s]["test"]["mrr"] - runs[b][s]["test"]["mrr"] for s in keys])
                contrasts[name] = paired(d)
                entry[name] = contrasts[name]
        report["sources"][source] = entry
        if contrasts:
            lines += ["", "| contrast | mean | 95% CI | signs + | p |", "|---|---|---|---|---|"]
            for name, st in contrasts.items():
                lines.append(f"| {name} | {st['mean']:+.4f} | "
                             f"[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] | "
                             f"{st['positive_seeds']}/{st['n_seeds']} | {st['p']:.4f} |")

    (OUT / "e7_citation_control_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "e7_citation_control_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
