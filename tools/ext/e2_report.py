"""Aggregate the E2 structural-baseline cells into per-dataset tables.

For every dataset: the content floor, each heuristic under the honest observed
graph (``obs``) and under the reported test-time message-passing graph
(``test``), each with the mean, the SD across the seeded partitions and the
paired difference against the floor on the same seed.

Usage:
    python3 tools/ext/e2_report.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
SRC = ROOT / "results_ext" / "e2_structural"
OUT = ROOT / "results_ext"


def load() -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(str(SRC / "cell_*.json"))):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        cells.setdefault(d["dataset"], []).append(d)
    return cells


def paired(diff: np.ndarray) -> dict:
    n = diff.size
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    from scipy import stats
    crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    p = float(stats.ttest_rel(diff, np.zeros_like(diff)).pvalue) if n > 1 else None
    return {"mean": float(diff.mean()), "sd": float(diff.std(ddof=1)) if n > 1 else 0.0,
            "ci95": [float(diff.mean() - crit * se), float(diff.mean() + crit * se)],
            "positive_seeds": int((diff > 0).sum()), "n": int(n), "p": p}


def main() -> int:
    cells = load()
    summary: dict = {}
    lines = ["# E2 -- structural baselines under the frozen protocol", ""]
    for dataset, rows in sorted(cells.items()):
        heuristics = sorted(rows[0]["heuristics"])
        floor = np.array([r["floor"]["mrr"] for r in rows])
        entry = {"n_seeds": len(rows), "floor": {"mean": float(floor.mean()),
                                                 "sd": float(floor.std(ddof=1)) if len(rows) > 1 else 0.0},
                 "sources": rows[0]["floor"]["sources"],
                 "candidates_mean": rows[0]["floor"]["candidates_mean"] if "candidates_mean" in rows[0]["floor"] else None,
                 "heuristics": {}}
        lines += [f"## {dataset} ({len(rows)} seeds, {rows[0]['floor']['sources']} test sources)",
                  "", f"Content floor: **{floor.mean():.4f}** (sd {entry['floor']['sd']:.4f})", "",
                  "| heuristic | obs MRR | Δ vs floor (obs) | 95% CI | test-graph MRR |",
                  "|---|---|---|---|---|"]
        for h in heuristics:
            obs = np.array([r["heuristics"][h]["obs"]["mrr"] for r in rows])
            tst = np.array([r["heuristics"][h]["test"]["mrr"] for r in rows])
            st = paired(obs - floor)
            entry["heuristics"][h] = {
                "obs_mean": float(obs.mean()),
                "obs_sd": float(obs.std(ddof=1)) if len(rows) > 1 else 0.0,
                "obs_paired_vs_floor": st,
                "test_mean": float(tst.mean()),
                "test_sd": float(tst.std(ddof=1)) if len(rows) > 1 else 0.0,
                "obs_seconds_mean": float(np.mean([r["heuristics"][h]["obs"]["seconds"] for r in rows])),
            }
            lines.append(f"| {h} | {obs.mean():.4f} | {st['mean']:+.4f} | "
                         f"[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] | {tst.mean():.4f} |")
        best = max(heuristics, key=lambda h: entry["heuristics"][h]["obs_mean"])
        entry["best_obs"] = best
        lines += ["", f"Best honest-graph baseline: **{best}** "
                      f"({entry['heuristics'][best]['obs_mean']:.4f} vs floor {floor.mean():.4f}).", ""]
        summary[dataset] = entry

    (OUT / "e2_structural_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "e2_structural_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
