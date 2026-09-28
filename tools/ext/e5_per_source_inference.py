"""E5 -- source-level inference on the paired contrasts.

The frozen analysis is a paired t-test over the ten *seeds*, which treats a seed
(a partition) as the unit of replication.  The reviewer asks for the other unit:
the *source* (the node doing the ranking).  With ~29-360 sources per dataset the
source is the unit that actually carries the signal, and an effect can be real
for the population of sources while the seed-level test is underpowered, or the
reverse.

Reads the ``ps_*.npz`` per-source arrays written by ``e4_instrument`` and
reports, for each (dataset, model):

  * mean per-source reciprocal-rank difference against the content floor, with a
    normal and a source-level bootstrap 95% interval;
  * paired t / Wilcoxon signed-rank / sign test over sources (the last two are
    distribution-free);
  * the share of sources improved, and the concentration of the total effect
    (share of the positive mass contributed by the best 1% / 5% of sources);
  * the same statistics after trimming the sources with the largest absolute
    difference, so a single source cannot carry the conclusion;
  * the same for the gate contrast (ragat_sym - gat_dir), which the manuscript
    reports as null.

Usage:
    python3 tools/ext/e5_per_source_inference.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
SRC = ROOT / "results_ext" / "e4_instrument"
OUT = ROOT / "results_ext" / "e5_persource"
BOOT = 10000
RNG = np.random.default_rng(20260928)


def _load(pattern: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(glob.glob(str(SRC / pattern))):
        d = np.load(path)
        seed = int(pattern.split("seed")[1].split("_")[0])
        out[seed] = {k: d[k] for k in d.files}
    return out


def _ci_t(x: np.ndarray) -> list[float]:
    n = x.size
    se = x.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    return [float(x.mean() - crit * se), float(x.mean() + crit * se)]


def _ci_boot(x: np.ndarray, boot: int = BOOT) -> list[float]:
    idx = RNG.integers(0, x.size, size=(boot, x.size))
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(diff: np.ndarray, n_pos: np.ndarray) -> dict:
    n = diff.size
    pos_mass = float(np.sort(diff)[::-1][: max(1, n // 100)].sum() / max(diff[diff > 0].sum(), 1e-12))
    pos_mass5 = float(np.sort(diff)[::-1][: max(1, n // 20)].sum() / max(diff[diff > 0].sum(), 1e-12))
    out = {
        "n_sources": int(n),
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)) if n > 1 else 0.0,
        "se_t": float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "ci95_t": _ci_t(diff) if n > 1 else [float(diff.mean())] * 2,
        "ci95_boot": _ci_boot(diff) if n > 1 else [float(diff.mean())] * 2,
        "share_positive": float(np.mean(diff > 0)), "share_negative": float(np.mean(diff < 0)),
        "share_tied": float(np.mean(diff == 0)),
        "concentration_positive_mass_top1pct": pos_mass,
        "concentration_positive_mass_top5pct": pos_mass5,
    }
    if n > 2:
        out["p_ttest"] = float(stats.ttest_rel(diff, np.zeros_like(diff)).pvalue)
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(diff).pvalue)
        except Exception:  # noqa: BLE001
            out["p_wilcoxon"] = None
        k = int((diff > 0).sum())
        out["p_sign_test"] = float(stats.binomtest(k, n, 0.5).pvalue)
    for trim in (0.01, 0.05):
        m = int(np.floor(n * trim))
        if m and 2 * m < n:
            s = np.sort(diff)
            out[f"trimmed_{int(trim * 100)}pct"] = {
                "mean_diff": float(s[m:n - m].mean()),
                "n_sources": int(n - 2 * m),
                "p_ttest": float(stats.ttest_rel(s[m:n - m], np.zeros(n - 2 * m)).pvalue),
            }
    if n_pos.size == n and np.unique(n_pos).size > 1:
        out["corr_diff_npositives"] = float(np.corrcoef(diff, n_pos)[0, 1])
    return out


def collect(dataset: str, seeds: tuple[int, ...], model: str) -> dict | None:
    """Per-source arrays averaged over seeds, aligned by source id."""
    model_rr, floor_rr, npos = [], [], []
    used = 0
    for seed in seeds:
        p = SRC / f"ps_{dataset}_{model}_seed{seed}_zero.npz"
        if not p.exists():
            continue
        d = np.load(p)
        used += 1
        model_rr.append({int(s): float(r) for s, r in zip(d["test_arrays_source"], d["test_arrays_rr"])})
        floor_rr.append({int(s): float(r) for s, r in zip(d["floor_test_arrays_source"], d["floor_test_arrays_rr"])})
        npos.append({int(s): float(r) for s, r in zip(d["test_arrays_source"], d["test_arrays_n_positives"])})
    if not model_rr:
        return None
    ids = sorted(set.intersection(*[set(m) for m in model_rr], *[set(f) for f in floor_rr]))
    return {
        "seeds_used": used, "sources": np.array(ids),
        "model": np.array([np.mean([m[i] for m in model_rr]) for i in ids]),
        "floor": np.array([np.mean([f[i] for f in floor_rr]) for i in ids]),
        "n_positives": np.array([np.mean([q[i] for q in npos]) for i in ids]),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"source_dir": str(SRC), "bootstrap": BOOT}
    datasets = {"npm": (101, 102, 103, 104, 105), "maven": (121, 122, 123, 124, 125)}
    models = ["gat_dir", "ragat_sym"]
    for dataset, seeds in datasets.items():
        per_model: dict[str, dict] = {}
        gate_diffs = []
        for model in models:
            col = collect(dataset, seeds, model)
            if col is None:
                continue
            diff = col["model"] - col["floor"]
            per_model[model] = {"seeds_used": col["seeds_used"],
                                "model_mean_mrr": float(col["model"].mean()),
                                "floor_mean_mrr": float(col["floor"].mean()),
                                **summarise(diff, col["n_positives"])}
            per_model[model]["per_source_diff"] = [round(float(x), 6) for x in diff]
            per_model[model]["sources"] = [int(i) for i in col["sources"]]
            gate_diffs.append({int(i): float(col["model"][k]) for k, i in enumerate(col["sources"])})
        if len(gate_diffs) == 2:
            ids = sorted(set(gate_diffs[0]) & set(gate_diffs[1]))
            g = np.array([gate_diffs[1][i] - gate_diffs[0][i] for i in ids])
            per_model["gate_contrast_per_source"] = {"sources": ids, **summarise(g, np.zeros(g.size))}
        report[dataset] = per_model
    (OUT / "e5_persource.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# E5 -- source-level inference on the paired contrasts", "",
             "Unit of replication: the *source* (the node that does the ranking), "
             "averaged over the seeds in which it is rankable; intervals are a "
             f"source-level bootstrap ({BOOT} resamples) and a paired t interval.", "",
             "| dataset | contrast | sources | mean per-source Δ | 95% t CI | 95% bootstrap CI | "
             "improved | sign test p | Wilcoxon p | trimmed 5% mean |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for dataset in datasets:
        for key, val in report.get(dataset, {}).items():
            if not isinstance(val, dict) or "mean_diff" not in val:
                continue
            trim = val.get("trimmed_5pct", {}).get("mean_diff")
            lines.append(
                f"| {dataset} | {key} | {val['n_sources']} | {val['mean_diff']:+.4f} | "
                f"[{val['ci95_t'][0]:+.4f},{val['ci95_t'][1]:+.4f}] | "
                f"[{val['ci95_boot'][0]:+.4f},{val['ci95_boot'][1]:+.4f}] | "
                f"{100 * val['share_positive']:.0f}% | "
                f"{val.get('p_sign_test'):.4f} | {val.get('p_wilcoxon'):.4f} | "
                f"{trim:+.4f} |" if trim is not None else "")
    (OUT / "e5_persource.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for dataset in datasets:
        for key, val in report.get(dataset, {}).items():
            if not isinstance(val, dict) or "mean_diff" not in val:
                continue
            print(f"{dataset:6s} {key:28s} n={val['n_sources']:4d} "
                  f"mean={val['mean_diff']:+.4f} CI_boot={val['ci95_boot']} "
                  f"pos={100 * val['share_positive']:.0f}% p_w={val.get('p_wilcoxon')} "
                  f"p_sign={val.get('p_sign_test')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
