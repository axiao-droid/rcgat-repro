"""Aggregate the instrumented runs (E4) into gate / gradient / ablation tables.

Inputs: ``results_ext/e4_instrument/job_*.json`` written by ``e4_instrument.py``.
Outputs: ``results_ext/e4_instrument_report.{json,md}``.

Reported per (dataset, model, condition, graph_free):
  * test MRR and its paired difference against the same-seed content floor;
  * the epoch-0 anchor ``init_val_mrr`` next to the floor's validation MRR -- with
    the frozen zero initialisation the two must coincide exactly, which is what
    makes the untrained model *be* the content floor;
  * the gate value distribution at the selected checkpoint (mean, SD, range,
    share of gates moved by more than 1%),
  * which descriptors the gates respond to (|correlation| of the pre-activation
    with each of the 13 standardised descriptors);
  * gradient norms per parameter group over training.

Usage:
    python3 tools/ext/e4_report.py
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
OUT = ROOT / "results_ext"

DESC_NAMES = ["log1p_in_deg", "log1p_out_deg", "log1p_mean_in_nbr_deg", "log1p_mean_out_nbr_deg",
              "cos_in", "cos_out", "recip_out", "has_edge", "log1p_recent_in",
              "log1p_recent_out", "recent_in_share", "cit_age", "node_recency"]


def load() -> dict[str, dict]:
    jobs = {}
    for path in sorted(glob.glob(str(SRC / "job_*.json"))):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        key = f"{d['dataset']}|{d['model']}|{d['condition']}|{int(d['graph_free'])}"
        jobs.setdefault(key, []).append(d)
    return jobs


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)) if n > 1 else 0.0,
            "ci95": [float(d.mean() - crit * se), float(d.mean() + crit * se)],
            "positive_seeds": int((d > 0).sum()), "n": int(n),
            "p": float(stats.ttest_rel(d, np.zeros_like(d)).pvalue) if n > 1 else None}


def gate_summary(jobs: list[dict]) -> dict:
    stats_out = {}
    for job in jobs:
        for gate, s in (job.get("selected_gate_stats") or {}).items():
            stats_out.setdefault(gate, []).append(s)
    out = {}
    for gate, rows in stats_out.items():
        out[gate] = {
            "runs": len(rows),
            "g_mean": float(np.mean([r["g_mean"] for r in rows])),
            "g_sd_across_nodes": float(np.mean([r["g_sd"] for r in rows])),
            "g_min": float(np.min([r["g_min"] for r in rows])),
            "g_max": float(np.max([r["g_max"] for r in rows])),
            "moved_share": float(np.mean([r["share_g_below_0.99"] + r["share_g_above_1.01"] for r in rows])),
            "pre_abs_mean": float(np.mean([r["pre_abs_mean"] for r in rows])),
            "pre_abs_p95": float(np.mean([r["pre_abs_p95"] for r in rows])),
            "weight_norm": float(np.mean([r["weight_norm"] for r in rows])),
        }
    return out


def descriptor_response(jobs: list[dict]) -> dict:
    """Correlation of each gate's pre-activation with the 13 descriptors.

    ``gate_descriptor_correlation`` in the job files is stored as
    ``r * sqrt(n_nodes)`` (the numerator is a sum over nodes); it is normalised
    back to a correlation here so the reported numbers are interpretable.
    """
    acc: dict[str, list[np.ndarray]] = {}
    for job in jobs:
        for gate, corr in (job.get("gate_descriptor_correlation") or {}).items():
            n = float((job.get("selected_gate_stats") or {}).get(gate, {}).get("n") or 1.0)
            acc.setdefault(gate, []).append(np.array(corr) / np.sqrt(n))
    out = {}
    for gate, rows in acc.items():
        m = np.mean(np.stack(rows), axis=0)
        order = np.argsort(-np.abs(m))
        out[gate] = {
            "mean_abs_corr": float(np.mean(np.abs(m))),
            "top": [{"descriptor": DESC_NAMES[i] if i < len(DESC_NAMES) else str(i),
                     "corr": float(m[i])} for i in order[:5]],
        }
    return out


def gradient_summary(jobs: list[dict]) -> dict:
    rows: dict[str, list[dict]] = {}
    for job in jobs:
        for entry in job.get("history", []):
            for group, value in (entry.get("grad_norms") or {}).items():
                if value is None:
                    continue
                rows.setdefault(group, []).append({"epoch": entry["epoch"], "norm": value})
    out = {}
    for group, rs in rows.items():
        by_epoch: dict[int, list[float]] = {}
        for r in rs:
            by_epoch.setdefault(r["epoch"], []).append(r["norm"])
        epochs = sorted(by_epoch)
        med = {e: float(np.median(by_epoch[e])) for e in epochs}
        out[group] = {
            "n_observations": len(rs),
            "first_epoch_median": med[epochs[0]], "last_epoch_median": med[epochs[-1]],
            "max_median": max(med.values()), "argmax_epoch": max(med, key=lambda e: med[e]),
            "curve": [[e, med[e]] for e in epochs],
        }
    return out


def main() -> int:
    jobs = load()
    report: dict = {}
    lines = ["# E4 -- gate behaviour, gradient norms, initialisation ablation", ""]
    for key, runs in sorted(jobs.items()):
        dataset, model, cond, gf = key.split("|")
        tag = f"{dataset}/{model}/{cond}{'/text-only' if gf == '1' else ''}"
        test = np.array([r["test"]["mrr"] for r in runs])
        floor = np.array([r["floor_test"]["mrr"] for r in runs])
        init = np.array([r["init_val_mrr"] for r in runs])
        floor_val = np.array([r["floor_val"]["mrr"] for r in runs])
        entry = {
            "dataset": dataset, "model": model, "condition": cond, "graph_free": gf == "1",
            "n_seeds": len(runs), "seeds": sorted(r["seed"] for r in runs),
            "test_mean": float(test.mean()),
            "test_sd": float(test.std(ddof=1)) if len(runs) > 1 else 0.0,
            "floor_mean": float(floor.mean()),
            "paired_vs_floor": paired(test, floor),
            "init_val_mrr_mean": float(init.mean()),
            "floor_val_mrr_mean": float(floor_val.mean()),
            "init_equals_floor_val": bool(np.allclose(init, floor_val, atol=1e-9)),
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in runs])),
            "seconds_mean": float(np.mean([r["train_seconds"] for r in runs])),
            "gates": gate_summary(runs),
            "gate_descriptor_response": descriptor_response(runs),
            "gradients": gradient_summary(runs),
            "configs": sorted({r["config_id"] for r in runs}),
        }
        report[key] = entry
        lines.append(f"## {tag}")
        lines.append("")
        lines.append(f"{len(runs)} seeds, config {','.join(entry['configs'])}; test MRR "
                     f"{test.mean():.4f} (floor {floor.mean():.4f}, paired Δ "
                     f"{entry['paired_vs_floor']['mean']:+.4f} "
                     f"[{entry['paired_vs_floor']['ci95'][0]:+.4f},{entry['paired_vs_floor']['ci95'][1]:+.4f}]); "
                     f"epoch-0 anchor == floor(val): {entry['init_equals_floor_val']}.")
        for gate, s in entry["gates"].items():
            top = entry["gate_descriptor_response"].get(gate, {}).get("top", [])
            lines.append(f"- gate `{gate}`: g in [{s['g_min']:.3f},{s['g_max']:.3f}], "
                         f"mean {s['g_mean']:.3f}, sd {s['g_sd_across_nodes']:.3f}, "
                         f"|g-1|>0.01 for {100 * s['moved_share']:.1f}% of nodes; "
                         f"pre-act |a| p95 {s['pre_abs_p95']:.3f}; weight norm {s['weight_norm']:.3f}; "
                         f"descriptors: " + ", ".join(f"{t['descriptor']} ({t['corr']:+.2f})" for t in top))
        for group, g in entry["gradients"].items():
            lines.append(f"- grad `{group}`: median norm {g['first_epoch_median']:.4f} at epoch "
                         f"{g['curve'][0][0]} -> {g['last_epoch_median']:.4f} at epoch "
                         f"{g['curve'][-1][0]}, max median {g['max_median']:.4f} at epoch {g['argmax_epoch']}")
        lines.append("")

    # the two ablation contrasts, paired by seed
    contrasts = {"rand_vs_zero": {}, "textonly_vs_zero": {}}
    for dataset in sorted({k.split("|")[0] for k in jobs}):
        for model in sorted({k.split("|")[1] for k in jobs}):
            def by_seed(cond: str, gf: str) -> dict[int, float]:
                key = f"{dataset}|{model}|{cond}|{gf}"
                return {r["seed"]: r["test"]["mrr"] for r in jobs.get(key, [])}
            z = by_seed("zero", "0")
            for name, other in (("rand_vs_zero", by_seed("rand", "0")),
                                ("textonly_vs_zero", by_seed("zero", "1"))):
                common = sorted(set(z) & set(other))
                if len(common) < 2:
                    continue
                st = paired(np.array([other[s] for s in common]), np.array([z[s] for s in common]))
                contrasts[name][f"{dataset}|{model}"] = {**st, "seeds": common,
                                                         "leaves": {s: other[s] for s in common},
                                                         "zero": {s: z[s] for s in common}}
    report["contrasts"] = contrasts
    lines += ["# Ablation contrasts (paired by seed)", ""]
    for name, entries in contrasts.items():
        if not entries:
            continue
        lines += [f"## {name}", "", "| cell | mean Δ | 95% CI | seeds + | p |", "|---|---|---|---|---|"]
        for cell, st in sorted(entries.items()):
            lines.append(f"| {cell} | {st['mean']:+.4f} | "
                         f"[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] | "
                         f"{st['positive_seeds']}/{st['n']} | {st['p']:.4f} |")
        lines.append("")
    (OUT / "e4_instrument_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "e4_instrument_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'e4_instrument_report.md'} ({len(jobs)} cells, "
          f"{sum(len(v) for v in jobs.values())} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
