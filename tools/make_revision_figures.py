"""Regenerate the three result figures from the rebuilt runs.

The manuscript's figures on disk disagree with the rebuilt numbers (they were
drawn from the earlier round), and a revision that keeps them would contradict
its own tables.  Everything here comes from ``results/runs/*.json`` and
``results/floor/*.json``, i.e. from the same artifacts the tables are built from.

Confidence bars use the pre-registered convention: a *paired Student t* interval
at 95 %, i.e. ``t(0.975, n-1) * sd / sqrt(n)`` with ``sd`` the per-seed paired
difference (ddof=1).  This is the same convention ``src/summarize.py`` writes into
``results/summary/summary.json`` (``ci95_lo`` / ``ci95_hi``) and the one the
judgment rule of the paper is stated in.  Do not substitute 1.96 here: at n = 10
the normal quantile is 13 % narrower than the t quantile (2.262), so the interval
in the figure would no longer match the interval printed next to it.

Usage::

    python tools/make_revision_figures.py --out <dir>      # writes <dir>/figures/*.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"npm": list(range(101, 111)), "maven": list(range(121, 131))}
CELLS = ["gat_dir", "ragat_sym", "gat_time", "ragat_time"]
BASELINES = ["gcn_dir", "sage_dir", "gatv2_dir"]
PRETTY = {
    "gat_dir": "gat_dir", "ragat_sym": "rcgat_sym", "gat_time": "gat_time",
    "ragat_time": "rcgat_time", "gcn_dir": "gcn*", "sage_dir": "sage*", "gatv2_dir": "gatv2*",
}
GATED = "ragat_sym"
GATED_COLOR = "#d95f02"
UNGATED_COLOR = "#8c8c8c"
BASE_COLOR = "#b9b9b9"
CONTRAST_LABEL = {
    "gate": "gate  (rcgat_sym - gat_dir)",
    "time": "time  (gat_time - gat_dir)",
    "gate_under_time": "gate under time  (rcgat_time - gat_time)",
    "full_vs_gatdir": "full  (rcgat_time - gat_dir)",
    "gat_vs_floor": "ungated vs floor  (gat_dir - floor)",
    "rcgat_vs_floor": "gate vs floor  (rcgat_sym - floor)",
}


def load(dataset: str) -> dict:
    floor = {}
    for seed in SEEDS[dataset]:
        path = ROOT / "results" / "floor" / f"floor_{dataset}_seed{seed}.json"
        floor[seed] = json.loads(path.read_text(encoding="utf-8"))["floor"]["test"]["mrr"]
    runs: dict[str, dict[int, dict]] = {}
    for model in CELLS + BASELINES:
        for seed in SEEDS[dataset]:
            path = ROOT / "results" / "runs" / f"run_{dataset}_{model}_seed{seed}.json"
            if not path.exists():
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            runs.setdefault(model, {})[seed] = {"test": rec["test"], "val": rec["val"]}
    return {"floor": floor, "runs": runs}


def summary() -> dict:
    return json.loads((ROOT / "results" / "summary" / "summary.json").read_text(encoding="utf-8"))


def paired_t_interval(values) -> tuple[float, float]:
    """Mean and half-width of the paired Student t 95 % interval (ddof=1)."""
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    if a.size < 2:
        return mean, 0.0
    half = float(stats.t.ppf(0.975, a.size - 1) * a.std(ddof=1) / np.sqrt(a.size))
    return mean, half


def fig_main_results(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=False)
    for ax, dataset in zip(axes, ("npm", "maven")):
        d = data[dataset]
        models = CELLS + BASELINES
        means, sds, colors, hatches = [], [], [], []
        for m in models:
            per_seed = [d["runs"][m][s]["test"]["mrr"] for s in SEEDS[dataset]]
            means.append(float(np.mean(per_seed)))
            sds.append(float(np.std(per_seed, ddof=1)))
            colors.append(GATED_COLOR if m == GATED else (UNGATED_COLOR if m in CELLS else BASE_COLOR))
            hatches.append("" if m == GATED else ("//" if m in CELLS else "..."))
        x = np.arange(len(models))
        for xi, (mean, sd, color, hatch) in enumerate(zip(means, sds, colors, hatches)):
            ax.bar(xi, mean, yerr=sd, capsize=2.5, color=color, hatch=hatch,
                   edgecolor="black", linewidth=0.5,
                   error_kw={"elinewidth": 0.8, "ecolor": "black"})
        floor = float(np.mean(list(d["floor"].values())))
        ax.axhline(floor, ls="--", lw=1.0, color="black")
        ax.text(0.02, 0.97, f"dashed: content floor {floor:.4f}\nsolid orange: gated cell, "
                f"hatched: ungated, dotted: baselines", transform=ax.transAxes, va="top", ha="left",
                fontsize=6.5, linespacing=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY[m] for m in models], rotation=35, ha="right", fontsize=7)
        ax.set_title("npm" if dataset == "npm" else "Maven", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(0, max(means) * 1.28 + 0.05)
        if dataset == "npm":
            ax.set_ylabel("test MRR", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_main_results.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_gate_contrast(summ: dict, out: Path) -> None:
    keys = ["gate", "time", "gate_under_time", "full_vs_gatdir", "gat_vs_floor", "rcgat_vs_floor"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharex=True)
    for ax, dataset in zip(axes, ("npm", "maven")):
        contrasts = {c["contrast"]: c for c in summ[dataset]["contrasts"]}
        y = np.arange(len(keys))[::-1]
        for yi, key in zip(y, keys):
            c = contrasts[key]
            color = GATED_COLOR if key == "gate" else "#4d4d4d"
            ax.errorbar(c["mean"], yi, xerr=[[c["mean"] - c["ci95_lo"]], [c["ci95_hi"] - c["mean"]]],
                        fmt="o", ms=4, color=color, ecolor=color, elinewidth=1.1, capsize=2.5)
        ax.axvline(0, ls="-", lw=0.8, color="black")
        ax.set_yticks(y)
        ax.set_yticklabels([CONTRAST_LABEL[k] for k in keys], fontsize=7)
        ax.set_title("npm" if dataset == "npm" else "Maven", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("paired test-MRR difference", fontsize=8)
        ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(out / "fig_gate_contrast.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_gate_perseed(data: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0))
    for ax, dataset in zip(axes, ("npm", "maven")):
        d = data[dataset]
        seeds = SEEDS[dataset]
        diffs = np.array([d["runs"][GATED][s]["test"]["mrr"] - d["runs"]["gat_dir"][s]["test"]["mrr"]
                          for s in seeds])
        mean, half = paired_t_interval(diffs)
        x = np.arange(len(seeds))
        for xi, v in zip(x, diffs):
            color = GATED_COLOR if v > 0 else "#333333"
            ax.vlines(xi, 0, v, color=color, lw=1.2)
            marker = "o" if v > 0 else "s"
            ax.plot(xi, v, marker=marker, ms=4.5, color=color)
        ax.axhline(mean, ls="--", lw=1.0, color="#1b9e77")
        ax.axhline(0, lw=0.8, color="black")
        ax.errorbar(len(seeds) + 0.8, mean, yerr=half, fmt="o", ms=4, color="#1b9e77",
                    ecolor="#1b9e77", capsize=3, elinewidth=1.2)
        ax.set_xticks(list(x) + [len(seeds) + 0.8])
        ax.set_xticklabels([str(s) for s in seeds] + ["CI"], fontsize=6.5)
        ax.set_title("npm" if dataset == "npm" else "Maven", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        if dataset == "npm":
            ax.set_ylabel("paired gate gain (test MRR)", fontsize=8)
        ax.margins(x=0.06)
    fig.tight_layout()
    fig.savefig(out / "fig_gate_perseed.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = {ds: load(ds) for ds in ("npm", "maven")}
    summ = summary()
    fig_main_results(data, out)
    fig_gate_contrast(summ, out)
    fig_gate_perseed(data, out)
    for name in ("fig_main_results.pdf", "fig_gate_contrast.pdf", "fig_gate_perseed.pdf"):
        p = out / name
        print(f"wrote {p} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
