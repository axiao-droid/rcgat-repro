"""Emit the revision tables (main / paired / protocol sensitivity / checkpoint rule).

Reads only artifacts of the rebuilt pipeline (``results/summary/summary.json``,
``logs/floor_pool_readings*.json``, ``logs/checkpoint_rules.json``) and writes
LaTeX tables in the manuscript's own layout, so the revised files are drop-in
replacements for ``tables/table_main_results.tex`` and ``tables/table_paired.tex``.

Two conventions differ from the original tables on purpose:

* $\\dagger$ marks the best mean test MRR on each graph, because the ordering
  changed; boldface still marks the pre-registered gated cell (a label, not a
  claim about the best value);
* the paired table keeps the six pre-registered contrasts first and puts the
  baseline contrasts under a separate heading, flagged as not pre-registered.

Usage::

    python tools/make_revision_tables.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CELL_LABEL = {
    "gcn_dir": r"gcn\_dir (external GCN)",
    "sage_dir": r"sage\_dir (external GraphSAGE)",
    "gatv2_dir": r"gatv2\_dir (external GATv2)",
    "gat_dir": r"gat\_dir (directed GAT)",
    "ragat_sym": r"rcgat\_sym (gate)",
    "gat_time": r"gat\_time (time branch)",
    "ragat_time": r"rcgat\_time (gate $+$ time)",
}
ORDER = ["gcn_dir", "sage_dir", "gatv2_dir", "gat_dir", "ragat_sym", "gat_time", "ragat_time"]
GATED = "ragat_sym"
CELLS = ("gat_dir", "ragat_sym", "gat_time", "ragat_time")
GRAPH = {"npm": "npm", "maven": "Maven"}
CFG = {"c1_h128_d0.2": "c1", "c2_h64_d0.2": "c2", "c3_h64_d0.5": "c3", "c4_h128_d0.5": "c4"}
PAIRED_MAIN = [
    (r"gate (rcgat\_sym $-$ gat\_dir)", "gate"),
    (r"time (gat\_time $-$ gat\_dir)", "time"),
    (r"gate under time (rcgat\_time $-$ gat\_time)", "gate_under_time"),
    (r"full (rcgat\_time $-$ gat\_dir)", "full_vs_gatdir"),
    (r"ungated vs content floor (gat\_dir $-$ floor)", "gat_vs_floor"),
    (r"gate vs content floor (rcgat\_sym $-$ floor)", "rcgat_vs_floor"),
]
PAIRED_EXTRA = [
    (r"GCN vs ungated (gcn\_dir $-$ gat\_dir)", "gcn_vs_gatdir"),
    (r"GraphSAGE vs ungated (sage\_dir $-$ gat\_dir)", "sage_vs_gatdir"),
    (r"GATv2 vs ungated (gatv2\_dir $-$ gat\_dir)", "gatv2_vs_gatdir"),
]


def _by_model(summary: dict, dataset: str) -> dict:
    return {row["model"]: row for row in summary[dataset]["table"]}


def _floor_row(dataset: str) -> dict:
    """Mean content-floor metrics over the ten final seeds (test split)."""
    seeds = range(101, 111) if dataset == "npm" else range(121, 131)
    rows = [json.loads((ROOT / "results" / "floor" / f"floor_{dataset}_seed{s}.json")
                       .read_text(encoding="utf-8"))["floor"]["test"] for s in seeds]
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    return {"mrr": mean("mrr"), "h10": mean("hits@10"), "h100": mean("hits@100")}


def _contrasts(summary: dict, dataset: str) -> dict:
    return {row["contrast"]: row for row in summary[dataset]["contrasts"]}


def main_table(summary: dict) -> str:
    L = [r"\begin{table}[t]",
         r"\caption{Test MRR, hits@10, and hits@100 (mean over ten seeds $\pm$ sample standard "
         r"deviation) of the four model cells on the two dependency networks, together with the "
         r"content floor and three external baselines (GCN, GraphSAGE, and GATv2) trained under the "
         r"identical protocol. All cells share the identical task, loss, budget, candidate pools, and "
         r"per-seed splits; hyperparameters were selected per cell on validation MRR with a single "
         r"tuning seed and frozen for all ten final seeds. $\Delta$ floor is the mean test MRR minus "
         r"the content floor. Boldface marks the pre-registered gated cell (a label, not a claim "
         r"about the best value); $\dagger$ marks the best mean test MRR on that graph. The Maven "
         r"graph is a later snapshot of the registry, so absolute values are not comparable with "
         r"earlier rounds even though the protocol is identical.}",
         r"\label{tab:main}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Graph & Cell & MRR & Hits@10 & Hits@100 & $\Delta$ floor & Config & Params \\", r"\midrule"]
    for dataset in ("npm", "maven"):
        rows = _by_model(summary, dataset)
        f = _floor_row(dataset)
        L.append(f"{GRAPH[dataset]} & Content floor (TF-IDF cosine) & {f['mrr']:.4f} & {f['h10']:.4f} & "
                 f"{f['h100']:.4f} & -- & -- & -- \\\\")
        best = max((m for m in ORDER if m in rows), key=lambda m: rows[m]["test_mrr_mean"])
        for model in ORDER:
            rec = rows.get(model)
            if rec is None:
                continue
            name = CELL_LABEL[model]
            if model == GATED:
                name = r"\textbf{" + name + "}"
            if model == best:
                name = r"$\dagger$ " + name
            mrr = f"{rec['test_mrr_mean']:.4f} $\\pm$ {rec['test_mrr_sd']:.4f}"
            if model == GATED:
                mrr = r"\textbf{" + mrr + "}"
            L.append(f"{GRAPH[dataset]} & {name} & {mrr} & {rec['hits@10']:.4f} & "
                     f"{rec['hits@100']:.4f} & ${rec['delta_floor_mean']:+.4f}$ & "
                     f"{CFG.get(rec['cfg'], rec['cfg'])} & {rec['params']:,} \\\\".replace(",", "{,}"))
        if dataset == "npm":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def paired_table(summary: dict) -> str:
    L = [r"\begin{table}[t]",
         r"\caption{Paired contrasts on the two dependency networks. Each seed contributes one "
         r"within-seed difference; the interval is a Student $t$ 95\% confidence interval on the mean "
         r"paired difference (df~$=9$). Sign$+$ is the number of seeds in which the difference is "
         r"positive. The pre-registered judgment rule requires the interval to exclude zero and "
         r"sign$+$ to be at least 9. The last three rows per graph are reported for completeness and "
         r"are not pre-specified.}",
         r"\label{tab:paired}",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"Graph & Contrast & Mean paired difference & 95\% CI & Sign$+$ / 10 \\", r"\midrule"]
    for dataset in ("npm", "maven"):
        contrasts = _contrasts(summary, dataset)

        def row(label: str, key: str) -> str:
            c = contrasts[key]
            return (f"{GRAPH[dataset]} & {label} & ${c['mean']:+.4f}$ & "
                    f"$[{c['ci95_lo']:+.4f},{c['ci95_hi']:+.4f}]$ & {c['signs_positive']} \\\\")

        for label, key in PAIRED_MAIN:
            if key in contrasts:
                L.append(row(label, key))
        L.append(r"\midrule")
        L.append(r"\multicolumn{5}{l}{\emph{not pre-specified}} \\")
        for label, key in PAIRED_EXTRA:
            if key in contrasts:
                L.append(row(label, key))
        if dataset == "npm":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def _pool_means(readings: list[dict], reading: str) -> tuple[dict, dict]:
    floor = {r["seed"]: r["floor"][reading]["test"]["mrr"] for r in readings}
    models: dict[str, list[float]] = {}
    for r in readings:
        for m, v in r["model_A"].items():
            models.setdefault(m, []).append(v)
    return floor, {m: float(np.mean(v)) for m, v in models.items()}


def sensitivity_table(np_readings: list[dict], mav_readings: list[dict]) -> str:
    L = [r"\begin{table}[t]",
         r"\caption{Protocol sensitivity of the content floor. The candidate pool is either the nodes "
         r"outside the test window (reading~A, the reading implied by the published mean candidate "
         r"count) or every node except the source (reading~B). When models and floor are read the "
         r"same way, every cell stays above the floor; crediting the floor under reading~B while "
         r"scoring the models under reading~A reverses the sign. $\Delta$ is the mean test MRR of the "
         r"cell minus the mean floor of the stated reading; the parenthetical value is the number of "
         r"seeds of ten whose MRR falls below that mean floor. npm, ten seeds.}",
         r"\label{tab:protocol}",
         r"\begin{tabular}{llcccc}", r"\toprule",
         r"Graph & Accounting & gat\_dir & rcgat\_sym & gat\_time & rcgat\_time \\", r"\midrule"]
    for name, readings in (("npm", np_readings), ("Maven", mav_readings)):
        floor_a, mean = _pool_means(readings, "A")
        floor_b, _ = _pool_means(readings, "B")
        fa, fb = float(np.mean(list(floor_a.values()))), float(np.mean(list(floor_b.values())))
        for tag, floor in ((r"same reading (A/A)", fa), (r"mixed (models A, floor B)", fb)):
            parts = []
            for m in CELLS:
                if m not in mean:
                    continue
                d = mean[m] - floor
                below = sum(1 for r in readings if r["model_A"][m] < floor)
                parts.append(f"${d:+.4f}$ ({below}/{len(readings)})")
            L.append(f"{name} & {tag} & " + " & ".join(parts) + r" \\")
        L.append(f"{name} & floor (A / B) & \\multicolumn{{4}}{{c}}"
                 f"{{{fa:.4f} / {fb:.4f}}} \\\\")
        if name == "npm":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def checkpoint_table(ckpt: dict) -> str:
    L = [r"\begin{table}[t]",
         r"\caption{Robustness of the reported cells to the checkpoint policy, which the protocol "
         r"description does not fix. \texttt{best\_val} keeps the weights that maximise validation "
         r"MRR, \texttt{last} keeps the weights at the end of the budget or of early stopping, and "
         r"\texttt{init} would keep the untrained initialisation, which for the gated cell sits "
         r"exactly at the floor by construction. ``Below'' counts the seeds of ten whose test MRR "
         r"falls below the content floor of the same seed. No cell's mean falls below the floor under "
         r"either policy, and the verdict on both contrasts is unchanged. npm, ten seeds.}",
         r"\label{tab:checkpoint}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{\texttt{best\_val}} & \multicolumn{3}{c}{\texttt{last}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Cell / contrast & MRR & $\Delta$ floor & Below & MRR & $\Delta$ floor & Below \\",
         r"\midrule"]
    bv, last = ckpt.get("best_val", {}), ckpt.get("last", {})
    for model in CELLS:
        a = bv.get("models", {}).get(model)
        b = last.get("models", {}).get(model)
        if not a or not b:
            continue
        name = CELL_LABEL[model]
        if model == GATED:
            name = r"\textbf{" + name + "}"
        L.append(f"{name} & {a['test_mean']:.4f} & ${a['diff_mean']:+.4f}$ & "
                 f"{a['seeds_below_floor']} & {b['test_mean']:.4f} & ${b['diff_mean']:+.4f}$ & "
                 f"{b['seeds_below_floor']} \\\\")
    L.append(r"\midrule")
    for tag, key in ((r"gate (rcgat\_sym $-$ gat\_dir)", "gate"),
                     (r"time (gat\_time $-$ gat\_dir)", "time")):
        a = bv.get("contrasts", {}).get(key)
        b = last.get("contrasts", {}).get(key)
        if a and b:
            ca = "$" + f"{np.mean(a):+.4f}" + "$"
            cb = "$" + f"{np.mean(b):+.4f}" + "$"
            L.append(f"{tag} & \\multicolumn{{3}}{{c}}{{{ca}}} & "
                     f"\\multicolumn{{3}}{{c}}{{{cb}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = json.loads((ROOT / "results" / "summary" / "summary.json").read_text(encoding="utf-8"))
    np_readings = json.loads((ROOT / "logs" / "floor_pool_readings.json").read_text(encoding="utf-8"))
    mav_readings = json.loads((ROOT / "logs" / "floor_pool_readings_maven.json").read_text(encoding="utf-8"))
    ckpt = json.loads((ROOT / "logs" / "checkpoint_rules.json").read_text(encoding="utf-8"))

    files = {
        "table_main_results.tex": main_table(summary),
        "table_paired.tex": paired_table(summary),
        "table_protocol_sensitivity.tex": sensitivity_table(np_readings, mav_readings),
        "table_checkpoint_rule.tex": checkpoint_table(ckpt),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
        print(f"wrote {out / name} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
