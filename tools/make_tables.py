"""Emit manuscript-ready LaTeX tables from ``results/``.

The three tables mirror the layout of the submission package
(``table_main_results.tex``, ``table_paired.tex``, ``table_dataset_stats.tex``)
so the numbers can be swapped in one go after a re-run:

    python tools/make_tables.py --datasets npm maven --out results/tables

Everything is read from the run/floor/diag artefacts -- nothing is hard-coded,
which is the defect that made the recovered ``summarize6.py`` unverifiable.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
MANIFEST = json.loads((PROJECT / "MANIFEST.json").read_text(encoding="utf-8"))
CELLS = MANIFEST["models"]["cells"]
BASELINES = MANIFEST["models"]["baselines"]
MODELS = CELLS + BASELINES

LABEL = {
    "content_floor": "Content floor",
    "gcn_dir": "gcn\\_dir (external GCN)",
    "sage_dir": "sage\\_dir (external GraphSAGE)",
    "gatv2_dir": "gatv2\\_dir (external GATv2)",
    "gat_dir": "gat\\_dir (directed GAT)",
    "ragat_sym": "rcgat\\_sym (gate)",
    "gat_time": "gat\\_time (time branch)",
    "ragat_time": "rcgat\\_time (gate $+$ time)",
}
CONTRAST_LABEL = {
    "gate": "gate (rcgat\\_sym $-$ gat\\_dir)",
    "time": "time (gat\\_time $-$ gat\\_dir)",
    "gate_under_time": "gate under time (rcgat\\_time $-$ gat\\_time)",
    "full_vs_gatdir": "full (rcgat\\_time $-$ gat\\_dir)",
    "gcn_vs_gatdir": "gcn\\_dir $-$ gat\\_dir",
    "sage_vs_gatdir": "sage\\_dir $-$ gat\\_dir",
    "gatv2_vs_gatdir": "gatv2\\_dir $-$ gat\\_dir",
    "gat_vs_floor": "ungated vs floor (gat\\_dir $-$ floor)",
    "rcgat_vs_floor": "gate vs floor (rcgat\\_sym $-$ floor)",
    "gcn_vs_floor": "gcn\\_dir $-$ floor",
    "sage_vs_floor": "sage\\_dir $-$ floor",
    "gatv2_vs_floor": "gatv2\\_dir $-$ floor",
}
GRAPH_LABEL = {"npm": "npm", "maven": "Maven"}


def _num(v: float) -> str:
    """0.1565 -> 0.1565 ; keeps four decimals like the submission tables."""
    return f"{v:.4f}"


def _signed(v: float) -> str:
    return f"${'+' if v >= 0 else '-'}{abs(v):.4f}$"


def main_results(summary: dict, datasets: list[str]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\caption{Test MRR, hits@10, and hits@100 (mean over ten seeds $\\pm$ sample standard",
        "deviation) of the four model cells, the content floor, and the three external baselines.",
        "Hyperparameters were selected per cell on validation MRR with a single tuning seed and",
        "frozen for the ten final seeds. $\\Delta$ floor is the mean test MRR minus the content floor.}",
        "\\label{tab:main}",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Graph & Cell & MRR & Hits@10 & Hits@100 & $\\Delta$ floor & Config & Params \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        block = summary.get(dataset)
        if not block:
            continue
        floor = block["floor"]["mean"]
        rows = []
        rows.append(("content_floor", floor, 0.0, None, None, None, None))
        by_model = {r["model"]: r for r in block["table"]}
        for model in BASELINES + CELLS:
            r = by_model.get(model)
            if not r:
                continue
            rows.append((model, r["test_mrr_mean"], r["test_mrr_sd"], r["hits@10"], r["hits@100"],
                         r["cfg"], r["params"]))
        for i, (model, mrr, sd, h10, h100, cfg, params) in enumerate(rows):
            label = LABEL.get(model, model)
            mrr_tex = _num(mrr) if sd == 0 else f"{_num(mrr)} $\\pm$ {_num(sd)}"
            h10_tex = "--" if h10 is None else _num(h10)
            h100_tex = "--" if h100 is None else _num(h100)
            if model == "content_floor":
                d_tex = "--"
            else:
                d = mrr - floor
                d_tex = f"${'+' if d >= 0 else '-'}{abs(d):.4f}$"
            cfg_tex = "--" if cfg is None else str(cfg).split("_")[0]
            p_tex = "--" if params is None else f"{params:,}".replace(",", "{,}")
            if model in CELLS:
                label = "\\textbf{" + label + "}"
                mrr_tex = "\\textbf{" + mrr_tex + "}"
            lines.append(f"{GRAPH_LABEL.get(dataset, dataset)} & {label} & {mrr_tex} & {h10_tex} & "
                         f"{h100_tex} & {d_tex} & {cfg_tex} & {p_tex} \\\\")
        if dataset != datasets[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def paired_table(summary: dict, datasets: list[str]) -> str:
    order = ["gate", "time", "gate_under_time", "full_vs_gatdir", "gat_vs_floor", "rcgat_vs_floor"]
    lines = [
        "\\begin{table}[t]",
        "\\caption{Pre-registered paired contrasts. Each seed contributes one within-seed",
        "difference; the interval is a Student $t$ 95\\% confidence interval on the mean paired",
        "difference (df $=9$). Sign$+$ is the number of seeds in which the difference is positive.}",
        "\\label{tab:paired}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Graph & Contrast & Mean paired difference & 95\\% CI & Sign$+$ / 10 \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        block = summary.get(dataset)
        if not block:
            continue
        found = {c["contrast"]: c for c in block["contrasts"]}
        for name in order:
            c = found.get(name)
            if not c:
                continue
            ci = f"[{_signed(c['ci95_lo'])[1:-1]},{_signed(c['ci95_hi'])[1:-1]}]" \
                if c["ci95_lo"] is not None else "n/a"
            lines.append(f"{GRAPH_LABEL.get(dataset, dataset)} & {CONTRAST_LABEL.get(name, name)} & "
                         f"{_signed(c['mean'])} & {ci} & {c['signs_positive']} \\\\")
        if dataset != datasets[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def dataset_table(datasets: list[str]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\caption{Software dependency networks, and the content floor. The floor is a train-only",
        "TF-IDF plus truncated-SVD (300-d) cosine ranker evaluated with the same per-source",
        "full-filtered protocol. Test sources and positive edges are means over the ten seeded test",
        "windows; the candidate pool size is the mean per source.}",
        "\\label{tab:datasets}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Graph & Nodes & Directed edges & Content floor test MRR & Test sources (positives) & Mean candidates \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        with gzip.open(PROJECT / "data" / "graphs" / f"{dataset}_graph.json.gz", "rb") as fh:
            stats = json.loads(fh.read().decode())["stats"]
        diag_path = PROJECT / "results" / "diag" / f"diag_{dataset}.json"
        summary_path = PROJECT / "results" / "summary" / "summary.json"
        floor = "-"
        if summary_path.exists():
            floor = _num(json.loads(summary_path.read_text(encoding="utf-8"))[dataset]["floor"]["mean"])
        src = pos = cand = "-"
        if diag_path.exists():
            m = json.loads(diag_path.read_text(encoding="utf-8"))["mean"]
            src, pos, cand = f"{m['test_sources']:.0f}", f"{m['positives']:.0f}", f"{m['candidates_mean']:,.0f}"
        name = "npm (JavaScript registry)" if dataset == "npm" else "Maven (Java artifact repos.)"
        nodes = f"{stats['nodes']:,}".replace(",", "{,}")
        edges = f"{stats['edges']:,}".replace(",", "{,}")
        cand_tex = cand.replace(",", "{,}")
        lines.append(f"{name} & {nodes} & {edges} & {floor} & {src} ({pos}) & {cand_tex} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=MANIFEST["datasets"])
    ap.add_argument("--out", default=str(PROJECT / "results" / "tables"))
    args = ap.parse_args()
    summary_path = PROJECT / "results" / "summary" / "summary.json"
    if not summary_path.exists():
        print("results/summary/summary.json missing -- run src/summarize.py first")
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_main_results.tex").write_text(main_results(summary, args.datasets), encoding="utf-8")
    (out / "table_paired.tex").write_text(paired_table(summary, args.datasets), encoding="utf-8")
    (out / "table_dataset_stats.tex").write_text(dataset_table(args.datasets), encoding="utf-8")
    print(f"written: {out}/table_main_results.tex, table_paired.tex, table_dataset_stats.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
