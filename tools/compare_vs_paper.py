"""Build the Markdown comparison tables (ours vs the manuscript's) for the report.

Reads the summary produced by ``src/summarize.py`` and prints two Markdown tables:

* main results   : content floor + the seven cells, our test MRR vs Table 1;
* paired contrasts: our paired means/intervals/signs vs Table 2;
* selected config: our per-cell grid point / parameter count vs the Config column.

Usage::

    python tools/compare_vs_paper.py > /tmp/compare.md
    SUMMARY_JSON=/path/to/other_summary.json python tools/compare_vs_paper.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = Path(os.environ.get("SUMMARY_JSON", ROOT / "results" / "summary" / "summary.json"))
SUMMARY = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

# Table 1 of the manuscript (07_results.tex -> table_main_results.tex)
PAPER_MAIN = {
    "npm": {"content_floor": (0.1565, None), "gcn_dir": (0.1387, 0.0273), "sage_dir": (0.1587, 0.0286),
            "gatv2_dir": (0.1394, 0.0151), "gat_dir": (0.1214, 0.0233), "ragat_sym": (0.1511, 0.0127),
            "gat_time": (0.1607, 0.0172), "ragat_time": (0.1583, 0.0190)},
    "maven": {"content_floor": (0.3377, None), "gcn_dir": (0.3501, 0.0129), "sage_dir": (0.3600, 0.0278),
              "gatv2_dir": (0.3563, 0.0103), "gat_dir": (0.3370, 0.0312), "ragat_sym": (0.3681, 0.0104),
              "gat_time": (0.3399, 0.0158), "ragat_time": (0.3399, 0.0076)},
}
PAPER_CFG = {
    "npm": {"gcn_dir": "c3", "sage_dir": "c1", "gatv2_dir": "c2", "gat_dir": "c3",
            "ragat_sym": "c2", "gat_time": "c1", "ragat_time": "c4"},
    "maven": {"gcn_dir": "c4", "sage_dir": "c3", "gatv2_dir": "c1", "gat_dir": "c4",
              "ragat_sym": "c3", "gat_time": "c4", "ragat_time": "c3"},
}
# Table 2 of the manuscript (table_paired.tex)
PAPER_PAIRED = {
    "npm": {"gate": (0.0297, 0.0097, 0.0496, 9), "time": (0.0393, 0.0209, 0.0578, 10),
            "gate_under_time": (-0.0024, -0.0171, 0.0122, 6), "full_vs_gatdir": (0.0369, 0.0187, 0.0551, 9),
            "gat_vs_floor": (-0.0351, -0.0517, -0.0184, 2), "rcgat_vs_floor": (-0.0054, -0.0145, 0.0037, 4)},
    "maven": {"gate": (0.0311, 0.0094, 0.0528, 9), "time": (0.0029, -0.0238, 0.0296, 5),
              "gate_under_time": (0.0000, -0.0116, 0.0116, 5), "full_vs_gatdir": (0.0029, -0.0174, 0.0232, 4),
              "gat_vs_floor": (-0.0007, -0.0231, 0.0216, 6), "rcgat_vs_floor": (0.0304, 0.0229, 0.0379, 10)},
}
ORDER = ["content_floor", "gcn_dir", "sage_dir", "gatv2_dir", "gat_dir", "ragat_sym", "gat_time", "ragat_time"]
LABEL = {"content_floor": "Content floor", "gcn_dir": "gcn_dir", "sage_dir": "sage_dir",
         "gatv2_dir": "gatv2_dir", "gat_dir": "gat_dir", "ragat_sym": "**rcgat_sym (gate)**",
         "gat_time": "gat_time", "ragat_time": "rcgat_time"}
CONTRAST_ORDER = ["gate", "time", "gate_under_time", "full_vs_gatdir", "gat_vs_floor", "rcgat_vs_floor",
                  "gcn_vs_floor", "sage_vs_floor", "gatv2_vs_floor", "gcn_vs_gatdir", "sage_vs_gatdir",
                  "gatv2_vs_gatdir"]


def _f(v: float, sign: bool = False) -> str:
    return f"{v:+.4f}" if sign else f"{v:.4f}"


def _cell_row(dataset: str, cell: str):
    return next((r for r in SUMMARY[dataset]["table"] if r["model"] == cell), None)


def _contrast(dataset: str, name: str):
    return next((c for c in SUMMARY[dataset]["contrasts"] if c["contrast"] == name), None)


def main() -> int:
    print(f"_source: {SUMMARY_PATH}_\n")
    print("### 主结果(测试 MRR,十种子均值 ± 标准差)\n")
    print("| Cell | 本包 npm | 论文 npm | 本包 Maven | 论文 Maven |")
    print("|---|---|---|---|---|")
    for cell in ORDER:
        cells = []
        for ds in ("npm", "maven"):
            if cell == "content_floor":
                mine = f"**{_f(SUMMARY[ds]['floor']['mean'])}** ± {_f(SUMMARY[ds]['floor']['sd'])}"
            else:
                row = _cell_row(ds, cell)
                mine = "n/a" if row is None else f"**{_f(row['test_mrr_mean'])}** ± {_f(row['test_mrr_sd'])}"
            p, sd = PAPER_MAIN[ds][cell]
            theirs = _f(p) if sd is None else f"{_f(p)} ± {_f(sd)}"
            cells += [mine, theirs]
        print(f"| {LABEL[cell]} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    print()
    print("### 配对对比(每种子配对差,Student t 95% 区间)\n")
    print("| Contrast | 本包 npm | 论文 npm | 本包 Maven | 论文 Maven |")
    print("|---|---|---|---|---|")
    for name in CONTRAST_ORDER:
        cells = []
        for ds in ("npm", "maven"):
            c = _contrast(ds, name)
            mine = ("n/a" if c is None else
                    f"**{_f(c['mean'], True)}** [{_f(c['ci95_lo'], True)},{_f(c['ci95_hi'], True)}] "
                    f"sign+ {c['signs_positive']}/{c['seeds']}")
            cells.append(mine)
            if name in PAPER_PAIRED[ds]:
                m, lo, hi, sgn = PAPER_PAIRED[ds][name]
                cells.append(f"{_f(m, True)} [{_f(lo, True)},{_f(hi, True)}] sign+ {sgn}/10")
            else:
                cells.append("—")
        print(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    print()
    print("### 选中配置(逐 cell,验证集 MRR 最优)\n")
    print("| Cell | 本包 npm | 论文 npm | 本包 Maven | 论文 Maven |")
    print("|---|---|---|---|---|")
    for cell in [c for c in ORDER if c != "content_floor"]:
        cells = []
        for ds in ("npm", "maven"):
            row = _cell_row(ds, cell)
            cfg = "n/a" if row is None else str(row["cfg"]).split("_")[0]
            params = "" if row is None or row.get("params") is None else f" ({row['params']:,})"
            cells += [cfg + params, PAPER_CFG[ds][cell]]
        print(f"| {cell} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
