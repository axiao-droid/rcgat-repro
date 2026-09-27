"""Regenerate fig_datasets.pdf for the two dependency graphs of this study.

The original figure showed four panels covering npm, Maven, HepTh and HepPh.
The citation graphs are not part of this study any more (no re-run, no source
data in the reconstruction), so the revised figure covers npm and Maven only:

  top-left   complementary CDF of out-degree, log-log
  top-right  complementary CDF of in-degree, log-log
  bottom     cumulative share of nodes against first-published date

Inputs : data/graphs/{npm,maven}_graph.json.gz  (nodes[i] = {name, date, ...},
         edges = [src_idx, dst_idx] pairs)
Output : results/revision_figures/fig_datasets.pdf
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
GRAPHS = REPO / "data" / "graphs"
OUT = REPO / "results" / "revision_figures"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = {
    "npm": dict(color="#1f77b4", marker="o", label="npm"),
    "maven": dict(color="#d62728", marker="s", label="Maven"),
}


def load(name: str):
    with gzip.open(GRAPHS / f"{name}_graph.json.gz", "rt", encoding="utf-8") as fh:
        d = json.load(fh)
    nodes, edges = d["nodes"], d["edges"]
    out_deg = Counter()
    in_deg = Counter()
    for s, t in edges:
        out_deg[s] += 1
        in_deg[t] += 1
    n = len(nodes)
    dates = []
    for nd in nodes:
        raw = nd.get("date")
        if not raw:
            continue
        dates.append(date.fromisoformat(str(raw)[:10]))
    return {
        "nodes": n,
        "edges": len(edges),
        "out": [out_deg.get(i, 0) for i in range(n)],
        "in": [in_deg.get(i, 0) for i in range(n)],
        "dates": sorted(dates),
    }


def ccdf(degrees):
    deg = sorted(d for d in degrees if d > 0)
    total = len(deg)
    xs, ys, seen = [], [], 0
    for value, count in sorted(Counter(deg).items()):
        seen += count
        xs.append(value)
        ys.append(1.0 - (seen - count) / total)
    return xs, ys


def main() -> None:
    data = {name: load(name) for name in ("npm", "maven")}

    fig = plt.figure(figsize=(10.0, 7.2))
    ax_out = fig.add_subplot(2, 2, 1)
    ax_in = fig.add_subplot(2, 2, 2)
    ax_time = fig.add_subplot(2, 1, 2)

    for ax, key, title in (
        (ax_out, "out", "Out-degree"),
        (ax_in, "in", "In-degree"),
    ):
        for name, st in STYLE.items():
            xs, ys = ccdf(data[name][key])
            ax.loglog(xs, ys, st["marker"], ms=3.0, lw=1.0, color=st["color"],
                      label=st["label"], alpha=0.85)
        ax.set_title(f"{title} complementary CDF", fontsize=10)
        ax.set_xlabel("degree $k$")
        ax.set_ylabel(r"$P(K \geq k)$")
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
        ax.legend(frameon=False, fontsize=9)

    for name, st in STYLE.items():
        ds = data[name]["dates"]
        n = len(ds)
        ax_time.plot(ds, [(i + 1) / n for i in range(n)], lw=1.4,
                     color=st["color"], label=f"{st['label']} ($n={n:,}$)")
    ax_time.set_title("Cumulative share of dated nodes", fontsize=10)
    ax_time.set_xlabel("first-published date")
    ax_time.set_ylabel("cumulative share")
    ax_time.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax_time.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out = OUT / "fig_datasets.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    for name, d in data.items():
        print(f"{name}: nodes={d['nodes']} edges={d['edges']} "
              f"avg_degree={d['edges'] / d['nodes']:.4f} "
              f"max_in={max(d['in'])} max_out={max(d['out'])} "
              f"with_out={sum(1 for x in d['out'] if x > 0)}")


if __name__ == "__main__":
    main()
