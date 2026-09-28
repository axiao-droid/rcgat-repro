"""E3c -- the content floor and the structural increment, stratified.

The assessment asked for a *stronger* content floor and named the two strata
that decide how strong a floor can be: description length and category.  This
script answers the question behind the request directly, with the frozen runs:
the per-source reciprocal ranks stored by ``e4_instrument`` are split by

  * the length of the source node's description (quartiles of the pool that
    seed actually ranks, so no stratum is empty by construction),
  * the source's namespace -- the npm scope (``@scope``) or the Maven groupId,
    with the long tail pooled into ``other``,
  * the number of positive targets the source has (quartiles),

and reports, per stratum, the content floor itself and the paired difference of
each model against it.  The content-floor hypothesis predicts that the floor is
weakest on the shortest descriptions, i.e. that the structural increment, if it
exists anywhere, lives there.

Usage:
    python3 tools/ext/e3c_stratified.py
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
SRC = ROOT / "results_ext" / "e4_instrument"
GRAPHS = {d: ROOT / "data" / "graphs" / f"{d}_graph.json.gz" for d in ("npm", "maven")}
OUT = ROOT / "results_ext" / "e3c_stratified"
SEEDS = {"npm": [101, 102, 103, 104, 105], "maven": [121, 122, 123, 124, 125]}
MODELS = ["gat_dir", "ragat_sym"]
TOP_GROUPS = 6


def _load_graph(dataset: str) -> tuple[np.ndarray, list[str]]:
    with gzip.open(GRAPHS[dataset], "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    names = [n["name"] for n in payload["nodes"]]
    lens = np.array([len(n.get("text", "")) for n in payload["nodes"]], dtype=float)
    return lens, names


def _group(name: str, dataset: str) -> str:
    if dataset == "npm":
        return name.split("/")[0] if name.startswith("@") else "(unscoped)"
    return name.split(":")[0]


def _quartile_labels(values: np.ndarray) -> np.ndarray:
    q = np.quantile(values, [0.25, 0.5, 0.75])
    return np.digitize(values, q)


def paired(diff: np.ndarray) -> dict:
    n = diff.size
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    return {"n": int(n), "mean": float(diff.mean()) if n else None,
            "ci95": [float(diff.mean() - crit * se), float(diff.mean() + crit * se)] if n else None,
            "positive_share": float((diff > 0).mean()) if n else None,
            "p": float(stats.ttest_1samp(diff, 0.0).pvalue) if n > 1 else None}


def _collect(dataset: str, model: str) -> dict:
    """Per-seed per-source rows: (source, rr_model, rr_floor, n_positives)."""
    rows = []
    for seed in SEEDS[dataset]:
        path = SRC / f"ps_{dataset}_{model}_seed{seed}_zero.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            src = z["test_arrays_source"]
            rr = z["test_arrays_rr"]
            npos = z["test_arrays_n_positives"]
            fsrc = z["floor_test_arrays_source"]
            frr = z["floor_test_arrays_rr"]
        idx = {int(s): k for k, s in enumerate(fsrc)}
        common = [k for k, s in enumerate(src) if int(s) in idx]
        rows.append({
            "seed": seed,
            "source": np.array([int(src[k]) for k in common]),
            "rr_model": np.array([rr[k] for k in common]),
            "rr_floor": np.array([frr[idx[int(src[k])]] for k in common]),
            "n_positives": np.array([npos[k] for k in common]),
        })
    return rows


def _stratify(rows: list[dict], lens: np.ndarray, names: list[str], dataset: str) -> dict:
    out: dict[str, dict] = {}
    # length quartiles, computed inside each seed's ranked pool
    length_buckets: dict[str, dict] = {}
    pos_buckets: dict[str, dict] = {}
    group_buckets: dict[str, dict] = {}
    group_counts: dict[str, int] = {}
    for r in rows:
        src = r["source"]
        L = lens[src]
        q = _quartile_labels(L)
        qp = _quartile_labels(r["n_positives"].astype(float))
        for label, bucket in [("length_q1", 0), ("length_q2", 1), ("length_q3", 2), ("length_q4", 3)]:
            m = q == bucket
            if m.any():
                d = length_buckets.setdefault(label, {"d": [], "floor": []})
                d["d"].append(r["rr_model"][m] - r["rr_floor"][m])
                d["floor"].append(r["rr_floor"][m])
        for label, bucket in [("pos_q1", 0), ("pos_q2", 1), ("pos_q3", 2), ("pos_q4", 3)]:
            m = qp == bucket
            if m.any():
                d = pos_buckets.setdefault(label, {"d": [], "floor": []})
                d["d"].append(r["rr_model"][m] - r["rr_floor"][m])
                d["floor"].append(r["rr_floor"][m])
        for g in {_group(names[s], dataset) for s in src.tolist()}:
            group_counts[g] = group_counts.get(g, 0) + int(sum(1 for s in src.tolist() if _group(names[s], dataset) == g))
    top = {g for g, _ in sorted(group_counts.items(), key=lambda kv: -kv[1])[:TOP_GROUPS]}
    for r in rows:
        src = r["source"]
        labels = np.array([_group(names[s], dataset) if _group(names[s], dataset) in top else "other"
                           for s in src.tolist()])
        for g in sorted(set(labels.tolist())):
            m = labels == g
            d = group_buckets.setdefault(g, {"d": [], "floor": []})
            d["d"].append(r["rr_model"][m] - r["rr_floor"][m])
            d["floor"].append(r["rr_floor"][m])

    def summarise(buckets: dict) -> dict:
        res = {}
        for label, d in sorted(buckets.items()):
            diff = np.concatenate(d["d"])
            floor = np.concatenate(d["floor"])
            st = paired(diff)
            st["floor_mean"] = float(floor.mean())
            res[label] = st
        return res

    out["length_quartiles"] = summarise(length_buckets)
    out["positives_quartiles"] = summarise(pos_buckets)
    out["groups"] = summarise(group_buckets)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    lines = ["# E3c -- content floor and structural increment, stratified", "",
             "Per-source paired difference (model minus content floor, test stage) of the frozen",
             "zero-initialised runs, split by the length of the source's description, by the source's",
             "namespace, and by the number of positive targets. ``floor`` is the content floor itself",
             "in that stratum: the picture to look for is a weak floor together with a large",
             "structural increment.", ""]
    for dataset in GRAPHS:
        lens, names = _load_graph(dataset)
        for model in MODELS:
            rows = _collect(dataset, model)
            if not rows:
                continue
            entry = _stratify(rows, lens, names, dataset)
            entry["seeds"] = [r["seed"] for r in rows]
            entry["sources_mean"] = float(np.mean([r["source"].size for r in rows]))
            report[f"{dataset}|{model}"] = entry
            lines += [f"## {dataset} / {model}", "",
                      f"{entry['sources_mean']:.0f} rankable sources per seed "
                      f"({len(rows)} seeds)", ""]
            for title, key in (("description length (quartiles)", "length_quartiles"),
                               ("positives per source (quartiles)", "positives_quartiles"),
                               ("namespace / groupId", "groups")):
                lines += [f"**{title}**", "",
                          "| stratum | sources | floor MRR | Δ vs floor | 95% CI | share > 0 |",
                          "|---|---|---|---|---|---|"]
                for label, st in entry[key].items():
                    lines.append(f"| {label} | {st['n']} | {st['floor_mean']:.4f} | "
                                 f"{st['mean']:+.4f} | [{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] | "
                                 f"{100 * st['positive_share']:.0f}% |")
                lines.append("")
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
