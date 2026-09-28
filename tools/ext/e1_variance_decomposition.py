"""E1 -- variance decomposition of the reported metrics.

The external assessment asks how much of the reported spread is *split* noise
(the ten seeded train/test partitions) rather than *model* differences, because
that ratio decides whether a +0.005 MRR gap is a finding or a coin flip.

Inputs are the frozen per-run JSON files only -- no training, no new numbers
invented.  Output: a JSON blob plus a markdown fragment.

Analyses
--------
1. per (dataset, model) across-seed SD of test MRR, next to the content floor's
   own across-seed SD (the floor is deterministic given the split, so that SD
   *is* the split component);
2. two-way ANOVA with replication on test MRR: dataset x model, error = the ten
   seeded partitions inside each cell, reported as variance components
   (sigma^2_split, sigma^2_interaction, sigma^2_model, sigma^2_dataset);
3. the same on the paired contrast Delta = model - floor, which removes the
   per-split difficulty level;
4. rank stability: how often the per-seed best model equals the pooled best, and
   the mean pairwise Spearman correlation of model rankings across seeds;
5. the seed correlation of test MRR between models (high = the split, not the
   model, is moving the number).

Usage:
    python3 tools/ext/e1_variance_decomposition.py [--roots repro/results ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
OUT = ROOT / "results_ext" / "e1_variance"


def _load_runs(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "runs").glob("run_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if "test" not in d or "floor" not in d:
            continue
        rows.append({
            "dataset": d["dataset"],
            "model": d["registry_model"] if d.get("registry_model") else d["model"],
            "seed": int(d["seed"]),
            "grid_id": d.get("grid_id"),
            "test": float(d["test"]["mrr"]),
            "val": float(d["val"]["mrr"]),
            "floor_test": float(d["floor"]["mrr"]),
            "floor_val": float(d["floor_val"]["mrr"]) if d.get("floor_val") else float("nan"),
            "delta": float(d["test"]["mrr"]) - float(d["floor"]["mrr"]),
            "init_val_mrr": d.get("init_val_mrr"),
            "best_epoch": d.get("best_epoch"),
            "epochs_run": d.get("epochs_run"),
            "source": str(path),
        })
    return rows


def _matrix(rows: list[dict], datasets: list[str], models: list[str], key: str):
    """(n_dataset, n_model, n_seed) array.

    The two datasets do not share seed *values* (npm uses 101..110, Maven
    121..130), so the seed axis is the within-dataset rank of the seed value --
    seed k of npm and seed k of Maven are the k-th partition under the same
    split rule, which is the axis the ANOVA wants.
    """
    per_ds_seeds = {ds: sorted({r["seed"] for r in rows if r["dataset"] == ds}) for ds in datasets}
    width = max((len(v) for v in per_ds_seeds.values()), default=0)
    arr = np.full((len(datasets), len(models), width), np.nan)
    for r in rows:
        if r["dataset"] not in datasets or r["model"] not in models:
            continue
        i = datasets.index(r["dataset"])
        j = models.index(r["model"])
        k = per_ds_seeds[r["dataset"]].index(r["seed"])
        arr[i, j, k] = r[key]
    return arr, per_ds_seeds


def _two_way_anova(arr: np.ndarray) -> dict:
    """Balanced two-way ANOVA with replication; returns sums of squares."""
    a, b, n = arr.shape
    grand = arr.mean()
    row = arr.mean(axis=(1, 2))
    col = arr.mean(axis=(0, 2))
    cell = arr.mean(axis=2)
    ss_a = b * n * ((row - grand) ** 2).sum()
    ss_b = a * n * ((col - grand) ** 2).sum()
    ss_ab = n * ((cell - row[:, None] - col[None, :] + grand) ** 2).sum()
    ss_err = ((arr - cell[:, :, None]) ** 2).sum()
    ss_tot = ss_a + ss_b + ss_ab + ss_err
    df_a, df_b, df_ab, df_err = a - 1, b - 1, (a - 1) * (b - 1), a * b * (n - 1)
    ms_a = ss_a / df_a if df_a else 0.0
    ms_b = ss_b / df_b if df_b else 0.0
    ms_ab = ss_ab / df_ab if df_ab else 0.0
    ms_err = ss_err / df_err if df_err else 0.0
    # random-effects variance components (Searle/ANOVA estimator)
    comp_split = ms_err
    comp_ab = max((ms_ab - ms_err) / n, 0.0)
    comp_model = max((ms_b - ms_ab) / (a * n), 0.0)
    comp_dataset = max((ms_a - ms_ab) / (b * n), 0.0)
    total_comp = comp_split + comp_ab + comp_model + comp_dataset
    return {
        "ss": {"dataset": ss_a, "model": ss_b, "interaction": ss_ab, "split": ss_err},
        "ss_share": {k: (v / ss_tot if ss_tot else 0.0)
                     for k, v in (("dataset", ss_a), ("model", ss_b),
                                  ("interaction", ss_ab), ("split", ss_err))},
        "df": {"dataset": df_a, "model": df_b, "interaction": df_ab, "split": df_err},
        "ms": {"dataset": ms_a, "model": ms_b, "interaction": ms_ab, "split": ms_err},
        "variance_components": {"dataset": comp_dataset, "model": comp_model,
                                "interaction": comp_ab, "split": comp_split},
        "variance_share": {k: (v / total_comp if total_comp else 0.0)
                           for k, v in (("dataset", comp_dataset), ("model", comp_model),
                                        ("interaction", comp_ab), ("split", comp_split))},
        "sd_components": {k: math.sqrt(v) for k, v in
                          (("dataset", comp_dataset), ("model", comp_model),
                           ("interaction", comp_ab), ("split", comp_split))},
        "n_seeds": n,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def _rank_stability(arr: np.ndarray, model_names: list[str]) -> dict:
    """How the per-seed model ranking moves relative to the pooled ranking."""
    pooled = model_names[int(np.argmax(arr.mean(axis=1)))]
    per_seed = [model_names[int(np.argmax(arr[:, k]))] for k in range(arr.shape[1])]
    cors = [_spearman(arr[:, k], arr.mean(axis=1)) for k in range(arr.shape[1])]
    cross = []
    for k, l in combinations(range(arr.shape[1]), 2):
        cross.append(_spearman(arr[:, k], arr[:, l]))
    return {
        "pooled_best": pooled,
        "per_seed_best": per_seed,
        "share_seed_best_agrees": float(np.mean([p == pooled for p in per_seed])),
        "mean_spearman_seed_vs_pooled": float(np.mean(cors)),
        "min_spearman_seed_vs_pooled": float(np.min(cors)),
        "mean_spearman_seed_pairs": float(np.mean(cross)) if cross else None,
    }


def analyse(rows: list[dict]) -> dict:
    datasets = sorted({r["dataset"] for r in rows})
    models = sorted({r["model"] for r in rows})
    out: dict = {"datasets": datasets, "models": models,
                 "n_runs": len(rows), "sources": sorted({r["source"] for r in rows})}

    # 1. per-cell across-seed spread, and the floor's own
    cells = {}
    for ds in datasets:
        for m in models:
            v = np.array([r["test"] for r in rows if r["dataset"] == ds and r["model"] == m])
            dv = np.array([r["delta"] for r in rows if r["dataset"] == ds and r["model"] == m])
            fl = np.array([r["floor_test"] for r in rows if r["dataset"] == ds and r["model"] == m])
            if v.size == 0:
                continue
            cells[f"{ds}|{m}"] = {
                "n_seeds": int(v.size),
                "test_mean": float(v.mean()), "test_sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "delta_mean": float(dv.mean()), "delta_sd": float(dv.std(ddof=1)) if dv.size > 1 else 0.0,
                "floor_mean": float(fl.mean()), "floor_sd": float(fl.std(ddof=1)) if fl.size > 1 else 0.0,
            }
    out["cells"] = cells

    # 2/3. variance decomposition on test MRR and on the paired delta
    for key, label in (("test", "test_mrr"), ("delta", "delta_vs_floor"), ("val", "val_mrr")):
        arr, seeds = _matrix(rows, datasets, models, key)
        if np.isnan(arr).any():
            out[label] = {"error": "unbalanced cells; two-way ANOVA skipped",
                          "missing": int(np.isnan(arr).sum())}
            continue
        block = {"anova": _two_way_anova(arr), "seeds": seeds}
        for ds in datasets:
            a = arr[datasets.index(ds)]
            block[f"rank_stability_{ds}"] = _rank_stability(a, models)
            block[f"seed_sd_{ds}"] = float(a.std(axis=1, ddof=1).mean())
            block[f"model_sd_{ds}"] = float(a.mean(axis=1).std(ddof=1))
            # correlation between models across seeds
            cors = np.corrcoef(a)
            off = cors[np.triu_indices(len(models), 1)]
            block[f"seed_corr_between_models_{ds}"] = {
                "mean": float(np.mean(off)), "min": float(np.min(off)), "max": float(np.max(off)),
            }
        out[label] = block

    # 5. the gate contrast specifically (ragat_sym - gat_dir), per dataset
    gate = {}
    for ds in datasets:
        pairs = {}
        for r in rows:
            if r["dataset"] != ds:
                continue
            pairs.setdefault(r["seed"], {})[r["model"]] = r["test"]
        d = np.array([p["ragat_sym"] - p["gat_dir"] for p in pairs.values()
                      if "ragat_sym" in p and "gat_dir" in p])
        dg = np.array([p["ragat_sym"] - p["gat_dir"] for p in pairs.values()
                       if "ragat_sym" in p and "gat_dir" in p])  # gate = sym - dir on test
        if d.size:
            se = d.std(ddof=1) / math.sqrt(d.size) if d.size > 1 else 0.0
            gate[ds] = {
                "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
                "n_seeds": int(d.size), "positive_seeds": int((d > 0).sum()),
                "se": float(se),
                "ci95": [float(d.mean() - 2.262 * se), float(d.mean() + 2.262 * se)],
            }
        gate[f"{ds}_note"] = ("gate contrast here is the test-MRR difference "
                              "ragat_sym - gat_dir; the manuscript's gate contrast "
                              "is the same pair, so signs match")
    out["gate_contrast"] = gate
    return out


def to_markdown(res: dict) -> str:
    L: list[str] = []
    L.append("# E1 -- variance decomposition (seed/split vs model)\n")
    L.append(f"Runs read: {res['n_runs']} (datasets {', '.join(res['datasets'])}; "
             f"models {', '.join(res['models'])}).\n")
    for label, title in (("test_mrr", "test MRR"), ("delta_vs_floor", "Delta vs content floor (paired)")):
        b = res.get(label, {})
        if "anova" not in b:
            continue
        vc, vs, sc = b["anova"]["variance_components"], b["anova"]["variance_share"], b["anova"]["sd_components"]
        L.append(f"\n## {title}\n")
        L.append("| component | variance | share | sd |")
        L.append("|---|---|---|---|")
        for k in ("split", "model", "interaction", "dataset"):
            L.append(f"| {k} | {vc[k]:.3e} | {100 * vs[k]:.1f}% | {sc[k]:.5f} |")
        for ds in res["datasets"]:
            rs = b.get(f"rank_stability_{ds}")
            if not rs:
                continue
            L.append(f"\n**{ds}** -- pooled best `{rs['pooled_best']}`; per-seed best agrees on "
                     f"{100 * rs['share_seed_best_agrees']:.0f}% of seeds; mean Spearman(seed, pooled) "
                     f"= {rs['mean_spearman_seed_vs_pooled']:.2f} (min {rs['min_spearman_seed_vs_pooled']:.2f}); "
                     f"mean across-seed SD {b[f'seed_sd_{ds}']:.4f} vs across-model SD "
                     f"{b[f'model_sd_{ds}']:.4f}; mean seed-correlation between models "
                     f"{b[f'seed_corr_between_models_{ds}']['mean']:.3f} "
                     f"(min {b[f'seed_corr_between_models_{ds}']['min']:.3f}).")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[str(ROOT / "results")])
    ap.add_argument("--tag", default="e1_variance")
    args = ap.parse_args()

    rows: list[dict] = []
    for r in args.roots:
        rows.extend(_load_runs(Path(r)))
    OUT.mkdir(parents=True, exist_ok=True)
    res = analyse(rows)
    (OUT / f"{args.tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (OUT / f"{args.tag}.md").write_text(to_markdown(res), encoding="utf-8")
    print(json.dumps({"runs": res["n_runs"], "out": str(OUT / args.tag)}, indent=2))
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
