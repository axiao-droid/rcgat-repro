"""Compare replication runs made under different checkpoint policies.

The manuscript never states which checkpoint is reported, and that choice decides
whether "training made the model *worse* than the content floor" is even
reachable: a checkpoint picked by validation MRR can only sit at or above the
validation-selected optimum, whereas a "last epoch" policy can keep training past
it.  The user does not remember which policy the original code used, so this tool
runs both and reports both.

For every results root it reads ``runs/run_<dataset>_<model>_seed<seed>.json``
(each file carries its own floor and fold metadata, so the roots are never mixed
up), and prints

* the floor-restricted subset of the five manuscript cells / baselines,
* mean test MRR and the mean paired difference against that seed's floor,
* how many of the ten seeds land *below* the floor, which is the pattern the
  manuscript reports for npm (gat_dir 0.1214 vs floor 0.1565),
* the two within-model contrasts the paper's Table 2 is built on
  (gate: ragat_sym - gat_dir, time: gat_time - gat_dir) paired per seed,
* the fraction of epochs reported as ``selected_epoch`` vs ``best_epoch``, so a
  policy is visible in the artifacts rather than assumed.

Paired intervals use the paper's pre-registered convention, a *paired Student t*
interval at 95 % (``t(0.975, n-1) * sd / sqrt(n)``, ddof=1), i.e. the same numbers
``src/summarize.py`` and ``src/audit.py`` produce.  An earlier revision of this
tool used the normal quantile 1.96; at n = 10 that is 13 % narrower than the t
quantile and would not match ``results/summary/summary.json``.

Usage::

    python tools/compare_checkpoint_rules.py --dataset npm \
        best_val=results last=results_ckpt_last
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# paper display name -> code name (gated.MODEL_ALIASES reverse)
ALIASES = {"rcgat_sym": "ragat_sym", "rcgat_time": "ragat_time"}
CELLS = ["gat_dir", "ragat_sym", "gat_time", "ragat_time"]
BASELINES = ["gcn_dir", "sage_dir", "gatv2_dir"]


def floor_mrr(rec: dict) -> float:
    """Test-stage floor MRR of the same fold.

    ``run_*.json`` stores the floor as a flat metric dict (``{"mrr": ...}``);
    older artifacts nest it per stage.  Both shapes are accepted so a results
    root produced before the schema settled can still be compared.
    """
    floor = rec["floor"]
    if isinstance(floor, dict) and isinstance(floor.get("test"), dict):
        return float(floor["test"]["mrr"])
    return float(floor["mrr"])


def load_runs(root: Path, dataset: str) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for path in sorted((root / "runs").glob(f"run_{dataset}_*_seed*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        raw = rec.get("registry_model") or rec["model"]
        model = ALIASES.get(raw, raw)   # artifacts written before the fix store the alias
        out[(model, int(rec["seed"]))] = rec
    return out


def _paired(diffs: list[float]) -> str:
    a = np.asarray(diffs, dtype=float)
    if a.size == 0:
        return "n/a"
    half = (float(stats.t.ppf(0.975, a.size - 1) * a.std(ddof=1) / np.sqrt(a.size))
            if a.size > 1 else 0.0)
    return f"{a.mean():+.4f} [{a.mean() - half:+.4f},{a.mean() + half:+.4f}] {int((a > 0).sum())}/{a.size}"


def report(tag: str, root: Path, dataset: str) -> dict:
    runs = load_runs(root, dataset)
    if not runs:
        print(f"## {tag}: no runs under {root}")
        return {}
    seeds = sorted({s for _, s in runs})
    print(f"\n## {tag}  ({root}, {len(runs)} runs, seeds {seeds[0]}..{seeds[-1]})")
    print("| model | mean test MRR | mean floor | mean diff vs floor | seeds below floor | selected<best epochs |")
    print("|---|---|---|---|---|---|")
    floors = {s: floor_mrr(runs[(m, s)]) for (m, s) in runs}
    summary: dict = {"seeds": seeds, "models": {}}
    for model in CELLS + BASELINES:
        pair = [(runs[(model, s)]["test"]["mrr"], floors[s]) for s in seeds if (model, s) in runs]
        if not pair:
            continue
        test = np.array([p[0] for p in pair])
        floor = np.array([p[1] for p in pair])
        diff = test - floor
        sel = [runs[(model, s)].get("selected_epoch") for s in seeds if (model, s) in runs]
        best = [runs[(model, s)].get("best_epoch") for s in seeds if (model, s) in runs]
        behind = sum(1 for a, b in zip(sel, best) if a is not None and b is not None and a < b)
        print(f"| {model} | {test.mean():.4f} ± {test.std(ddof=1):.4f} | {floor.mean():.4f} | "
              f"{diff.mean():+.4f} | {int((diff < 0).sum())}/{len(diff)} | {behind}/{len(sel)} |")
        summary["models"][model] = {
            "test_mean": float(test.mean()), "test_sd": float(test.std(ddof=1)) if len(test) > 1 else 0.0,
            "floor_mean": float(floor.mean()), "diff_mean": float(diff.mean()),
            "seeds_below_floor": int((diff < 0).sum()), "n_seeds": int(len(diff)),
            "selected_epochs": sel, "best_epochs": best,
        }
    for name, base, treat in (("gate", "gat_dir", "ragat_sym"), ("time", "gat_dir", "gat_time"),
                              ("gate_under_time", "gat_time", "ragat_time")):
        diffs = [runs[(treat, s)]["test"]["mrr"] - runs[(base, s)]["test"]["mrr"]
                 for s in seeds if (treat, s) in runs and (base, s) in runs]
        if diffs:
            print(f"- paired {name} ({treat} − {base}): {_paired(diffs)}")
            summary.setdefault("contrasts", {})[name] = diffs
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="npm")
    ap.add_argument("--out", default=None, help="write the collected numbers as JSON here")
    ap.add_argument("roots", nargs="*", default=["best_val=results", "last=results_ckpt_last"])
    args = ap.parse_args()

    collected = {}
    for spec in args.roots:
        tag, _, path = spec.partition("=")
        collected[tag] = report(tag, (ROOT / path) if not Path(path).is_absolute() else Path(path), args.dataset)
    if args.out:
        Path(args.out).write_text(json.dumps(collected, indent=2), encoding="utf-8")
        print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
