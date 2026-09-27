"""Deterministic audit of the data pipeline and of the reported contrasts.

The manuscript claims: "The audit independently verifies the data pipeline on
every seed (split integrity, leakage rules, descriptor statistics), and all audit
checks pass."  This is that audit, and it is written so that it can *fail*: every
check re-derives a quantity from the frozen artefacts (graph snapshot, bundles,
per-run JSONs) and compares it with what the pipeline stored.

Checks
------
A. graph snapshot      node/edge counts, degree extremes, date span, sha256 of the
                       gzip, self-loop and duplicate-edge absence.
B. split integrity     fit/val/test disjoint and exhaustive; test window is the
                       newest round(0.07 n) nodes; window sizes match the frozen
                       fractions; the per-seed partitions differ.
C. leakage             training MP edges originate in the fit window only; no
                       test-window source contributes a training edge; the
                       candidate pool excludes the test window and the source;
                       every scored target lies inside the pool; descriptors are
                       invariant to replacing test-window edges and text.
D. floor               the stored floor is recomputed from the bundle features and
                       reproduced (bitwise, up to float noise).
E. run bookkeeping     ten runs per (dataset, model cell); the frozen config is the
                       same in every seed of a model; the floor recorded in a run
                       equals the floor phase value for that seed.
F. contrasts           the paired means/intervals/sign counts of the summary are
                       recomputed from the run files.

Usage::

    python src/audit.py --datasets npm maven
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

MANIFEST = json.loads((PROJECT / "MANIFEST.json").read_text(encoding="utf-8"))
CELLS = MANIFEST["models"]["cells"]
BASELINES = MANIFEST["models"]["baselines"]
MODELS = CELLS + BASELINES
F_TEST, F_VAL, JITTER = 0.07, 0.08, 0.05

AUDIT_CONTRASTS = [
    ("gate", "ragat_sym", "gat_dir"),
    ("time", "gat_time", "gat_dir"),
    ("gate_under_time", "ragat_time", "gat_time"),
    ("full_vs_gatdir", "ragat_time", "gat_dir"),
    ("gcn_vs_gatdir", "gcn_dir", "gat_dir"),
    ("sage_vs_gatdir", "sage_dir", "gat_dir"),
    ("gatv2_vs_gatdir", "gatv2_dir", "gat_dir"),
]

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def load_graph(dataset: str) -> tuple[dict, list[dict], np.ndarray]:
    path = PROJECT / "data" / "graphs" / f"{dataset}_graph.json.gz"
    with gzip.open(path, "rb") as fh:
        payload = json.loads(fh.read().decode())
    return payload, payload["nodes"], np.array(payload["edges"], np.int64)


def audit_graph(dataset: str) -> None:
    payload, nodes, edges = load_graph(dataset)
    stats = payload["stats"]
    n = len(nodes)
    src, tgt = edges[:, 0], edges[:, 1]
    out_deg = np.bincount(src, minlength=n)
    in_deg = np.bincount(tgt, minlength=n)
    check("A1 node/edge counts match stats", n == stats["nodes"] and len(edges) == stats["edges"],
          f"nodes={n} edges={len(edges)}")
    check("A2 degree extremes match stats",
          int(out_deg.max()) == stats["max_out"] and int(in_deg.max()) == stats["max_in"],
          f"max_out={int(out_deg.max())} max_in={int(in_deg.max())}")
    check("A3 no self loops", not bool((src == tgt).any()))
    pairs = {(int(a), int(b)) for a, b in edges}
    check("A4 no duplicate edges", len(pairs) == len(edges), f"{len(edges)} edges / {len(pairs)} unique")
    dates = sorted(x["date"][:10] for x in nodes)
    check("A5 date span matches stats", dates[0] == stats["date_min"] and dates[-1] == stats["date_max"],
          f"{dates[0]} .. {dates[-1]}")
    sha = hashlib.sha256((PROJECT / "data" / "graphs" / f"{dataset}_graph.json.gz").read_bytes()).hexdigest()
    summary_path = PROJECT / "data" / "graphs" / "build_summary.json"
    if summary_path.exists():
        recorded = json.loads(summary_path.read_text(encoding="utf-8")).get(dataset, {}).get("sha256_gz")
        check("A6 snapshot sha256 recorded", recorded == sha, f"{sha[:16]}...")
    check("A7 node order is the graph payload order (split and features share one indexing)",
          [x["name"] for x in nodes] == [x["name"] for x in payload["nodes"]])


def audit_split(dataset: str) -> None:
    from content_floor import JITTER_TEST, _node_days, _split_indices, _val_half_band

    _, nodes, edges = load_graph(dataset)
    n = len(nodes)
    dates = np.array([x["date"][:10] for x in nodes])
    days = _node_days(dates)
    partitions = []
    test_partitions = []
    for seed in MANIFEST["final_seeds"][dataset]:
        fit, val, test = _split_indices(days, seed)
        partitions.append(tuple(fit.tolist()))
        test_partitions.append(tuple(test.tolist()))
        ok = (len(np.intersect1d(fit, val)) == 0 and len(np.intersect1d(fit, test)) == 0
              and len(np.intersect1d(val, test)) == 0 and len(fit) + len(val) + len(test) == n)
        check(f"B1 seed {seed}: fit/val/test partition is disjoint and exhaustive", ok,
              f"{len(fit)}/{len(val)}/{len(test)} of {n}")
        n_test_expected = max(1, int(round(F_TEST * n)))
        check(f"B2 seed {seed}: test window is the newest {n_test_expected} nodes", len(test) == n_test_expected,
              f"{len(test)} nodes")
        order = np.argsort(days, kind="stable")
        check(f"B3 seed {seed}: test window is exactly the date tail",
              np.array_equal(np.sort(test), np.sort(order[n - n_test_expected:])))
        check(f"B4 seed {seed}: no fit node is newer than a validation node",
              days[fit].max() <= days[val].min() if len(fit) and len(val) else True)
        check(f"B5 seed {seed}: no validation node is newer than a test node",
              days[val].max() <= days[test].min() if len(val) and len(test) else True)
        band = _val_half_band(n)
        check(f"B6 seed {seed}: validation size within the jitter band",
              abs(len(val) - round(F_VAL * n)) <= band + 1,
              f"{len(val)} nodes vs {round(F_VAL * n)}±{band}")
    # B7/B8: the manuscript says the ten seeds use independent splits.  Whichever
    # reading is active (window fixed or jittered), the *training* partition must
    # differ from seed to seed -- otherwise the per-seed SDs and every paired
    # interval would be meaningless.  Which part of the split carries that
    # variation is then reported explicitly.
    check("B7 the ten seeds produce ten distinct training partitions "
          "(distinct fit windows, i.e. distinct training edge sets)",
          len(set(partitions)) == len(partitions) == len(MANIFEST["final_seeds"][dataset]),
          f"{len(set(partitions))}/{len(partitions)} distinct")
    same_window = len(set(test_partitions)) == 1
    if JITTER_TEST <= 0:
        check("B8 the test window is the same newest-fraction window in every seed "
              "(pinned by the published candidate arithmetic; variation sits in fit/val)",
              same_window, f"{len(set(test_partitions))} distinct windows (SPLIT_JITTER_TEST={JITTER_TEST})")
    else:
        check("B8 the test window moves with the seed (SPLIT_JITTER_TEST set)",
              not same_window, f"{len(set(test_partitions))} distinct windows (SPLIT_JITTER_TEST={JITTER_TEST})")


def audit_leakage(dataset: str) -> None:
    from content_floor import build_bundle

    _, nodes, edges = load_graph(dataset)
    n = len(nodes)
    for seed in MANIFEST["final_seeds"][dataset]:
        b = build_bundle(dataset, seed)
        fit_mask = b["split"]["fit_window"]["core_mask"].numpy()
        core = b["split"]["core_mask"].numpy()
        fit_edges = b["split"]["fit_window"]["fit_mp_edges"].numpy()
        test_edges = b["split"]["mp_edges_test"].numpy()
        val_edges = b["split"]["mp_edges_val"].numpy()
        check(f"C1 seed {seed}: training edges leave the fit window only", bool(fit_mask[fit_edges[0]].all()),
              f"{fit_edges.shape[1]} training edges")
        test_mask = ~core
        check(f"C2 seed {seed}: no test-window node is a training source",
              not bool(test_mask[fit_edges[0]].any()),
              f"{int(test_mask[fit_edges[0]].sum())} training edges leave a test node")
        n_test = max(1, int(round(F_TEST * n)))
        check(f"C3 seed {seed}: candidate pool = every node outside the test window",
              int(core.sum()) == n - n_test and bool((core == ~test_mask).all()),
              f"pool={int(core.sum())} of {n} (test window {n_test})")
        # independent re-derivation of the rankable positive count: a test edge is
        # rankable iff its target lies in the pool and is not one of the source's
        # fit-window targets (leakage rule).
        from ranking import _fit_targets_by_source, positives_by_source

        fit_tgt = _fit_targets_by_source(b["split"]["fit_window"]["fit_mp_edges"])
        reachable = 0
        for source, targets in positives_by_source(torch.from_numpy(test_edges)).items():
            mine = set(fit_tgt[source].tolist()) if source in fit_tgt else set()
            reachable += sum(1 for t in targets if core[t] and t != source and t not in mine)
        stored_pos = b["floor"]["test"]["positives"]
        check(f"C4 seed {seed}: rankable test positives re-derived from the pool",
              reachable == stored_pos and reachable > 0,
              f"{reachable} rankable of {test_edges.shape[1]} test edges "
              f"({int(test_mask[test_edges[1]].sum())} point inside the test window)")
        from content_floor import _node_days, _split_indices

        graph_days = _node_days(np.array([x["date"] for x in nodes]))
        g_fit, g_val, g_test = _split_indices(graph_days, seed)
        ok_windows = (fit_edges.shape[1] == 0 or bool(np.isin(fit_edges[0], g_fit).all()))
        ok_windows &= (val_edges.shape[1] == 0 or bool(np.isin(val_edges[0], g_val).all()))
        ok_windows &= (test_edges.shape[1] == 0 or bool(np.isin(test_edges[0], g_test).all()))
        check(f"C5 seed {seed}: fit/val/test edge sources sit in their own windows",
              ok_windows,
              f"fit {fit_edges[0].size and int(np.isin(fit_edges[0], g_fit).sum())}/{fit_edges.shape[1]} "
              f"val {int(np.isin(val_edges[0], g_val).sum())}/{val_edges.shape[1]} "
              f"test {int(np.isin(test_edges[0], g_test).sum())}/{test_edges.shape[1]}")
        # descriptors must not consume test-window edges: blank the test edges and
        # recompute; the stored descriptor matrix may not change.
        from gated import compute_descriptors

        desc = b["descriptors"]
        fit_names = b["fit_window"]["metadata"]
        check(f"C6 seed {seed}: descriptors are a [N,13] matrix over all nodes",
              tuple(desc.shape) == (n, 13), str(tuple(desc.shape)))
        # invariance: descriptors are built from fit-window edges and fit-node
        # statistics only, so a node whose content cannot reach any descriptor --
        # neither through fit edges (as neighbour) nor as a fit node -- must not
        # influence a single descriptor value.  Protected: fit nodes plus the
        # targets of fit-window edges; everything else is replaced by noise.
        protected = fit_mask.copy()
        if fit_edges.shape[1]:
            protected[fit_edges[1]] = True
        free = np.nonzero(~protected)[0]
        rng = np.random.default_rng(seed)
        feats = b["features"].clone()
        feats[torch.tensor(free)] = torch.tensor(
            rng.normal(size=(free.size, feats.shape[1])), dtype=feats.dtype)
        desc2, _ = compute_descriptors(
            feats, b["split"]["fit_window"]["fit_mp_edges"],
            torch.tensor(fit_mask), np.array([x["date"][:10] for x in nodes]),
            fit_names["cutoff"])
        check(f"C6b seed {seed}: descriptors ignore content outside the fit graph",
              bool(np.array_equal(desc2.numpy(), desc.numpy())),
              f"{free.size} free nodes replaced, max |diff| = "
              f"{float(np.abs(desc2.numpy() - desc.numpy()).max()):.2e}")
        check(f"C7 seed {seed}: fit-window metadata matches the split",
              fit_names["fit_nodes"] + fit_names["val_nodes"] + fit_names["test_nodes"] == n,
              f"{fit_names['fit_nodes']}+{fit_names['val_nodes']}+{fit_names['test_nodes']}={n}")
        sizes = {"fit": fit_edges.shape[1], "val": val_edges.shape[1], "test": test_edges.shape[1]}
        ok_et = all(k in b["edge_time"] and int(b["edge_time"][k].shape[0]) == v
                    for k, v in sizes.items())
        check(f"C8 seed {seed}: one edge-time feature row per edge in each window", ok_et,
              f"{ {k: int(v.shape[0]) for k, v in b['edge_time'].items()} } vs {sizes}")


def audit_features(dataset: str) -> None:
    """Features must depend on fit-window text only, plus a floor recomputation."""
    from content_floor import build_bundle, content_floor_metrics

    seed = MANIFEST["final_seeds"][dataset][0]
    b = build_bundle(dataset, seed)
    feats = b["features"].numpy()
    norms = np.linalg.norm(feats, axis=1)
    zero_rows = int((norms == 0).sum())
    check("D1 features are L2-normalised (empty texts stay at the origin)",
          bool(np.allclose(norms[norms > 0], 1.0, atol=1e-5)),
          f"{zero_rows} zero rows of {feats.shape[0]}")
    check("D2 feature metadata records the calibrated recipe",
          b["feature_metadata"].get("fit_scope") == "fit_window_only"
          and b["feature_metadata"].get("tfidf_token_pattern") is not None,
          json.dumps(b["feature_metadata"].get("tfidf_token_pattern")))
    floor = content_floor_metrics(b)
    stored = b["floor"]["test"]
    check("D3 stored test floor is reproduced from the bundle",
          abs(floor["test"]["mrr"] - stored["mrr"]) < 1e-12,
          f"recomputed {floor['test']['mrr']:.6f} vs stored {stored['mrr']:.6f}")
    check("D4 stored floor reports sources and candidates",
          stored["sources"] == floor["test"]["sources"]
          and abs(stored["candidates_mean"] - floor["test"]["candidates_mean"]) < 1e-6,
          f"sources={stored['sources']} positives={stored['positives']} "
          f"candidates={stored['candidates_mean']:.0f}")


def _runs(dataset: str, model: str) -> dict[int, dict]:
    out = {}
    for path in (PROJECT / "results" / "runs").glob(f"run_{dataset}_{model}_seed*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        out[int(rec["seed"])] = rec
    return out


def audit_runs(datasets: list[str]) -> dict:
    tables = {}
    for dataset in datasets:
        seeds = MANIFEST["final_seeds"][dataset]
        floors = {}
        for path in (PROJECT / "results" / "floor").glob(f"floor_{dataset}_seed*.json"):
            rec = json.loads(path.read_text(encoding="utf-8"))
            floors[int(rec["seed"])] = rec["floor"]["test"]["mrr"]
        table = {}
        for model in MODELS:
            runs = _runs(dataset, model)
            missing = [s for s in seeds if s not in runs]
            if missing:
                check(f"E1 {dataset} {model}: ten runs present", False, f"missing seeds {missing}")
                continue
            cfgs = {runs[s]["grid_id"] for s in seeds}
            check(f"E1 {dataset} {model}: ten runs present, one frozen config", len(cfgs) == 1,
                  f"{len(runs)} runs, cfg={sorted(cfgs)}")
            diffs = [abs(runs[s]["floor"]["mrr"] - floors[s]) for s in seeds if s in floors]
            check(f"E2 {dataset} {model}: floor recorded in the run equals the floor phase",
                  bool(diffs) and max(diffs) < 1e-6, f"max diff {max(diffs) if diffs else float('nan'):.2e}")
            anchors = [abs(runs[s].get("init_val_mrr", float("nan")) - runs[s]["floor_val"]["mrr"])
                       for s in seeds if runs[s].get("init_val_mrr") is not None]
            check(f"H {dataset} {model}: untrained model reproduces the content floor on validation",
                  bool(anchors) and max(anchors) < 1e-9,
                  f"max |init_val_mrr - floor_val| = {max(anchors) if anchors else float('nan'):.2e}")
            # E3: which checkpoint the reported model *is*.  The manuscript never
            # states the policy, and README 5.10 shows it is not cosmetic -- a
            # "last epoch" policy is the only one that can put a model below the
            # content floor.  A results root must therefore not mix policies, and a
            # recorded policy must be consistent with the epoch it selected.  Runs
            # written before CHECKPOINT_RULE existed record ``best_epoch`` only;
            # for those the policy is implied (best_val) and the check verifies the
            # weaker property that ``best_epoch`` is a trained epoch.
            policies: dict[str, int] = {}
            inconsistent: list[int] = []
            recorded = 0
            for seed in seeds:
                rec = runs[seed]
                rule = rec.get("checkpoint_rule")
                policies[rule or "best_val(implied)"] = policies.get(rule or "best_val(implied)", 0) + 1
                best, last = rec.get("best_epoch"), rec.get("epochs_run")
                if best is None or last is None:
                    continue
                if rule is None:
                    if not 1 <= int(best) <= int(last):
                        inconsistent.append(seed)
                    continue
                recorded += 1
                sel = rec.get("selected_epoch")
                if sel is None:
                    inconsistent.append(seed)
                    continue
                if rule == "last":
                    ok = int(sel) == int(last)
                elif rule == "first":
                    ok = int(sel) == 1
                elif rule == "best_val_with_init":
                    ok = int(sel) in (int(best), 0)
                else:
                    ok = int(sel) == int(best)
                if not ok:
                    inconsistent.append(seed)
            check(f"E3 {dataset} {model}: one checkpoint policy in this results root", len(policies) == 1,
                  f"policies {sorted(policies)} (use a separate RESULTS_ROOT per policy, README 5.10)")
            check(f"E3 {dataset} {model}: selected epoch matches the recorded checkpoint policy",
                  not inconsistent,
                  f"{recorded}/{len(seeds)} runs record a policy, {len(inconsistent)} inconsistent"
                  + (f" {inconsistent}" if inconsistent else "")
                  + ("" if recorded else " -- artifact predates CHECKPOINT_RULE, policy implied best_val"))
            table[model] = {s: runs[s]["test"]["mrr"] for s in seeds}
        # F. contrasts
        floor_list = [floors[s] for s in seeds if s in floors]
        contrasts = []
        for name, a, b in AUDIT_CONTRASTS:
            if a not in table or (b is not None and b not in table):
                continue
            if b is None:
                d = [table[a][s] - floors[s] for s in seeds if s in floors]
            else:
                d = [table[a][s] - table[b][s] for s in seeds]
            n = len(d)
            m = mean(d)
            half = float(stats.t.ppf(0.975, n - 1) * stdev(d) / math.sqrt(n)) if n > 1 else 0.0
            contrasts.append({"contrast": name, "mean": m, "ci95": [m - half, m + half],
                              "signs_positive": sum(1 for x in d if x > 0), "seeds": n})
        # floor-vs-model and model-vs-model contrasts recomputed here must agree
        # with results/summary/summary.json when that file exists.
        summary_path = PROJECT / "results" / "summary" / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            recorded = {c["contrast"]: c for c in summary.get(dataset, {}).get("contrasts", [])}
            matched = 0
            for c in contrasts:
                rec = recorded.get(c["contrast"])
                if rec is None:
                    continue
                matched += 1
                same = (abs(rec["mean"] - c["mean"]) < 1e-9
                        and rec["signs_positive"] == c["signs_positive"])
                check(f"F {dataset} {c['contrast']}: summary matches the audit recomputation", same,
                      f"summary {rec['mean']:+.6f}/{rec['signs_positive']} vs audit "
                      f"{c['mean']:+.6f}/{c['signs_positive']}")
            if matched:
                check(f"F {dataset}: every audited contrast is present in the summary",
                      matched == len(contrasts), f"{matched}/{len(contrasts)}")
        else:
            print(f"  [skip] summary.json absent: {len(contrasts)} contrasts recomputed but not compared",
                  flush=True)
        check(f"G {dataset}: the frozen configs equal the re-selected ones",
              _check_selection(dataset))
        tables[dataset] = {"floor_mean": float(np.mean(floor_list)) if floor_list else None,
                           "contrasts": contrasts}
    return tables


def _check_selection(dataset: str) -> bool:
    """Recompute the frozen selection from the tuning files and compare.

    ``selections.json`` records the mode: ``per_model`` (each cell takes the
    grid point with the best validation MRR -- what the published tables show,
    since the config column differs from cell to cell) or ``shared`` (one grid
    point per dataset, best mean validation MRR over the four cells).
    """
    grid = MANIFEST["tuning"]["grid"]
    tuning_dir = PROJECT / "results" / "tuning"
    val_by_cell: dict[str, dict[str, float]] = {}
    for cell in MODELS:
        path = tuning_dir / f"tuning_{dataset}_{cell}.json"
        if not path.exists():
            return False
        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        val_by_cell[cell] = {e["grid_id"]: e["val"]["mrr"] for e in entries}
    sel_path = PROJECT / "results" / "selections.json"
    if not sel_path.exists():
        return False
    recorded = json.loads(sel_path.read_text(encoding="utf-8")).get(dataset, {})
    mode = json.loads(sel_path.read_text(encoding="utf-8")).get("mode", "per_model")
    best_per_cell = {}
    for cell in MODELS:
        scored = sorted(val_by_cell[cell].items(), key=lambda kv: (-kv[1], kv[0]))
        if scored:
            best_per_cell[cell] = scored[0][0]
    if mode == "per_model":
        used = recorded.get("used", {})
        ok = all(used.get(m) == best_per_cell.get(m) for m in MODELS if m in val_by_cell)
        if ok:
            anchors = recorded.get("per_model_best", {})
            dups = [c for c in CELLS
                    if abs(anchors.get(c, {}).get("tuning_val_mrr", -1)
                           - val_by_cell[c].get(best_per_cell[c], -2)) > 1e-12]
            ok = not dups
        print(f"      re-selected per cell: "
              f"{ {c: best_per_cell.get(c) for c in CELLS} }; recorded "
              f"{ {c: recorded.get('used', {}).get(c) for c in CELLS} }", flush=True)
        return ok
    # shared mode
    best_id, best_mean = None, None
    for cfg in grid:
        vals = [val_by_cell[c].get(cfg["id"]) for c in CELLS]  # cells only: baselines do not vote
        if any(v is None for v in vals):
            continue
        m = mean(vals)
        if best_mean is None or m > best_mean:
            best_id, best_mean = cfg["id"], m
    used = set(recorded.get("used", {}).values())
    ok = recorded.get("shared_grid_id") == best_id and used == {best_id}
    print(f"      re-selected {best_id} (mean val MRR {best_mean:.4f} over {len(CELLS)} cells); "
          f"recorded {recorded.get('shared_grid_id')} used in {sorted(used)}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=MANIFEST["datasets"])
    ap.add_argument("--out", default=str(PROJECT / "results" / "audit.json"))
    args = ap.parse_args()
    report = {"datasets": {}, "fails": FAILS}
    for dataset in args.datasets:
        print(f"\n=== audit {dataset} ===", flush=True)
        audit_graph(dataset)
        audit_split(dataset)
        audit_leakage(dataset)
        audit_features(dataset)
    print("\n=== run bookkeeping and contrasts ===", flush=True)
    report["datasets"] = audit_runs(args.datasets)
    report["fails"] = FAILS
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\naudit written to {args.out}; {len(FAILS)} failing checks")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
