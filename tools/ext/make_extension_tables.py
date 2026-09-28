"""Emit the LaTeX tables for the extension experiments (E1-E8).

Every number here is read from an artifact produced by the extension scripts in
``tools/ext`` (``results_ext/*.json``), so the tables cannot drift from the runs:
they are regenerated, not transcribed.

    table_structural_baselines.tex   E2 heuristics + E8 learned pairwise ranker
    table_text_floor.tex             E3a stronger text pipelines + E3b no-graph
    table_stratified.tex             E3c floor and increment by description length
    table_gate_diagnostics.tex       E4 gate values, gradients, init ablation
    table_persource.tex              E5 source-level inference
    table_variance.tex               E1 variance decomposition
    table_maven_dates.tex            E6 Maven under the first-publication date
    table_citation_control.tex       E7 rebuilt HepTh / HepPh control
    table_pairwise.tex               E8 learned pairwise structural ranker

Usage:
    python3 tools/ext/make_extension_tables.py --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "results_ext"

HEUR_LABEL = {
    "degree": "degree (popularity)",
    "pa": "preferential attachment",
    "cn": "common neighbours",
    "ncn": "normalised common neighbours",
    "aa": "Adamic--Adar",
    "ra": "resource allocation",
    "ppr": "personalised PageRank",
    "spectral": "spectral embedding (rank 128)",
    "random": "random score (sanity check)",
}
HEUR_ORDER = ["degree", "pa", "cn", "ncn", "aa", "ra", "ppr", "spectral"]
CELL_LABEL = {"gat_dir": r"gat\_dir", "ragat_sym": r"rcgat\_sym",
              "gat_time": r"gat\_time", "ragat_time": r"rcgat\_time"}
ROW = r" \\"


def _stack(num: str, ci: str) -> str:
    """A two-line table cell: the estimate over its interval.

    The extension tables carry a point estimate *and* a 95% interval in the same
    cell, and the interval alone is wider than the value it qualifies.  Typeset
    side by side the natural width of three of these tables ran 36-109 pt past
    the 370.7 pt text width; stacked, they fit with room to spare and no font is
    shrunk.  ``\shortstack[c]`` centres the two lines, which reads better than
    the kernel default (right).
    """
    return r"\shortstack[c]{" + num + r"\\" + ci + "}"


def _num(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def _ci(lo: float, hi: float, digits: int = 4) -> str:
    return f"$[{lo:+.{digits}f},{hi:+.{digits}f}]$"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def structural_baselines() -> str:
    e2 = _load(EXT / "e2_structural_summary.json")
    e8_cells = {Path(p).name: _load(Path(p))
                for p in sorted(glob.glob(str(EXT / "e8_pairwise" / "cell_*.json")))}
    if e2 is None:
        return ""
    lines = [
        r"\begin{table}[t]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Structure-only rankers under the frozen protocol (per-source MRR over the",
        r"test candidate pool; ten seeds for the training-free heuristics, five for the learned",
        r"pairwise scorer). ``observed'' is the honest inductive graph (fit and validation edges,",
        r"everything strictly before the test window), which gives these baselines strictly less",
        r"information than the message-passing networks have; ``as scored'' uses the networks' own",
        r"test-time message-passing graph (\texttt{mp\_edges\_test}), which contains the edges being",
        r"predicted, so the value there describes the protocol rather than the ranker. The content",
        r"floor is the same train-only TF-IDF cosine ranker as everywhere else in the paper.}",
        r"\label{tab:structural}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{npm} & \multicolumn{2}{c}{Maven} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Ranker & observed & as scored & observed & as scored \\",
        r"\midrule",
    ]

    def row(label: str, npm_o: float, npm_t: float, mav_o: float, mav_t: float) -> str:
        return (f"{label} & {_num(npm_o)} & {_num(npm_t)} & {_num(mav_o)} & {_num(mav_t)}"
                + ROW)

    lines.append(row("content floor", e2["npm"]["floor"]["mean"], e2["npm"]["floor"]["mean"],
                     e2["maven"]["floor"]["mean"], e2["maven"]["floor"]["mean"]))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{@{}l}{\emph{training-free heuristics}} \\")
    for h in HEUR_ORDER:
        n = e2["npm"]["heuristics"].get(h)
        m = e2["maven"]["heuristics"].get(h)
        if not n or not m:
            continue
        lines.append(row(HEUR_LABEL.get(h, h), n["obs_mean"], n["test_mean"],
                         m["obs_mean"], m["test_mean"]))
    per: dict = {"npm": {}, "maven": {}}
    for cell in e8_cells.values():
        for variant, gd in cell["variants"].items():
            per[cell["dataset"]].setdefault(variant, {"obs": [], "test": []})
            per[cell["dataset"]][variant]["obs"].append(gd["obs"]["mrr"])
            per[cell["dataset"]][variant]["test"].append(gd["test"]["mrr"])
    if per["npm"] and per["maven"]:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{@{}l}{\emph{learned pairwise scorer (NCN-style: MLP on "
                     r"1--3-hop neighbourhood features)}} \\")
        for variant, label in (("onehop", "one-hop features"), ("full", "multi-hop features")):
            if variant not in per["npm"] or variant not in per["maven"]:
                continue
            f = lambda d, v, k: sum(per[d][v][k]) / len(per[d][v][k])
            lines.append(row(label, f("npm", variant, "obs"), f("npm", variant, "test"),
                             f("maven", variant, "obs"), f("maven", variant, "test")))
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def text_floor() -> str:
    cells: dict = {}
    for path in sorted(glob.glob(str(EXT / "e3_text_floor" / "cell_*.json"))):
        d = _load(Path(path))
        cells.setdefault((d["dataset"], d["recipe"]), []).append(d)
    report = _load(EXT / "e4_instrument_report.json") or {}
    if not cells:
        return ""
    recipes = ["baseline", "char35", "union", "big_word"]
    label = {"baseline": "frozen recipe (word TF-IDF, 20k)",
             "char35": "character 3--5 grams",
             "union": "word $+$ character union",
             "big_word": "word TF-IDF, 60k, no stopword list"}
    lines = [
        r"\begin{table}[t]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Is the content floor weak because of its text representation? The same cosine",
        r"ranker, splits, candidate pools and metric as everywhere else in the paper, with the text",
        r"pipeline replaced (five seeds per graph). Each dataset cell carries the test MRR with",
        r"its seed standard deviation on the first line, and on the second the mean of the five",
        r"per-seed paired differences against the frozen floor with its 95\% $t$ interval; the frozen",
        r"recipe recomputes that floor and reproduces it seed for seed. The last two rows go the other way",
        r"and remove the graph from a trained cell: an empty edge set and zero descriptors, i.e.\ a",
        r"learned scorer on the text features alone.}",
        r"\label{tab:textfloor}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcc@{}}",
        r"\toprule",
        r"Text pipeline & npm & Maven \\",
        r"\midrule",
    ]

    def entry(dataset: str, recipe: str):
        """(mean, sd, paired delta vs the frozen floor, 95% CI of that delta)."""
        items = sorted(cells.get((dataset, recipe)) or [], key=lambda i: i["seed"])
        if not items:
            return None
        vals = np.array([i["test"]["mrr"] for i in items])
        frozen = np.array([i["frozen_floor_test"] for i in items])
        d = vals - frozen
        n = d.size
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
        return (float(vals.mean()), float(vals.std(ddof=1)) if n > 1 else 0.0,
                float(d.mean()), (float(d.mean() - crit * se), float(d.mean() + crit * se)))

    def cell(d) -> str:
        if not d:
            return "--"
        lo, hi = d[3]
        return _stack(f"${_num(d[0])} \\pm {_num(d[1])}$",
                      "{\\scriptsize" + f"$[{lo:+.4f},{hi:+.4f}]$" + "}")

    for recipe in recipes:
        n, m = entry("npm", recipe), entry("maven", recipe)
        if not n and not m:
            continue
        lines.append(f"{label[recipe]} & {cell(n)} & {cell(m)}" + ROW)

    lines.append(r"\midrule")
    for model in ("gat_dir", "ragat_sym"):
        vals = {}
        for dataset in ("npm", "maven"):
            d = report.get(f"{dataset}|{model}|zero|1")
            if d:
                c = d["paired_vs_floor"]
                vals[dataset] = (d["test_mean"], d.get("test_sd", 0.0), c["mean"],
                                 (c["ci95"][0], c["ci95"][1]))
            else:
                vals[dataset] = None
        lines.append(f"no graph: {CELL_LABEL[model]} & {cell(vals['npm'])} & "
                     f"{cell(vals['maven'])}" + ROW)

    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def stratified() -> str:
    """Two floats: the increment by description length, and by namespace."""
    d = _load(EXT / "e3c_stratified" / "summary.json")
    if not d:
        return ""
    blocks = sorted(d.items(), key=lambda kv: (kv[0].split("|")[0], kv[0].split("|")[1]))

    def header(caption_lines, label, title):
        return ([r"\begin{table}[t]", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}"]
                + caption_lines + [f"\\label{{{label}}}",
                                   r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccc@{}}",
                                   r"\toprule",
                                   f"Stratum & Sources & Floor MRR & $\\Delta$ vs floor & 95\\% CI \\\\",
                                   r"\midrule"])

    def block_and_rows(entry, key, rows):
        lines = [r"\multicolumn{5}{@{}l}{\emph{" + rows + r"}} \\", r"\midrule"]
        for i, label in enumerate(sorted(entry[key])):
            st = entry[key][label]
            name = label
            if key == "length_quartiles":
                name = f"Q{i + 1}" + (" (shortest)" if i == 0 else " (longest)" if i == 3 else "")
            lines.append(f"\\quad {name} & {st['n']} & {_num(st['floor_mean'])} & "
                         f"${st['mean']:+.4f}$ & {_ci(*st['ci95'])}" + ROW)
        lines.append(r"\midrule")
        return lines

    def render(key, caption_lines, label, heading):
        out = header(caption_lines, label, heading)
        for k, entry in blocks:
            dataset, model = k.split("|")
            if key not in entry:
                continue
            out += block_and_rows(entry, key, f"{dataset}, {CELL_LABEL.get(model, model)}, "
                                  f"{entry['sources_mean']:.0f} rankable sources per seed")
        out = out[:-1]
        out += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
        return out

    length_cap = [
        r"\caption{The content floor and the structural increment, stratified by the length of",
        r"the source's own description. Every rankable source of every seed is assigned to a",
        r"quartile of the candidate pool that seed ranks, so no stratum is empty by",
        r"construction, and each stratum reports the content floor inside it and the mean",
        r"per-source paired difference of the model against that floor. The content-floor",
        r"hypothesis predicts that the strata with the weakest floor carry the structural",
        r"increment; on npm the floor is weakest instead on the longest descriptions, and the",
        r"largest increment sits in the shortest stratum. The same decomposition by number of",
        r"positive targets is in the released report.}"]
    group_cap = [
        r"\caption{The same stratification by namespace: the npm scope, or the Maven",
        r"groupId, with the long tail pooled. Vendor scopes on npm (@devframes, @csstools,",
        r"@jsonjoy.com) have near-random floors, and the largest increments sit in the broad",
        r"``other'' bucket rather than in the scopes whose floor is weakest, which is the",
        r"opposite of what the content-floor hypothesis predicts. The Maven strata hold tens",
        r"of sources each and their intervals are correspondingly wide.}"]
    return "\n".join(render("length_quartiles", length_cap, "tab:stratified", "Bylength") 
                     + render("groups", group_cap, "tab:stratified_groups", "bygroup"))


def gate_diagnostics() -> str:
    report = _load(EXT / "e4_instrument_report.json")
    if not report:
        return ""
    rows = []
    for dataset in ("npm", "maven"):
        for cond in ("zero", "rand"):
            d = report.get(f"{dataset}|ragat_sym|{cond}|0")
            if d:
                rows.append((dataset, cond, d))
    if not rows:
        return ""
    lines = [
        r"\begin{table}[t]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{What the gate learned. The gate multiplies the residual branch by",
        r"$1+\tanh(a)$, with $a$ linear in the descriptor and the weights initialised at zero, so",
        r"$g\equiv1$ and the gate gradient is exactly zero at step zero. The first row per graph is",
        r"that zero-initialised cell, the second the same cell with the gate weights drawn from",
        r"$\mathcal{N}(0,0.05^{2})$ before training (five seeds each, configuration fixed to the",
        r"frozen selection). ``range'' spans all four gate modules and all nodes at the selected",
        r"checkpoint, ``moved'' is the share of node-gates with $|g-1|>0.01$, ``$|a|$ p95'' the 95th",
        r"percentile of the pre-activation, and the gradient columns are the maximum of the median",
        r"per-epoch norms (gate, and encoder for scale).}",
        r"\label{tab:gatediag}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc@{}}",
        r"\toprule",
        r"Graph & Cell & Range of $g$ & Moved & $|a|$ p95 & gate grad & encoder grad & test MRR \\",
        r"\midrule",
    ]
    for dataset, cond, d in rows:
        g = d["gates"]
        lo = min(x["g_min"] for x in g.values())
        hi = max(x["g_max"] for x in g.values())
        moved = sum(x["moved_share"] for x in g.values()) / len(g)
        p95 = max(x["pre_abs_p95"] for x in g.values())
        gmax = (d["gradients"].get("gate") or {}).get("max_median")
        elo = (d["gradients"].get("encoder") or {}).get("last_epoch_median")
        lines.append(f"{dataset} & {'zero-init' if cond == 'zero' else 'random-init'} & "
                     f"$[{lo:.2f},{hi:.2f}]$ & {100 * moved:.0f}\\% & {p95:.3f} & "
                     f"{('%.3f' % gmax) if gmax is not None else '--'} & "
                     f"{('%.2f' % elo) if elo is not None else '--'} & {_num(d['test_mean'])}" + ROW)
    lines.append(r"\midrule")
    for name, label in (("rand_vs_zero", r"rand$-$zero init"),
                        ("textonly_vs_zero", r"no graph$-$full")):
        for key, st in sorted(report.get("contrasts", {}).get(name, {}).items()):
            dataset, model = key.split("|")[0], key.split("|")[1]
            degenerate = (name == "rand_vs_zero" and abs(st["mean"]) < 1e-9
                          and abs(st["ci95"][0]) < 1e-9 and abs(st["ci95"][1]) < 1e-9)
            body = ("no gate in this cell: the two conditions are one run"
                    if degenerate else
                    f"${st['mean']:+.4f}$ {_ci(*st['ci95'])}, "
                    f"{st['positive_seeds']}/{st['n']} seeds")
            lines.append(f"{dataset} & {label} ({CELL_LABEL.get(model, model)}) & "
                         r"\multicolumn{5}{@{}l}{" + body + r"} & --" + ROW)
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def persource() -> str:
    d = _load(EXT / "e5_persource" / "e5_persource.json")
    if not d:
        return ""
    label = {"gat_dir": r"gat\_dir $-$ floor", "ragat_sym": r"rcgat\_sym $-$ floor",
             "gate_contrast_per_source": r"gate: rcgat\_sym $-$ gat\_dir"}
    lines = [
        r"\begin{table}[t]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Source-level inference: the unit of replication is the source, not the seed.",
        r"Each rankable source contributes one paired difference against the content floor,",
        r"averaged over the five seeds in which it is rankable; ``bootstrap'' is a source-level",
        r"bootstrap 95\% interval (10{,}000 resamples), ``improved'' the share of sources with a",
        r"positive difference, and ``top 1\%'' the share of the total positive mass carried by the",
        r"1\% best sources, which bounds how much of the effect hangs on a handful of rows. The",
        r"sign test is distribution-free and the gate row is the per-source gate contrast that the",
        r"seed-level analysis reports as null.}",
        r"\label{tab:persource}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc@{}}",
        r"\toprule",
        r"Graph & Contrast & Sources & Mean $\Delta$ & 95\% bootstrap CI & Improved & Sign $p$ & Top 1\% \\",
        r"\midrule",
    ]
    for dataset in ("npm", "maven"):
        for key in ("gat_dir", "ragat_sym", "gate_contrast_per_source"):
            v = d.get(dataset, {}).get(key)
            if not v or "mean_diff" not in v:
                continue
            lines.append(f"{dataset} & {label[key]} & {v['n_sources']} & ${v['mean_diff']:+.4f}$ & "
                         f"{_ci(*v['ci95_boot'])} & {100 * v['share_positive']:.0f}\\% & "
                         f"{v.get('p_sign_test', float('nan')):.3f} & "
                         f"{100 * v['concentration_positive_mass_top1pct']:.0f}\\%" + ROW)
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def variance() -> str:
    d = _load(EXT / "e1_variance" / "e1_variance.json")
    if not d:
        return ""
    lines = [
        r"\begin{table}[t]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Where the variance of the results lives: seven cells $\times$ ten seeds",
        r"$\times$ two graphs, analysed as a two-way design (graph, cell) with the split nested",
        r"inside the graph. The entries are shares of the total sum of squares, so they say which",
        r"factor a reader should treat as the replication unit; the last two rows report how often",
        r"a single seed's own best cell is the pooled best, with the mean Spearman correlation",
        r"between a seed's ranking of the cells and the pooled ranking.}",
        r"\label{tab:variance}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}",
        r"\toprule",
        r"Quantity & Graph & Cell & Interaction & Split (seed) \\",
        r"\midrule",
    ]
    for name, key in ((r"test MRR", "test_mrr"), (r"$\Delta$ vs floor", "delta_vs_floor")):
        share = d[key]["anova"]["ss_share"]
        lines.append(f"{name} & {100 * share['dataset']:.1f}\\% & {100 * share['model']:.1f}\\% & "
                     f"{100 * share['interaction']:.1f}\\% & {100 * share['split']:.1f}\\%" + ROW)
    lines += [r"\midrule",
              r"\multicolumn{5}{@{}l}{\emph{rank stability (test MRR): share of seeds whose own best cell is the pooled best}} \\",
              r"\midrule"]
    for dataset in ("npm", "maven"):
        st = d["test_mrr"].get(f"rank_stability_{dataset}")
        if not st:
            continue
        # the pooled best cell is a model name such as gcn_dir: its underscore is a
        # text-mode subscript and aborts the run with "Missing $ inserted" if it is
        # not escaped, so escape it here rather than in the .tex by hand.
        best = str(st["pooled_best"]).replace("_", r"\_")
        lines.append(f"{dataset} & \\multicolumn{{4}}{{@{{}}l}}{{"
                     f"{100 * st['share_seed_best_agrees']:.0f}\\% agree ({best}); "
                     f"mean Spearman $\\rho = {st['mean_spearman_seed_vs_pooled']:.2f}$, "
                     f"minimum ${st['min_spearman_seed_vs_pooled']:.2f}$" + r"}" + ROW)
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def maven_dates() -> str:
    d = _load(EXT / "e6_maven_firstpub_report.json")
    if not d:
        return ""
    order = [k for k in ("maven", "maven_firstpub", "maven_firstpub_full") if k in d["graphs"]]
    name = {"maven": "latest release (frozen)",
            "maven_firstpub": "first publication, same snapshot",
            "maven_firstpub_full": "first publication, no window filter"}
    lines = [
        r"\begin{table}[t]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{The Maven node-date convention, rebuilt three ways. The frozen graph dates each",
        r"node by its latest non-prerelease release; the second variant rebuilds the graph from the",
        r"same snapshot with the first publication date, and the third drops the build-time window",
        r"filter as well. The date profile decides how much of the graph falls inside the test",
        r"window, and the contrasts below show that the branch-level verdicts move with it.}",
        r"\label{tab:mavendates}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}",
        r"\toprule",
        r"Date convention & Nodes & Edges & Within 3 months & Within 12 months \\",
        r"\midrule",
    ]
    for key in order:
        g = d["graphs"][key]
        lines.append(f"{name.get(key, key)} & {g['nodes']} & {g['edges']} & "
                     f"{100 * g['share_within_3_months']:.1f}\\% & "
                     f"{100 * g['share_within_12_months']:.1f}\\%" + ROW)
    lines += [r"\bottomrule", r"\end{tabular*}", "",
              r"\vspace{4pt}", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}",
              r"\toprule",
              r"Date convention & Floor & gat\_dir & rcgat\_sym & gate $\Delta$ & time $\Delta$ \\",
              r"\midrule"]
    for key in order:
        e = d["datasets"].get(key)
        if not e:
            continue
        models = e["models"]
        cell = lambda m: f"{_num(models[m]['test_mean'], 4)}" if m in models else "--"
        gd = e.get("gate", {})
        td = e.get("time", {})
        def cstr(c):
            if not c:
                return "--"
            return _stack(f"${c['mean']:+.4f}$", _ci(*c['ci95']))
        lines.append(f"{name.get(key, key)} & {_num(e['floor']['mean'])} & {cell('gat_dir')} & "
                     f"{cell('ragat_sym')} & {cstr(gd)} & {cstr(td)}" + ROW)
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def citation() -> str:
    d = _load(EXT / "e7_citation_control_report.json")
    if not d:
        return ""
    sources = [s for s in ("hepth", "hepph") if s in d["sources"]]
    if not sources:
        return ""
    label = {"hepth": "HepTh", "hepph": "HepPh"}
    lines = [
        r"\begin{table}[t]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Strong-content control, rebuilt from the primary sources rather than restated:",
        r"the SNAP directed citation lists (\texttt{cit-HepTh}, \texttt{cit-HepPh}) restricted to the",
        r"papers of the arXiv submission years 2002--2003, with node text (title and abstract) and",
        r"node dates taken from the arXiv API. Five seeds per graph, four cells, the same protocol",
        r"as every other table in this paper; the levels are therefore those of the rebuilt subgraph",
        r"and are not comparable with the full-graph numbers of the earlier round.}",
        r"\label{tab:citation}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}",
        r"\toprule",
        r"Graph & Floor & gat\_dir & rcgat\_sym (gate) & gat\_time & rcgat\_time \\",
        r"\midrule",
    ]
    for s in sources:
        e = d["sources"][s]
        cells = [e["models"].get(m) for m in ("gat_dir", "ragat_sym", "gat_time", "ragat_time")]
        body = " & ".join(f"{_num(c['test_mean'], 4)} $\\pm$ {_num(c['test_sd'], 4)}" if c else "--"
                          for c in cells)
        lines.append(f"{label.get(s, s)} & {_num(e['floor']['mean'], 4)} & {body}" + ROW)
    lines += [r"\bottomrule", r"\end{tabular*}", "", r"\vspace{4pt}",
              r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}",
              r"\toprule",
              r"Graph & Gate contrast & & Time contrast & \\",
              r"\midrule"]

    def cstr(c):
        return f"${c['mean']:+.4f}$ & {_ci(*c['ci95'])}" if c else "-- & --"

    for s in sources:
        e = d["sources"][s]
        lines.append(f"{label.get(s, s)} & {cstr(e.get('gate'))} & {cstr(e.get('time'))}" + ROW)
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def e8_pairwise() -> str:
    cells: dict = {}
    for path in sorted(glob.glob(str(EXT / "e8_pairwise" / "cell_*.json"))):
        d = _load(Path(path))
        if d:
            cells.setdefault(d["dataset"], []).append(d)
    if not cells:
        return ""
    label = {"onehop": "1-hop features (3)", "full": "full features (8)"}
    lines = [
        r"\begin{table}[t]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{A learned pairwise structural ranker (the NCN recipe, MLP over the pairwise",
        r"features of the two nodes) trained with a Bayesian-personalised-ranking loss, scored on",
        r"each seed's own candidate pool and compared with that seed's frozen content floor.",
        r"\emph{observed} uses the edges that exist in the released graph; \emph{test} uses the",
        r"message-passing edge set the GNNs are trained on, in which the test positives are",
        r"themselves edges, so a ranker that can read the adjacency returns the positive first by",
        r"construction. The last two rows are the GNN cell with its graph, for scale.}",
        r"\label{tab:pairwise}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccc@{}}",
        r"\toprule",
        r"Graph & Features & MRR (observed) & $\Delta$ vs floor & MRR (test graph) \\",
        r"\midrule",
    ]
    for dataset in ("npm", "maven"):
        items = cells.get(dataset)
        if not items:
            continue
        for variant in ("onehop", "full"):
            sel = [i for i in items if variant in i["variants"]]
            if not sel:
                continue
            obs = np.array([i["variants"][variant]["obs"]["mrr"] for i in sel])
            tst = np.array([i["variants"][variant]["test"]["mrr"] for i in sel])
            flo = np.array([i["floor"]["mrr"] for i in sel])
            d = obs - flo
            n = d.size
            se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
            lines.append(f"{dataset} & {label[variant]} & {_num(obs.mean())} $\\pm$ {_num(obs.std(ddof=1)) if n > 1 else '--'} & "
                         f"${d.mean():+.4f}$ " + "{\\scriptsize" + f"$[{d.mean() - crit * se:+.4f},{d.mean() + crit * se:+.4f}]$" + "} & "
                         f"{_num(tst.mean())}" + ROW)
    rep = _load(EXT / "e4_instrument_report.json") or {}
    for dataset in ("npm", "maven"):
        d = rep.get(f"{dataset}|gat_dir|zero|0")
        if d:
            lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{dataset}: {CELL_LABEL['gat_dir']} with its graph "
                         f"(for scale), {_num(d['test_mean'])} $\\pm$ {_num(d.get('test_sd', 0.0))} over 5 seeds}}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "table_structural_baselines.tex": structural_baselines(),
        "table_text_floor.tex": text_floor(),
        "table_stratified.tex": stratified(),
        "table_gate_diagnostics.tex": gate_diagnostics(),
        "table_persource.tex": persource(),
        "table_variance.tex": variance(),
        "table_maven_dates.tex": maven_dates(),
        "table_citation_control.tex": citation(),
        "table_pairwise.tex": e8_pairwise(),
    }
    for name, text in files.items():
        if not text:
            print(f"skip {name} (input missing)")
            continue
        (out / name).write_text(text, encoding="utf-8")
        print(f"wrote {out / name} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
