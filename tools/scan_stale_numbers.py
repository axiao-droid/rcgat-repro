"""Scan a manuscript tree for numbers and claims that the reconstruction retired.

Every entry below is a value that the original manuscript states and that the
rebuilt pipeline replaces with a different value (see
artifacts/rcgat-revision-package/CHANGELOG-numbers.md, sections 4-5).  Running
this after `apply.sh` shows exactly which files still carry the old numbers --
typically 05_methodology.tex and anything the revision package does not own.

    python tools/scan_stale_numbers.py /path/to/manuscript
    python tools/scan_stale_numbers.py /tmp/mscheck --formats tex,txt
    python tools/scan_stale_numbers.py /path/to/manuscript --quiet   # counts only

Exit status is 1 when hits are found (so it can gate a release), 0 otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (regex, what the old value was, what it is now).  The regex matches the
# literal as it appears in LaTeX: "4{,}962", "4,962" and "4962" are all caught
# by making the thousands separator optional and the braces optional.
STALE: list[tuple[str, str, str]] = [
    # ---- dataset scale -----------------------------------------------------
    (r"4\{?,\}?962\b", "npm node count", "4,959 records (13,602 edges)"),
    (r"1\{?,\}?484\b", "Maven node count", "1,664 records (5,651 edges)"),
    (r"13\{?,\}?596\b", "npm edge count", "13,602"),
    (r"5\{?,\}?482\b", "Maven edge count", "5,651"),
    (r"\b234\b\s*\(\s*686\s*\)", "npm test sources (positives)",
     "181 (494) per seeded window"),
    (r"\b686\b", "npm positive edges", "494 per seeded window"),
    (r"4\{?,\}?614\b", "npm mean candidates per source", "4,611 under reading A"),
    (r"1\{?,\}?376\b", "Maven mean candidates per source", "1,547 under reading A"),
    (r"4\{?,\}?895\b", "npm nodes with at least one edge", "3,081 with outgoing edges, "
     "1,814 target-only, 64 isolated"),
    # ---- content floors ----------------------------------------------------
    (r"0\.1565\b", "npm content floor", "0.1566"),
    (r"0\.3377\b", "Maven content floor (earlier snapshot)", "0.0922 here; 0.338 belongs "
     "to the earlier snapshot and must be quoted as such"),
    (r"0\.338\b", "Maven floor of the EARLIER snapshot quoted without attribution",
     "0.092 in this snapshot; 0.338 only when attributed"),
    # ---- the gate contrasts that flipped -----------------------------------
    (r"0\.0297\b", "gate contrast, npm (positive)", "null: -0.0021 [-0.0060,+0.0019], 5/10"),
    (r"0\.0311\b", "gate contrast, Maven (positive)", "null: -0.0014 [-0.0121,+0.0092], 5/10"),
    (r"0\.0351\b", "npm ungated baseline below the floor", "+0.0078 [+0.0040,+0.0115], 10/10 "
     "above the floor"),
    (r"0\.0496\b", "gate CI upper bound, npm (old)", "[+0.0040,+0.0019]"),
    (r"0\.0528\b", "gate CI upper bound, Maven (old)", "[+0.0092]"),
    (r"0\.0304\b", "gate floor increment, Maven (structural increment)", "withdrawn"),
    (r"0\.0054\b", "npm gate floor deficit after protection", "+0.0057 above the floor"),
    # ---- the (old) best model ----------------------------------------------
    (r"0\.3681\b", "Maven best MRR (the gated cell)", "0.1625 (gcn_dir)"),
    (r"0\.3501\b", "Maven GCN MRR", "0.1625"),
    (r"0\.3563\b", "Maven GATv2 MRR", "0.1052"),
    (r"0\.3600\b", "Maven GraphSAGE MRR", "0.1463"),
    (r"0\.1511\b", "npm gated cell MRR", "0.1623"),
    (r"0\.1587\b", "npm GraphSAGE MRR", "0.1696"),
    # ---- time branch -------------------------------------------------------
    (r"0\.0393\b", "npm time contrast (10/10)", "+0.0167 [+0.0040,+0.0294], 8/10, "
     "one seed short of the rule"),
    # ---- old baseline floor contrasts (all below/at the floor) -------------
    (r"-?0\.0178\b", "npm GCN vs floor", "+0.0553 [+0.0401,+0.0704]"),
    (r"-?0\.0171\b", "npm GATv2 vs floor", "+0.0068 [+0.0035,+0.0101]"),
    (r"0\.0022\b", "npm GraphSAGE vs floor", "+0.0130 [+0.0109,+0.0151]"),
    (r"0\.0124\b", "Maven GCN vs floor", "+0.0703 [+0.0319,+0.1086]"),
    (r"0\.0223\b", "Maven GraphSAGE vs floor", "+0.0541 [+0.0220,+0.0862]"),
    # ---- old baseline-vs-baseline contrasts --------------------------------
    (r"0\.0373\b", "npm GraphSAGE vs directed GAT", "+0.0052 [+0.0013,+0.0091], 8/10"),
    (r"0\.0173\b", "npm GCN vs directed GAT", "+0.0475 [+0.0344,+0.0605], 10/10"),
    (r"0\.0180\b", "npm GATv2 vs directed GAT", "-0.0010 [-0.0064,+0.0044]"),
    (r"0\.0193\b", "Maven GATv2 vs directed GAT", "+0.0014 [-0.0091,+0.0119]"),
    # ---- claims, not numbers ----------------------------------------------
    (r"(keep|keeps) the citation networks? as a strong-content control|"
     r"with two citation graphs as a high-content control|"
     r"fails? to replicate on strong-content citation networks|"
     r"the citation networks? (reject|rejects) the gate",
     "the citation graphs as an evaluated control",
     "background only: not re-run (S-32)"),
    (r"\\ref\{tab:citation\}|\\input\{tables/table_citation_control",
     "reference to the retired citation table", "remove; the table is no longer input"),
    (r"3\.3 times fewer parameters", "parameter claim attached to being the best model",
     "keep the ratio (409,601 / 122,937) but drop the 'best model' reading"),
    (r"seed 0\b", "tuning seed in table_configs", "seed 11"),
]

COMPILED = [(re.compile(rx), old, new) for rx, old, new in STALE]


def _ctx(line: str, width: int = 110) -> str:
    return line.strip()[:width]
# Files that legitimately contain the retired values (they are the package's own
# before/after documentation, not the manuscript).
SKIP_NAMES = {"CHANGELOG-numbers.md"}


def scan(root: Path, suffixes: tuple[str, ...], quiet: bool) -> int:
    hits = 0
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix in suffixes
                   and p.name not in SKIP_NAMES and ".bak" not in p.suffixes)
    for path in files:
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        found = []
        for rx, old, new in COMPILED:
            for i, line in enumerate(lines, 1):
                stripped = line.lstrip()
                # comment lines are the revision notes themselves, not manuscript text
                if stripped.startswith("%"):
                    continue
                if rx.search(line):
                    if "0.338" in line and re.search(r"earlier|snapshot|that round", line):
                        continue      # properly attributed to the earlier snapshot
                    found.append((i, rx.pattern, old, new, _ctx(line)))
        if not found:
            continue
        rel = path.relative_to(root)
        print(f"\n{rel}: {len(found)} hit(s)")
        if not quiet:
            for i, _, old, new, ctx in sorted(found):
                print(f"  line {i:>4}  [{old}]")
                print(f"              now: {new}")
                print(f"              {ctx}")
        hits += len(found)
    print(f"\nscanned {len(files)} file(s); {hits} stale hit(s)")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="manuscript directory to scan")
    ap.add_argument("--formats", default="tex,txt",
                    help="comma-separated extensions (default: tex,txt)")
    ap.add_argument("--quiet", action="store_true",
                    help="print counts per file only")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    suffixes = tuple("." + s.strip().lstrip(".") for s in args.formats.split(",") if s.strip())
    return 1 if scan(root, suffixes, args.quiet) else 0


if __name__ == "__main__":
    sys.exit(main())
