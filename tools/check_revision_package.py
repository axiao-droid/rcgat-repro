"""Structural checks for the route-A revision package.

Usage:
    python tools/check_revision_package.py [--install-dir /tmp/ms]

It (1) installs the package into a throw-away copy of the manuscript using
apply.sh, then (2) on the installed tree checks

  * every \\input{...} resolves to a file (with or without the .tex suffix),
  * every \\ref/\\eqref/\\Cref has a matching \\label,
  * no \\label is defined twice,
  * $ and $$ delimiters are balanced and {} are balanced per file,
  * every \\begin{X} has a matching \\end{X},
  * the citation-control table is back on the page: table_citation_control.tex
    is \input somewhere and its key tab:citation is defined (the rebuilt E7
    control replaced the placeholder that had replaced the table).

Exit code is non-zero if any check fails, so it can be used in a release gate.
"""
from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "artifacts" / "rcgat-revision-package"
ORIG = PKG / "original"

COMMENT = re.compile(r"(?<!\\)%.*$")
LABEL = re.compile(r"\\label\{([^}]*)\}")
REF = re.compile(r"\\(?:ref|eqref|Cref|cref|autoref)\{([^}]*)\}")
INPUT = re.compile(r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]*)\}")
BEGIN = re.compile(r"\\begin\{([^}]*)\}")
END = re.compile(r"\\end\{([^}]*)\}")


def strip_comments(text: str) -> str:
    return "\n".join(COMMENT.sub("", line) for line in text.splitlines())


def manuscript_dir(name: str) -> str:
    """Where a file of the original manuscript lives, by naming convention."""
    if name.startswith("table_"):
        return "tables"
    if name.startswith("fig_") and name.endswith(".pdf"):
        return "figures"
    if re.match(r"^(01|03|04|05|05b|06|07|08|09|10)_", name) or name.startswith("highlights"):
        return "sections"
    return ""


def install(dest: Path, pkg: Path = PKG, force: bool = False) -> None:
    """Recreate the manuscript layout in `dest`, then run apply.sh on the copy.

    `dest` is DELETED first and rebuilt from the package's original/ files, so
    it must be a scratch (new or empty) directory -- never a real manuscript
    tree.  A non-empty destination is refused unless force=True.
    """
    if dest.exists():
        if any(dest.iterdir()) and not force:
            raise SystemExit(
                f"refusing to wipe {dest}: the directory is not empty.\n"
                "This tool deletes everything under --install-dir and rebuilds it from\n"
                "the package's original/ files, so it must point at a NEW or EMPTY path\n"
                "(e.g. --install-dir /tmp/ms).  Never point it at your manuscript tree.\n"
                "Pass --force only if you really mean to delete the contents."
            )
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in (pkg / "original").iterdir():
        if not f.is_file():
            continue
        sub = manuscript_dir(f.name)
        out = dest / sub
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / f.name)
    res = subprocess.run(["bash", str(pkg / "apply.sh"), str(dest)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"apply.sh failed ({res.returncode}):\n{res.stdout}\n{res.stderr}")
    print(res.stdout.splitlines()[0])


def check(root: Path) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    labels: dict[str, str] = {}
    refs: list[tuple[str, str]] = []
    tex = sorted(root.rglob("*.tex"))
    if not tex:
        problems.append(f"no .tex files under {root}")
    for path in tex:
        rel = path.relative_to(root)
        raw = path.read_text(errors="replace")
        text = strip_comments(raw)
        for m in LABEL.finditer(text):
            key = m.group(1)
            if key in labels:
                problems.append(f"duplicate label {key} in {rel} and {labels[key]}")
            else:
                labels[key] = str(rel)
        for m in REF.finditer(text):
            for key in m.group(1).split(","):
                refs.append((key.strip(), str(rel)))
        for m in INPUT.finditer(text):
            target = m.group(1)
            if target.startswith("http"):
                continue
            cand = [root / target, root / (target + ".tex"),
                    root / (target + ".pdf"), root / (target + ".png")]
            if any(c.exists() for c in cand):
                continue
            msg = f"{rel}: {m.group(0)} does not resolve"
            if m.group(0).startswith("\\includegraphics"):
                warnings.append(msg)      # the package ships only its own figures
            else:
                problems.append(msg)      # a missing \input is fatal
        # delimiter balance
        body = text.replace("\\$", "").replace("\\%", "")
        if body.count("$") % 2:
            problems.append(f"{rel}: odd number of $ delimiters")
        opens = body.count("{")
        closes = body.count("}")
        if opens != closes:
            problems.append(f"{rel}: braces unbalanced ({opens} vs {closes})")
        begins, ends = BEGIN.findall(body), END.findall(body)
        if sorted(begins) != sorted(ends):
            only_b = set(begins) - set(ends)
            only_e = set(ends) - set(begins)
            problems.append(f"{rel}: begin/end mismatch (unclosed {sorted(only_b)}, "
                            f"unopened {sorted(only_e)})")
    for key, rel in refs:
        if key not in labels:
            problems.append(f"{rel}: reference to undefined label {key}")
    # The citation-network control was retired in v17 and rebuilt in v18 (E7), so
    # the package must ship the table and something must read it in.  (Until v17
    # this check ran the other way round: tab:citation had to be *absent*.)
    citation_table = root / "tables" / "table_citation_control.tex"
    if not citation_table.is_file():
        problems.append("tables/table_citation_control.tex is missing from the package")
    elif "tab:citation" not in labels:
        problems.append("tables/table_citation_control.tex is not read in anywhere")
    print(f"files checked: {len(tex)}  labels: {len(labels)}  references: {len(refs)}")
    return problems, warnings



def compare_with_build(root: Path, expect: Path) -> tuple[list[str], list[str]]:
    """Byte-compare the installed tree with a built manuscript tree.

    This is the check that says the package *is* the manuscript: installing
    apply.sh over the original files must reproduce the tree that was compiled
    and audited.  Returns (problems, notes).
    """
    problems: list[str] = []
    notes: list[str] = []
    if not expect.is_dir():
        return [f"expected manuscript tree {expect} does not exist"], notes

    def pair(rel: str):
        return root / rel, expect / rel

    def differs(a: Path, b: Path) -> bool:
        """Installed vs built, tolerating the author-facing comment blocks.

        Every .tex twin is <comment block> + <the compiled file>, so the
        installed copy must END with the built file and the remaining prefix
        must be comments only.  That is exactly the invariant the package
        promises: applying it reproduces the manuscript that was compiled and
        audited, comment blocks aside.
        """
        if a.suffix.lower() == ".pdf":
            return a.read_bytes() != b.read_bytes()
        ia = a.read_text(errors="replace")
        ib = b.read_text(errors="replace")
        if ia == ib:
            return False
        if ia.endswith(ib):
            head = ia[: len(ia) - len(ib)]
            if all(l.startswith("%") or not l.strip() for l in head.splitlines()):
                return False
        return True

    checked = 0
    for src in sorted((PKG / "sections").glob("*.revised.*")):
        name = src.name.replace(".revised", "")
        a, b = pair(f"sections/{name}")
        sub = "" if name.startswith("highlights") else "sections/"
        if not b.exists():
            b = expect / sub / name
        if not b.exists():
            notes.append(f"{name}: not in the built tree, skipped")
            continue
        checked += 1
        if differs(a, b):
            problems.append(f"{name}: installed copy differs from the built manuscript")
    for name in ("07_results.tex",):
        a, b = pair(f"sections/{name}")
        checked += 1
        if not b.exists() or differs(a, b):
            problems.append(f"{name}: installed copy differs from the built manuscript")
    for folder in ("tables", "figures"):
        for src in sorted((PKG / folder).iterdir()):
            if not src.is_file():
                continue
            a, b = pair(f"{folder}/{src.name}")
            checked += 1
            if not b.exists():
                notes.append(f"{folder}/{src.name}: not in the built tree, skipped")
                checked -= 1
                continue
            if differs(a, b):
                problems.append(f"{folder}/{src.name}: installed copy differs from the built manuscript")
    for name in ("references.bib", "Orcidlogo.pdf"):
        a, b = pair(name)
        if not b.exists():
            continue
        checked += 1
        if differs(a, b):
            problems.append(f"{name}: installed copy differs from the built manuscript")
    # main.tex is edited in place, so a mismatch is a note rather than a failure:
    # the tree the package is installed over may have moved on since the snapshot.
    a, b = pair("main.tex")
    if b.exists():
        ia, ib = a.read_text(errors="replace"), b.read_text(errors="replace")
        code = lambda s: "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("%"))
        if ia == ib:
            notes.append("main.tex: identical to the built tree after the seven edits")
        elif code(ia) == code(ib):
            notes.append("main.tex: identical to the built tree once comment lines are ignored")
        else:
            diff = [l for l in difflib.unified_diff(
                code(ib).splitlines(), code(ia).splitlines(),
                "built", "installed", lineterm="", n=0)
                if l[:1] in "+-" and l[:3] not in ("+++", "---")]
            notes.append(f"main.tex: {len(diff)} non-comment line(s) differ from the built tree")
            for l in diff[:20]:
                notes.append(f"    {l}")
    print(f"compared with {expect}: {checked} file(s)")
    return problems, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-dir", default=None,
                    help="scratch directory to install into (wiped first; must be new or empty)")
    ap.add_argument("--force", action="store_true",
                    help="allow --install-dir to be a non-empty directory (it will be wiped)")
    ap.add_argument("--expect", default=None,
                    help="manuscript tree the installed copy must equal (default:"
                         " build/manuscript next to this repository, if it exists)")
    ap.add_argument("--package-dir", default=None,
                    help="path to the route-A revision package (default: the sibling"
                         " artifacts/rcgat-revision-package of this repository)")
    args = ap.parse_args()

    pkg = Path(args.package_dir).resolve() if args.package_dir else PKG
    if not (pkg / "apply.sh").exists():
        print(f"no revision package found at {pkg}\n"
              "This checker installs the route-A revision package with its apply.sh.  In the\n"
              "published repro repository that package is deliberately absent -- it ships with\n"
              "the manuscript submission, not with the pipeline.  Either run this checker from\n"
              "the submission tree (where artifacts/rcgat-revision-package sits next to repro/)\n"
              "or point at the package explicitly:\n"
              "    python tools/check_revision_package.py --package-dir /path/to/rcgat-revision-package")
        return 0

    if args.install_dir:
        root = Path(args.install_dir)
        install(root, pkg=pkg, force=args.force)
    else:
        root = Path(tempfile.mkdtemp(prefix="rcgat-revision-check-"))
        install(root, pkg=pkg)
        print(f"installed into {root}")
    problems, warnings = check(root)

    expect = args.expect
    if expect is None:
        cand = Path(__file__).resolve().parents[2] / "build" / "manuscript"
        if cand.is_dir():
            expect = cand
    if expect is not None:
        cmp_problems, notes = compare_with_build(root, Path(expect))
        problems += cmp_problems
        if notes:
            print(f"\n{len(notes)} note(s) from the comparison:")
            for n in notes:
                print(f"  - {n}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nall structural checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
