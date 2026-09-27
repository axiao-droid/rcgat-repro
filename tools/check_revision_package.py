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
  * no file still mentions the retired citation-control table key tab:citation.

Exit code is non-zero if any check fails, so it can be used in a release gate.
"""
from __future__ import annotations

import argparse
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


def install(dest: Path) -> None:
    """Recreate the manuscript layout, then run apply.sh on the copy."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for f in ORIG.iterdir():
        if not f.is_file():
            continue
        sub = manuscript_dir(f.name)
        out = dest / sub
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / f.name)
    res = subprocess.run(["bash", str(PKG / "apply.sh"), str(dest)],
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
        if key == "tab:citation":
            problems.append(f"{rel}: still references the retired table tab:citation")
    print(f"files checked: {len(tex)}  labels: {len(labels)}  references: {len(refs)}")
    return problems, warnings



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install-dir", default=None)
    args = ap.parse_args()
    if args.install_dir:
        root = Path(args.install_dir)
        install(root)
    else:
        root = Path(tempfile.mkdtemp(prefix="rcgat-revision-check-"))
        install(root)
        print(f"installed into {root}")
    problems, warnings = check(root)
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
