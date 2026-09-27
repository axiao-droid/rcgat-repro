"""Re-issue fig_mechanism.pdf with the gate labelled as the code implements it.

The submitted figure labels the gate ``g_d = 1 + tanh(MLP(rho(u)))`` and adds
``MLP zero-init`` below it.  The released encoder implements a *single* affine
map from the 13-dimensional descriptor to a scalar (``_Gate.mlp =
nn.Linear(13, 1)``, weight and bias zero-initialised, 14 parameters per gate,
4 gates -> 56 parameters per encoder), so "MLP" overstates the component.  The
schematic itself is architecture-accurate and is kept unchanged: only the two
labels inside the right-hand gate box are redrawn.

The replacement text is written with Nimbus Roman (URW's Times clone, embedded
as a subset) so that the figure stays font-complete for production, and the
Symbol glyph rho is taken from the italic cut of the same family.

Usage::

    python tools/fix_fig_mechanism_labels.py --src <in.pdf> --out <out.pdf>

Without arguments it reads the unmodified figure shipped in the repository
(``assets/fig_mechanism.original.pdf``, the version that was in the submission
tree; ``/tmp/proj2/fig_mechanism.pdf`` and ``figures/fig_mechanism.pdf`` are
tried next) and writes ``results/revision_figures/fig_mechanism.pdf``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
REGULAR = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"
ITALIC = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf"

# the two labels to remove, in PDF points (from the submitted figure)
GATE_FORMULA_BOX = fitz.Rect(324, 297.5, 430, 317.5)
GATE_SENTENCE_BOX = fitz.Rect(321, 322.0, 434, 338.0)
GATE_BOX_CENTRE_X = 377.35


def rewrite(src: str, out: str) -> str:
    for path in (REGULAR, ITALIC):
        if not os.path.exists(path):
            raise SystemExit(f"missing font {path}; install urw-base35-fonts")

    reg, ita = fitz.Font(fontfile=REGULAR), fitz.Font(fontfile=ITALIC)
    doc = fitz.open(src)
    page = doc[0]

    for rect in (GATE_FORMULA_BOX, GATE_SENTENCE_BOX):
        page.add_redact_annot(rect)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    def width(text: str, family: str, size: float) -> float:
        return (reg if family == "NR" else ita).text_length(text, fontsize=size)

    def draw(pieces, baseline: float) -> None:
        """pieces: (text, family, size, raise) with raise in points."""
        total = sum(width(t, f, s) for t, f, s, _ in pieces)
        x = GATE_BOX_CENTRE_X - total / 2.0
        for text, family, size, raise_ in pieces:
            page.insert_text((x, baseline - raise_), text, fontname=family, fontsize=size,
                             fontfile=(REGULAR if family == "NR" else ITALIC),
                             color=(0, 0, 0))
            x += width(text, family, size)

    # g_d = 1 + tanh(w^T rho(u) + b)
    draw([
        ("g", "NI", 9.5, 0.0), ("d", "NI", 6.7, -2.6),
        (" = 1 + tanh(", "NR", 9.5, 0.0),
        ("w", "NI", 9.5, 0.0), ("T", "NI", 6.7, 2.6),
        ("ρ", "NI", 9.5, 0.0),
        ("(", "NR", 9.5, 0.0), ("u", "NI", 9.5, 0.0), (") + ", "NR", 9.5, 0.0),
        ("b", "NI", 9.5, 0.0), (")", "NR", 9.5, 0.0),
    ], baseline=312.0)

    # single linear layer, zero-init, g_d = 1 at epoch zero
    draw([
        ("linear layer, zero-init, ", "NR", 7.6, 0.0),
        ("g", "NI", 7.6, 0.0), ("d", "NI", 5.3, -2.0),
        (" = 1 at epoch zero", "NR", 7.6, 0.0),
    ], baseline=332.5)

    page.clean_contents()
    doc.subset_fonts()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    doc.save(out, garbage=4, deflate=True, clean=True, use_objstms=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "revision_figures" / "fig_mechanism.pdf"))
    args = ap.parse_args()
    src = args.src
    if src is None:
        for candidate in (ROOT / "assets" / "fig_mechanism.original.pdf",
                          Path("/tmp/proj2/fig_mechanism.pdf"),
                          ROOT / "figures" / "fig_mechanism.pdf"):
            if candidate.exists():
                src = str(candidate)
                break
        else:
            raise SystemExit("no unmodified fig_mechanism.pdf found; pass --src")
    out = rewrite(src, args.out)
    doc = fitz.open(out)
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    for line in doc[0].get_text().splitlines():
        if "tanh" in line or "epoch zero" in line:
            print("  label:", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
