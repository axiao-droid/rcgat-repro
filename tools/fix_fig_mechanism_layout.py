"""Layout surgery on fig_mechanism.pdf (three visual defects found in review):

1. The ``xg_out`` gate label collided with the gate circle below it; it is
   re-drawn directly ABOVE the out-gate circle, mirroring the ``xg_in``
   placement (same font, size, colour, same centre-offsets).
2. ``mean-scatter`` in the edge-time branch box ran past the dashed border
   (x = 452.7 vs border 452.3); it is re-set at a slightly smaller size so it
   ends inside the border.
3. The two connector arrows between embedding/cosine/BCE cells were bare
   arrowheads (4.4 pt, one even starting inside the cosine box); they are
   replaced by proper shaft + head arrows filling the cell gaps.

Usage::

    python tools/fix_fig_mechanism_layout.py --src <in.pdf> --out <out.pdf>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import fitz  # PyMuPDF

ORANGE = (0xF2 / 255, 0x8E / 255, 0x2B / 255)   # gate label colour (#F28E2B)
RED = (0xE1 / 255, 0x57 / 255, 0x59 / 255)      # edge-time branch red (#E15759)
GREEN = (0x59 / 255, 0xA1 / 255, 0x4F / 255)    # rcgat_time legend green (#59A14F)

# ---------------------------------------------------------------- gate label
IN_CIRCLE_CENTRE = (332.90, 108.55)     # in-gate circle centre (from drawings)
OUT_CIRCLE_CENTRE = (308.30, 202.50)    # out-gate circle centre
IN_LABEL_ORIGIN_X = 325.44              # '×' origin of the (correct) ×g_in label
IN_LABEL_BASELINE = 94.45
GATE_MAIN_SIZE = 8.40
GATE_SUB_SIZE = 5.88
GATE_SUB_DROP = 1.06                    # sub-script baseline drop (from ×g_in)

# ------------------------------------------------------------- edge-time box
MS_ORIGIN = (415.19, 232.60)            # ' mean-scatter' original origin
MS_BOX_BORDER_X = 452.30                # dashed border right edge
MS_PAD = 3.30                           # keep this much inside the border

# ------------------------------------------------------------------- arrows
ARROW_W = 1.5
HEAD_LEN = 4.4
HEAD_HALF = 2.2
# (shaft_start_x, shaft_end_x == head_base_x, tip_x, y) per gap; tips keep a
# small clearance to the neighbouring box border stroke.
ARROWS = [
    (653.0, 655.0, 659.4, 155.5),       # embedding -> cosine
    (709.7, 711.0, 713.3, 155.5),       # cosine  -> BCE loss
]


def collect_fonts(doc: fitz.Document, page: fitz.Page):
    """Return (regular, italic) STIXGeneral buffers used by the figure."""
    reg = ita = None
    for xref, _ext, _ftype, basefont, _name, _enc in page.get_fonts():
        if "STIXGeneral" not in basefont:
            continue
        buf = doc.extract_font(xref)[3]
        if not buf:
            continue
        if "Italic" in basefont:
            ita = buf
        else:
            reg = buf
    return reg, ita


def gate_label_fallback():
    """Windows Times New Roman as a visual stand-in if the subset lacks glyphs."""
    reg = fitz.Font(fontfile=r"C:\Windows\Fonts\times.ttf")
    ita = fitz.Font(fontfile=r"C:\Windows\Fonts\timesi.ttf")
    return reg, ita


def rewrite(src: str, out: str) -> str:
    doc = fitz.open(src)
    page = doc[0]

    # -- 1. remove the misplaced ×g_out text (graphics untouched) -----------
    page.add_redact_annot(fitz.Rect(298.5, 206.5, 318.5, 221.0), fill=False)
    # -- 2. remove the overflowing ' mean-scatter' text ----------------------
    page.add_redact_annot(fitz.Rect(415.0, 225.0, 456.0, 236.0), fill=False)
    # -- 3. remove the two bare arrowheads (only paths fully covered) --------
    for x0, _s, tipx, y in ARROWS:
        page.add_redact_annot(fitz.Rect(x0 + 0.1, y - HEAD_HALF - 0.6,
                                        tipx + 0.1, y + HEAD_HALF + 0.6),
                              fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                          graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)

    # -- re-draw the gate label above the out-gate circle --------------------
    reg_buf, ita_buf = collect_fonts(doc, page)
    freg = fitz.Font(fontbuffer=reg_buf) if reg_buf else None
    fita = fitz.Font(fontbuffer=ita_buf) if ita_buf else None
    need = "×gout"
    for f in (freg, fita):
        if f is None:
            break
        missing = [c for c in need if not f.has_glyph(ord(c))]
        if missing:
            print(f"note: font subset lacks {missing}; falling back to Times")
            freg = fita = None
    if freg is None or fita is None:
        freg, fita = gate_label_fallback()

    dx = IN_LABEL_ORIGIN_X - IN_CIRCLE_CENTRE[0]        # -7.46
    baseline = OUT_CIRCLE_CENTRE[1] + (IN_LABEL_BASELINE - IN_CIRCLE_CENTRE[1])
    x_times = OUT_CIRCLE_CENTRE[0] + dx
    tw = fitz.TextWriter(page.rect)
    tw.append((x_times, baseline), "×", font=freg, fontsize=GATE_MAIN_SIZE)
    x_g = x_times + freg.text_length("×", fontsize=GATE_MAIN_SIZE)
    tw.append((x_g, baseline), "g", font=fita, fontsize=GATE_MAIN_SIZE)
    x_sub = x_g + fita.text_length("g", fontsize=GATE_MAIN_SIZE)
    tw.append((x_sub, baseline + GATE_SUB_DROP), "out", font=fita,
              fontsize=GATE_SUB_SIZE)
    tw.write_text(page, color=ORANGE)

    # -- re-set ' mean-scatter' inside the dashed border ---------------------
    tnr = fitz.Font("tiro")  # built-in Times-Roman (same metrics as PSMT)
    avail = MS_BOX_BORDER_X - MS_PAD - MS_ORIGIN[0]
    size = 7.0
    while size > 5.5 and tnr.text_length(" mean-scatter", fontsize=size) > avail:
        size -= 0.1
    page.insert_text(MS_ORIGIN, " mean-scatter", fontname="tiro", fontsize=size,
                     color=RED)
    print(f"mean-scatter re-set at {size:.1f} pt (ends at "
          f"{MS_ORIGIN[0] + tnr.text_length(' mean-scatter', fontsize=size):.1f}, "
          f"border at {MS_BOX_BORDER_X})")

    # -- draw the two proper arrows ------------------------------------------
    for sx, hx, tipx, y in ARROWS:
        page.draw_line(fitz.Point(sx, y), fitz.Point(hx, y),
                       color=GREEN, width=ARROW_W)
        page.draw_polyline(
            [fitz.Point(tipx, y), fitz.Point(hx, y - HEAD_HALF),
             fitz.Point(hx, y + HEAD_HALF), fitz.Point(tipx, y)],
            color=GREEN, fill=GREEN, width=0.8, closePath=True)

    page.clean_contents()
    doc.subset_fonts()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    doc.save(out, garbage=4, deflate=True, clean=True, use_objstms=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = rewrite(args.src, args.out)
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
