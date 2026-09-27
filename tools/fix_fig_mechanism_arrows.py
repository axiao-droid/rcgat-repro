"""Re-issue fig_mechanism.pdf: repair the two geometry defects the author circled.

The schematic is the author's own drawing and stays architecture-accurate; this
script only repairs geometry that is broken in the submitted vector PDF.  Run
tools/fix_fig_mechanism_labels.py first (it rewrites the two gate-box labels);
this script takes its output.  Two defects:

1.  **text occlusion.**  The gate arrow that rises from the region-conditioned
    gate box into the out-view multiplication node ends *on top of* the label
    ``x g_out``: its head (apex ``(308.26, 213.10)``) is drawn exactly over the
    subscript ``out`` of the label, whose ink is ``x[299.62,316.41]
    y[211.64,217.81]``.  Fix: the label is translated rigidly by
    ``(+1.10, -25.66)`` pt, which puts it above the node at the same offset that
    the mirror-image label ``x g_in`` keeps above the in-view node (ink bottom
    3.44 pt above the circle, ink left edge 0.63 pt left of the circle's left
    edge).  The five glyph outlines are re-emitted with the same path operators,
    so the typeface stays identical to ``x g_in``; the arrow head the label used
    to cover is re-drawn unchanged.

2.  **head-only connectors.**  Three connectors are drawn as an arrow head whose
    shaft is absent or completely hidden under the head, so the head reads as a
    stray triangle lying on the box border:

      * ``out-view aggregation`` -> out-view node: shaft 2.25 pt long, head base
        1.05 pt *inside* the border, tip 3.66 pt short of the node;
      * ``embedding`` -> ``cosine``: shaft 4.22 pt, all of it under the head
        (head base 653.08 < shaft start 653.26);
      * ``cosine`` -> ``BCE loss``: shaft 1.75 pt, all of it under the head.

    Fix: each head is re-placed so that it spans the gap - base flush with the
    source box, tip at the target - and the ``embedding`` -> ``cosine``
    connector gets a shaft as well (1.8 pt visible, of the same order as the
    2.8 pt the two working connectors of that strip already show).  Head size
    stays the author's (4.4 pt long / 4.4 pt wide) wherever the gap allows and
    is scaled down where it does not, keeping the author's apex angle, drawn the way the author draws
    them: filled *and* stroked in the same colour with width 1.5, which inflates
    the outline by 0.75 pt on every side.

Nothing else is touched: no box, no text, no font is added or removed.  The
output is byte-reproducible: run it twice and you get the same sha256 (the
trailer /ID is normalised, see normalise_id).  The
script refuses to run unless the input has exactly the expected geometry, so it
cannot silently patch the wrong figure.

Usage::

    python tools/fix_fig_mechanism_arrows.py --src <in.pdf> --out <out.pdf>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]

# palette of the figure (read off the source PDF)
ORANGE = (0.9490196108818054, 0.5568627715110779, 0.16862745583057404)
TEAL = (0.4627451002597809, 0.7176470756530762, 0.6980392336845398)
GREEN = (0.3490196168422699, 0.6313725709915161, 0.30980393290519714)

# --- 1. the "x g_out" label -------------------------------------------------
# redaction rectangle: the label ink, its whole text span box, and the head
LABEL_RECT = fitz.Rect(298.8, 206.9, 316.9, 220.4)
LABEL_SHIFT = (1.10, -25.66)  # rigid translation, taken from the x g_in offset
GATE_HEAD = ((306.06, 217.50), (308.26, 213.10), (310.46, 217.50))

# --- 2. the three head-only connectors --------------------------------------
# out-view aggregation box (right border path x=294.34) -> out-view node
# (circle x[301.35,315.17], stroke 1.7 -> outer edge 300.50); the 5.56 pt gap
# leaves no room for a shaft, so the head spans it (outer base on the border,
# miter tip 301.03, i.e. 0.32 pt clear of the white fill of the node).
OUT_VIEW_RECT = fitz.Rect(293.0, 200.0, 297.9, 205.0)
OUT_VIEW_HEAD = ((294.94, 200.31), (299.35, 202.51), (294.94, 204.71))
# embedding box (right border x=652.16) -> cosine box (left border x=660.26):
# head 3.6 pt; outer base 654.55 (leaves a 1.8 pt shaft), miter tip 660.58,
# i.e. 0.28 pt clear of the white fill of the cosine box.
STRIP_B_RECT = fitz.Rect(652.9, 152.9, 657.9, 158.2)
STRIP_B_SHAFT = ((652.30, 155.54), (655.50, 155.54))
STRIP_B_HEAD = ((655.30, 153.74), (658.90, 155.54), (655.30, 157.34))
# cosine box (right border x=708.92) -> BCE loss box (left border x=714.05):
# head 2.9 pt, the largest that fits: outer base 708.45 and miter tip 713.78
# both land inside the neighbouring border strokes, never on a box fill.
STRIP_C_RECT = fitz.Rect(706.5, 152.9, 711.7, 158.2)
STRIP_C_HEAD = ((709.20, 154.09), (712.10, 155.54), (709.20, 156.99))

LINE_WIDTH = 1.5


def emit(shape: fitz.Shape, items, dx: float = 0.0, dy: float = 0.0) -> None:
    """Re-emit one recorded path, optionally translated."""
    for item in items:
        op = item[0]
        if op == "l":
            shape.draw_line(
                (item[1].x + dx, item[1].y + dy), (item[2].x + dx, item[2].y + dy)
            )
        elif op == "c":
            shape.draw_bezier(*[(p.x + dx, p.y + dy) for p in item[1:]])
        elif op == "re":
            r = item[1]
            shape.draw_rect(fitz.Rect(r.x0 + dx, r.y0 + dy, r.x1 + dx, r.y1 + dy))
        else:  # "qu" and anything else never occurs in this figure
            raise SystemExit(f"unsupported path operator {op!r}")


def triangle(shape: fitz.Shape, pts, colour) -> None:
    """A filled + stroked arrow head, drawn exactly like the author's heads."""
    shape.draw_polyline(list(pts) + [pts[0]])
    shape.finish(fill=colour, color=colour, width=LINE_WIDTH,
                 even_odd=False, closePath=True)


def rewrite(src: str, out: str) -> str:
    doc = fitz.open(src)
    page = doc[0]
    paths = page.get_drawings()

    glyphs = [
        d
        for d in paths
        if LABEL_RECT.contains(d["rect"]) and d.get("fill") == ORANGE
        and len(d["items"]) != 3
    ]
    covered = [
        d for d in paths if LABEL_RECT.contains(d["rect"]) and len(d["items"]) == 3
    ]
    if len(glyphs) != 5 or len(covered) != 1:
        raise SystemExit(
            "label geometry changed: expected 5 glyph outlines and 1 arrow head "
            f"inside {LABEL_RECT}, found {len(glyphs)} and {len(covered)}"
        )
    for name, rect in (
        ("out-view connector", OUT_VIEW_RECT),
        ("embedding -> cosine", STRIP_B_RECT),
        ("cosine -> BCE loss", STRIP_C_RECT),
    ):
        found = [d for d in paths if rect.contains(d["rect"])]
        if len(found) != 2:
            raise SystemExit(
                f"{name}: expected 2 paths inside {rect}, found {len(found)}"
            )

    for rect in (LABEL_RECT, OUT_VIEW_RECT, STRIP_B_RECT, STRIP_C_RECT):
        page.add_redact_annot(rect)
    page.apply_redactions(
        images=fitz.PDF_REDACT_IMAGE_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
    )

    shape = page.new_shape()
    dx, dy = LABEL_SHIFT
    for d in glyphs:
        emit(shape, d["items"], dx, dy)
        shape.finish(fill=ORANGE, color=None, even_odd=False, closePath=True)
    triangle(shape, GATE_HEAD, ORANGE)
    triangle(shape, OUT_VIEW_HEAD, TEAL)
    shape.draw_line(*STRIP_B_SHAFT)
    shape.finish(color=GREEN, width=LINE_WIDTH, lineCap=0)
    triangle(shape, STRIP_B_HEAD, GREEN)
    triangle(shape, STRIP_C_HEAD, GREEN)
    shape.commit()

    page.clean_contents()
    doc.subset_fonts()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    doc.save(out, garbage=4, deflate=True, clean=True, use_objstms=1)
    doc.close()
    normalise_id(out)
    return out


def normalise_id(path: str) -> None:
    """Make the output byte-reproducible.

    MuPDF writes a fresh random second half into the trailer's ``/ID`` array on
    every save, so two runs of this script on the same input differ in exactly
    those 32 bytes (and nothing else).  We copy the first half over the second,
    which is what the ``/ID [<permanent><changing>]`` convention allows and what
    makes the released figure hash-stable.  Same length in, same length out, so
    no byte offset can move.
    """
    import re

    data = Path(path).read_bytes()
    match = re.search(rb"/ID\s*\[<([0-9A-Fa-f]{32})><([0-9A-Fa-f]{32})>\]", data)
    if match is None:
        raise SystemExit(
            f"no /ID array found in {path}; cannot normalise it ("
            f"tail: {data[-400:]!r})"
        )
    if match.group(1) == match.group(2):
        return
    Path(path).write_bytes(
        data[: match.start(2)] + match.group(1) + data[match.end(2):]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "revision_figures" / "fig_mechanism.pdf"),
    )
    args = ap.parse_args()
    src = args.src
    if src is None:
        for candidate in (
            ROOT / "results" / "revision_figures" / "fig_mechanism.pdf",
            Path("/tmp/proj2/fig_mechanism.pdf"),
        ):
            if candidate.exists():
                src = str(candidate)
                break
        else:
            raise SystemExit("no label-fixed fig_mechanism.pdf found; pass --src")
    out = rewrite(src, args.out)
    page = fitz.open(out)[0]
    moved = [
        d["rect"]
        for d in page.get_drawings()
        if fitz.Rect(297, 178, 320, 196).contains(d["rect"])
        and d.get("fill") == ORANGE
    ]
    left = sum(r.x0 for r in moved) / max(len(moved), 1)
    print(
        f"wrote {out} ({os.path.getsize(out)} bytes); "
        f"{len(moved)} glyph outlines now above the node (x0 mean {left:.2f})"
    )
    for line in page.get_text().splitlines():
        if "tanh" in line or "epoch zero" in line:
            print("  label:", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
