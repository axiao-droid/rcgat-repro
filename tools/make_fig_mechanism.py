# -*- coding: utf-8 -*-
"""
Mechanism figure (RC-GAT) -- precise vector redraw, journal-body restrained style.

Authoring model
---------------
The page IS the final print size for `\\includegraphics[width=\\textwidth]` in the
Springer `sn-jnl[sn-basic]` single-column layout: 415 pt wide, so every font size
below is the literal printed size (no downscaling -> no unreadable 5 pt text).
This is the central repair of the original figure, whose 9.5 pt labels shrank to
~5 pt when the 762 pt-wide artwork was placed at \\textwidth.

Deliverables (all from this one coordinate table, so they cannot drift apart):
  fig_mechanism_redraw.pdf    -> pdflatex \\includegraphics
  fig_mechanism_redraw.svg    -> real <text> elements, editable
  fig_mechanism_redraw.png    -> preview used for the visual diff loop

Geometry: see LAYOUT / the zone/card tables in draw_*(). Occlusion safety is by
construction: every label is centred/right-aligned using measured font metrics
(tw) inside its own card, arrowheads are filled-only triangles (no stroked miter
tip, the defect of the original), and gate risers run in a corridor free of cards.
"""
import os
import fitz

# ---------------------------------------------------------------- palette ----
# Anchored to the author's own artwork (Tableau-10 strokes, ~12 % tint fills).
GREY, ORANGE, GREEN, TEAL, BLUE, RED = (
    "#5A5A5A", "#F28E2B", "#59A14F", "#76B7B2", "#4E79A7", "#E15759")
TEAL_D, ORANGE_D, GREEN_D, RED_D, BLUE_D = (
    "#2F6E6A", "#B5651D", "#3C7A35", "#B03A3C", "#2F5F8F")
INK, SUB = "#333333", "#5A5A5A"
T_TEAL, T_ORANGE, T_GREEN, T_BLUE, T_GREY = (
    "#F4FAF9", "#FDF6EE", "#F3F9F1", "#F2F6FB", "#F7F7F7")

W, H = 415.0, 232.0          # page = final print size (pt)

# ----------------------------------------------------------------- fonts ----
# URW base35 (Nimbus Sans = the Helvetica clone this figure is set in).
# Debian/Ubuntu: apt-get install fonts-urw-base35 ; or point URW_FONT_DIR at it.
FDIR = os.environ.get("URW_FONT_DIR", "/usr/share/fonts/opentype/urw-base35/")
if not os.path.isdir(FDIR):
    raise SystemExit("URW base35 fonts not found at %s -- install fonts-urw-base35 "
                     "or set URW_FONT_DIR" % FDIR)
FILES = {"r": "NimbusSans-Regular.otf",
         "b": "NimbusSans-Bold.otf",
         "i": "NimbusSans-Italic.otf",
         "bi": "NimbusSans-BoldItalic.otf"}
_fonts = {k: fitz.Font(fontfile=FDIR + v) for k, v in FILES.items()}


def tw(s, sz, st="r"):
    """Measured text width in pt -- the basis of every alignment decision."""
    return _fonts[st].text_length(s, fontsize=sz)


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ------------------------------------------------------------ primitives ----
def R(t, st="r", sz=None, dy=0.0):
    return (t, st, sz, dy)


def M(sz, *parts):
    """Math run builder. kind: r/i = roman/italic, sup/sub = scripts."""
    out = []
    for t, k in parts:
        if k in ("r", "i", "b"):
            out.append(R(t, k, sz, 0.0))
        elif k == "sup":
            out.append(R(t, "r", sz * 0.70, -sz * 0.34))
        elif k == "sub":
            out.append(R(t, "r", sz * 0.70, sz * 0.24))
        elif k == "isup":
            out.append(R(t, "i", sz * 0.70, -sz * 0.34))
        elif k == "isub":
            out.append(R(t, "i", sz * 0.70, sz * 0.24))
        else:
            raise ValueError(k)
    return out


def runs(page, x, y, parts, anchor="l", color=INK):
    """Place a mixed-style run sequence; anchor uses measured widths."""
    if isinstance(parts, str):
        parts = [R(parts)]
    elif isinstance(parts, tuple):
        parts = [parts]
    parts = [(r[0], r[1] or "r", r[2] if r[2] is not None else 6.2, r[3]) for r in parts]
    wid = [tw(t, sz, st) for (t, st, sz, dy) in parts]
    tot = sum(wid)
    if anchor == "c":
        x -= tot / 2.0
    elif anchor == "r":
        x -= tot
    for (t, st, sz, dy), w in zip(parts, wid):
        page.insert_text((x, y + dy), t, fontsize=sz, fontname="f_" + st,
                         color=rgb(color))
        x += w
    return tot


def card(page, x0, y0, x1, y1, stroke, fill, w=0.85, dash=None, rad=2.4):
    """rad is given in pt; PyMuPDF wants a fraction of the shorter side."""
    frac = min(0.5, rad / max(1e-6, min(x1 - x0, y1 - y0)))
    page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=rgb(stroke), fill=rgb(fill),
                   width=w, radius=frac, dashes=dash)


def dot(page, x, y, r, stroke, fill, w=0.8):
    page.draw_circle(fitz.Point(x, y), r, color=rgb(stroke), fill=rgb(fill), width=w)


def arrow(page, pts, color, w=0.9, head=4.4, hw=2.0):
    """Polyline + filled-only triangular head. No stroked tip -> no miter bulge
    (the stroked head of the original overran its target by 1.68 pt)."""
    pts = [fitz.Point(*p) for p in pts]
    end, prev = pts[-1], pts[-2]
    d = fitz.Point(end.x - prev.x, end.y - prev.y)
    n = (d.x * d.x + d.y * d.y) ** 0.5 or 1.0
    ux, uy = d.x / n, d.y / n
    base = fitz.Point(end.x - ux * head, end.y - uy * head)
    page.draw_line(prev, base, color=rgb(color), width=w, lineCap=1, lineJoin=1)
    for i in range(len(pts) - 2):
        page.draw_line(pts[i], pts[i + 1], color=rgb(color), width=w, lineJoin=1)
    px, py = -uy, ux
    page.draw_polyline([end,
                        fitz.Point(base.x + px * hw, base.y + py * hw),
                        fitz.Point(base.x - px * hw, base.y - py * hw)],
                       color=None, fill=rgb(color), closePath=True)


def cross_gate(page, x, y, r, label_parts, lsz=6.4, color=TEAL_D, stroke=GREY):
    """(x) gate: circle + inscribed X, label centred above with clearance."""
    dot(page, x, y, r, stroke, "#FFFFFF", w=0.9)
    k = r * 0.60
    for sx, sy in ((-1, -1), (-1, 1)):
        page.draw_line(fitz.Point(x + sx * k, y + sy * k),
                       fitz.Point(x - sx * k, y - sy * k),
                       color=rgb(color), width=0.9, lineCap=1)
    if label_parts:
        runs(page, x, y - r - 2.2, label_parts, anchor="c", color=color)


def plus_gate(page, x, y, r, stroke=BLUE, color=BLUE_D):
    dot(page, x, y, r, stroke, "#FFFFFF", w=0.9)
    k = r * 0.62
    page.draw_line(fitz.Point(x - k, y), fitz.Point(x + k, y), color=rgb(color), width=0.9)
    page.draw_line(fitz.Point(x, y - k), fitz.Point(x, y + k), color=rgb(color), width=0.9)

# ---------------------------------------------------------------- layout ----
#  x band     col A (graph, legend)   6 .. 100
#             col B (encoder zone)  104 .. 306
#             col C (output stack)  314 .. 386 cards, 400 residual corridor
#  y bands    pipeline 8 .. 176 | bottom band 184 .. 226
def u_edge(cx, cy, r, tx, ty, pad=0.8):
    """Point on a node circle's boundary, toward (tx,ty); arrows never touch ink."""
    vx, vy = tx - cx, ty - cy
    n = (vx * vx + vy * vy) ** 0.5 or 1.0
    return (cx + vx / n * (r + pad), cy + vy / n * (r + pad))


def draw_graph(page):
    cx0, cx1, cy0, cy1 = 6, 100, 4, 118
    card(page, cx0, cy0, cx1, cy1, "#D6D6D6", T_GREY, w=0.7, rad=3.0)
    runs(page, 52, 15, R("directed graph", "b", 6.9, 0), "c", SUB)
    U = (72, 64, 9.5)                                   # subject node u
    IN = [(20, 44, 5.5), (20, 64, 5.5), (20, 84, 5.5)]   # in-neighbourhood
    OUT = [(34, 95, 5.5), (66, 95, 5.5)]                 # out-neighbourhood
    for (x, y, r) in IN:
        arrow(page, [u_edge(x, y, r, *U[:2]), u_edge(*U, x, y, pad=0.9)],
              GREY, w=0.8, head=3.6, hw=1.6)
    for (x, y, r) in OUT:
        arrow(page, [u_edge(*U, x, y), (x, y)], GREY, w=0.8, head=3.6, hw=1.6)
    for (x, y, r) in IN + OUT:
        dot(page, x, y, r, GREY, "#FFFFFF", w=0.8)
    runs(page, 52, 27, R("in-neighbours", None, 5.9, 0), "c", SUB)
    runs(page, 52, 35, M(5.9, ("N", "i"), ("in", "sub"), ("(u)", "r")), "c", TEAL_D)
    runs(page, 52, 113.5, R("out-neighbours", None, 5.9, 0), "c", SUB)
    runs(page, 52, 106.5, M(5.9, ("N", "i"), ("out", "sub"), ("(u)", "r")), "c", TEAL_D)
    dot(page, U[0], U[1], U[2], GREY, "#FFFFFF", w=1.0)
    runs(page, U[0], U[1] + 2.6, R("u", "i", 7.4, 0), "c", INK)
    return U


def draw_encoder(page):
    # ---- container: dashed zone = "one module", solid cards = "one operation"
    card(page, 104, 8, 306, 176, TEAL, T_TEAL, w=1.0, dash="[2.6 1.9] 0", rad=4.0)
    runs(page, 205, 20, R("Directed two-view attention encoder", "b", 7.2, 0), "c", TEAL_D)
    runs(page, 205, 29.5, R("2 layers, LayerNorm", None, 6.1, 0), "c", SUB)

    for (y0, yn, tag) in ((36, "in", "in"), (92, "out", "out")):
        card(page, 110, y0, 184, y0 + 36, TEAL, "#FFFFFF", w=0.85)
        runs(page, 147, y0 + 10.5, R(tag + "-view aggregation", "b", 6.6, 0), "c", TEAL_D)
        runs(page, 147, y0 + 20.5, R("multi-head GAT weights", None, 6.1, 0), "c", INK)
        runs(page, 147, y0 + 30.5,
             M(6.1, ("α", "i"), (f"({tag})", "sup"), ("uv", "sub"),
               (" over ", "r"), ("N", "i"), (tag, "sub"), ("(u)", "r")), "c", INK)

    # gates sit in the free corridor between the cards and the concat card
    cross_gate(page, 200, 54, 7, M(6.4, ("×", "r"), ("g", "i"), ("in", "sub")), color=ORANGE_D)
    cross_gate(page, 216, 110, 7, M(6.4, ("×", "r"), ("g", "i"), ("out", "sub")), color=ORANGE_D)
    arrow(page, [(184, 54), (191.6, 54)], TEAL, w=0.9, head=3.4, hw=1.5)
    arrow(page, [(184, 110), (207.6, 110)], TEAL, w=0.9, head=3.4, hw=1.5)
    arrow(page, [(207, 54), (226, 54)], TEAL, w=0.9, head=3.4, hw=1.5)   # in -> concat
    # out-view route: right, then a short rise into the concat card's bottom edge
    arrow(page, [(223, 110), (240, 110), (240, 104.6)], TEAL, w=0.9, head=3.4, hw=1.5)

    card(page, 226, 36, 300, 104, GREY, "#FFFFFF", w=0.85)
    runs(page, 263, 47, R("concat", "b", 6.7, 0), "c", INK)
    runs(page, 263, 57, R("in | out views", None, 6.4, 0), "c", INK)
    runs(page, 263, 68, R("2 directed GAT layers", None, 6.1, 0), "c", INK)
    runs(page, 263, 78, R("+ LayerNorm", None, 6.1, 0), "c", INK)
    runs(page, 263, 89, M(6.1, ("h", "i"), ("u", "sub"), (" = branch", "r")), "c", INK)

    card(page, 226, 130, 300, 172, RED, "#FDF3F3", w=0.9, dash="[2.2 1.7] 0")
    runs(page, 263, 140, R("edge-time branch", "b", 6.4, 0), "c", RED_D)
    runs(page, 263, 149.5, R("(optional)", None, 5.9, 0), "c", RED_D)
    runs(page, 263, 159,
         M(6.0, ("ψ", "i"), ("uv", "sub"), (" = log(1 + ", "r"), ("Δ", "i"), ("D", "i"), (")", "r")),
         "c", INK)
    runs(page, 263, 168.5, M(6.0, ("W", "i"), ("t", "sub"), (" mean-scatter", "r")), "c", INK)
    arrow(page, [(263, 130), (263, 104.6)], RED, w=0.9, head=3.4, hw=1.5)


def draw_output(page):
    card(page, 314, 42, 386, 66, BLUE, T_BLUE, w=0.85)
    runs(page, 350, 51.5, R("U (zero-init)", "b", 6.7, 0), "c", BLUE_D)
    runs(page, 350, 61.5, R("low-rank projection", None, 6.1, 0), "c", INK)
    arrow(page, [(300, 54), (314, 54)], GREY, w=0.95, head=4.2, hw=1.9)

    plus_gate(page, 350, 78, 6.0)
    arrow(page, [(350, 66), (350, 72)], BLUE, w=0.9, head=3.2, hw=1.4)

    card(page, 314, 89, 386, 113, GREEN, T_GREEN, w=0.85)
    runs(page, 350, 98.5, R("embedding", "b", 6.7, 0), "c", GREEN_D)
    runs(page, 350, 108.5, M(6.1, ("z", "i"), ("u", "sub"), (" = ", "r"), ("x", "i"),
                             ("u", "sub"), (" + ", "r"), ("U", "i"), ("h", "i"), ("u", "sub")),
         "c", INK)

    card(page, 314, 118, 386, 142, GREEN, T_GREEN, w=0.85)
    runs(page, 350, 127.5, R("cosine", "b", 6.7, 0), "c", GREEN_D)
    runs(page, 350, 137.5, M(6.1, ("s", "i"), (" = cos(", "r"), ("z", "i"), ("u", "sub"),
                             (", ", "r"), ("z", "i"), ("v", "sub"), (")", "r")), "c", INK)

    card(page, 314, 147, 386, 171, GREEN, T_GREEN, w=0.85)
    runs(page, 350, 156.5, R("BCE loss", "b", 6.7, 0), "c", GREEN_D)
    runs(page, 350, 166.5, R("1 pos, 10 neg", None, 6.1, 0), "c", INK)

    for y in (89, 118, 147):
        arrow(page, [(350, y - 5), (350, y)], GREEN, w=0.95, head=4.0, hw=1.8)


def draw_bottom(page):
    # content input -- its only destination is the residual junction (short route)
    card(page, 306, 184, 409, 226, BLUE, T_BLUE, w=0.85)
    runs(page, 357, 193.5, R("node content", "b", 6.6, 0), "c", BLUE_D)
    runs(page, 357, 203, M(6.0, ("x", "i"), ("u", "sub"), (": TF-IDF, SVD (300-d)", "r")),
         "c", INK)
    runs(page, 357, 212.5, R("content residual", None, 6.0, 0), "c", BLUE_D)
    runs(page, 357, 221.5, R("(epoch-zero anchor)", None, 6.0, 0), "c", BLUE_D)
    arrow(page, [(400, 184), (400, 78), (356.6, 78)], BLUE, w=0.95, head=4.2, hw=1.9)

    # region descriptor -> gate
    card(page, 6, 184, 112, 226, ORANGE, T_ORANGE, w=0.85)
    runs(page, 59, 193.5, R("region descriptor", "b", 6.6, 0), "c", ORANGE_D)
    runs(page, 59, 203, M(6.4, ("ρ", "i"), ("(u)", "r"), (" ∈ ", "r"), ("R", "i"), ("13", "sup")),
         "c", INK)
    runs(page, 59, 212.5, R("8 structural + 5 temporal", None, 6.0, 0), "c", INK)
    runs(page, 59, 221.5, R("on the fit graph + dates", None, 6.0, 0), "c", SUB)

    card(page, 124, 184, 300, 226, ORANGE, T_ORANGE, w=1.0, dash="[2.6 1.9] 0")
    runs(page, 212, 193.5, R("region-conditioned gate (proposed)", "b", 7.0, 0), "c", ORANGE_D)
    runs(page, 212, 204.5, M(6.6, ("g", "i"), ("d", "sub"), (" = 1 + tanh(", "r"), ("w", "i"),
                           ("d", "sub"), ("T", "sup"), (" ρ", "i"), ("(u) + ", "r"),
                           ("b", "i"), ("d", "sub"), (")", "r")), "c", INK)
    runs(page, 212, 215.5, M(6.0, ("linear layer, zero-init; ", "r"), ("g", "i"), ("d", "sub"),
                           (" = 1 at epoch zero", "r")), "c", INK)
    arrow(page, [(112, 205), (124, 205)], ORANGE, w=0.95, head=4.0, hw=1.8)


def draw_risers(page):
    """Gate strip -> the two (x) gates: clean vertical corridor, zero crossings."""
    for (x, y_to) in ((200, 61.6), (216, 117.6)):
        arrow(page, [(x, 184), (x, y_to)], ORANGE, w=1.15, head=4.6, hw=2.1)


def draw_legend(page):
    """Model cells as the gate x time factorial. The colour key sits inline in the
    title so the four rows keep a comfortable 10.5 pt pitch."""
    card(page, 6, 124, 100, 182, "#D6D6D6", "#FFFFFF", w=0.7, rad=3.0)
    runs(page, 10, 135, R("Model cells", "b", 6.0, 0), "l", SUB)
    x = 10 + tw("Model cells", 6.0, "b") + 6
    for lab, col in (("gate", ORANGE), ("time", RED)):
        page.draw_circle(fitz.Point(x + 1.6, 133.2), 1.6, color=None, fill=rgb(col))
        x += 4.6 + runs(page, x + 4.6, 135, R(lab, None, 5.6, 0), "l", SUB) + 6
    rows = ((GREY, "gat_dir", False, False),
            (ORANGE, "rcgat_sym", True, False),
            (GREY, "gat_time", False, True),
            (ORANGE, "rcgat_time", True, True))
    for i, (col, name, gt, tm) in enumerate(rows):
        y = 146 + i * 10.5
        runs(page, 10, y, R(name, "b", 6.1, 0), "l", INK)
        for (cx, on, c2) in ((64, gt, ORANGE), (88, tm, RED)):
            if on:
                page.draw_circle(fitz.Point(cx, y - 2.0), 2.0, color=None, fill=rgb(c2))
            else:
                page.draw_line(fitz.Point(cx - 2.1, y - 2.0), fitz.Point(cx + 2.1, y - 2.0),
                               color=rgb("#BFBFBF"), width=0.8, lineCap=1)


def build():
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    for k, v in FILES.items():
        page.insert_font(fontname="f_" + k, fontfile=FDIR + v)
    U = draw_graph(page)
    draw_encoder(page)
    draw_output(page)
    draw_bottom(page)
    draw_risers(page)
    draw_legend(page)
    # the two encoder branches leave the graph as one fork out of u
    from_ = u_edge(U[0], U[1], U[2], 110, 54, pad=1.4)
    arrow(page, [from_, (110, 54)], TEAL, w=1.05, head=4.6, hw=2.1)
    from2 = u_edge(U[0], U[1], U[2], 110, 110, pad=1.4)
    arrow(page, [from2, (110, 110)], TEAL, w=1.05, head=4.6, hw=2.1)
    page.clean_contents()
    return doc, page


def freeze_pdf_id(path):
    """PyMuPDF writes a random trailer /ID; freeze it to a content hash so a
    rebuild is byte-identical (the rest of the revision package is)."""
    import hashlib
    import re
    with open(path, "rb") as fh:
        b = fh.read()
    m = re.search(rb"/ID\s*\[<([0-9A-Fa-f]+)><([0-9A-Fa-f]+)>\]", b)
    if not m:
        return False
    blank = (b[:m.start(1)] + b"0" * (m.end(1) - m.start(1)) + b"><"
             + b"0" * (m.end(2) - m.start(2)) + b[m.end(2):])
    h = hashlib.sha256(blank).hexdigest().upper()
    out = (b[:m.start(1)] + h[:32].encode() + b"><" + h[32:].encode() + b[m.end(2):])
    assert len(out) == len(b)
    with open(path, "wb") as fh:
        fh.write(out)
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild the RC-GAT mechanism figure "
                                            "(vector PDF + editable SVG + preview PNG).")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),
                    help="output directory (default: the script's own folder)")
    ap.add_argument("--stem", default="fig_mechanism_redraw", help="output basename")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    doc, page = build()
    pdf = os.path.join(out, args.stem + ".pdf")
    doc.subset_fonts()
    doc.save(pdf, deflate=True, garbage=4)
    doc.close()
    freeze_pdf_id(pdf)      # deterministic build (see helper)
    # export from the file just written: garbage collection above invalidates the
    # in-memory page handle, and this also proves the shipped PDF is the source
    with fitz.open(pdf) as back:
        page = back[0]
        with open(os.path.join(out, args.stem + ".svg"), "w") as fh:
            fh.write(page.get_svg_image(text_as_path=False))
        pm = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    png = os.path.join(out, args.stem + ".png")
    pm.save(png)
    print("pdf  %.1f KB  %s" % (os.path.getsize(pdf) / 1024.0, pdf))
    print("svg  %.1f KB" % (os.path.getsize(os.path.join(out, args.stem + ".svg")) / 1024.0))
    print("png  %dx%d  %.1f KB" % (pm.width, pm.height, os.path.getsize(png) / 1024.0))
    print("page %.0f x %.0f pt (aspect %.2f:1)" % (W, H, W / H))
