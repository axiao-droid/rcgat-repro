# -*- coding: utf-8 -*-
"""
Occlusion audit -- evidence for "no occlusion", instead of eyeballing.

Runs on any PDF. For every word it builds a tight *ink* box from the glyph
baseline (not the em box, which would flag innocuous line spacing), then checks:

  1. text vs text      : no two ink boxes overlap
  2. line vs ink       : no drawn segment (arrow shaft, gate riser, connector,
                         node edge, frame border) passes through a word's ink
  3. text vs frame     : no word straddles a card border  <- the original
                         figure's failure mode (gate arrow tip over the 'out'
                         subscript, and the label edit that started this task)

Also reports the closest approach between any drawn segment and any word ink,
which is the quantitative "clearance" claim.

usage: python3 check_fig_occlusion.py fig.pdf [more.pdf ...]
"""
import json
import sys

import fitz

ASC, DESC = 0.80, 0.24      # ink extents as a fraction of the font size
PAD = 0.15                  # pt of shrink so touching boxes are not overlaps


def ink_box(sp):
    """Tight glyph box: baseline +/- ink extents, span width in x."""
    r = fitz.Rect(sp["bbox"])
    base = sp.get("origin", (r.x0, r.y1))[1]
    sz = sp["size"]
    return fitz.Rect(r.x0 + PAD, base - ASC * sz,
                     r.x1 - PAD, base + DESC * sz)


def seg_rect_hit(p, q, r):
    """Segment p-q vs rect r (Liang-Barsky clipping)."""
    dx, dy = q.x - p.x, q.y - p.y
    t0, t1 = 0.0, 1.0
    for pp, off in ((-dx, p.x - r.x0), (dx, r.x1 - p.x),
                    (-dy, p.y - r.y0), (dy, r.y1 - p.y)):
        if abs(pp) < 1e-9:
            if off < 0:
                return False
        else:
            t = off / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def seg_rect_dist(p, q, r):
    """Approximate distance from segment p-q to rect r (0 if intersecting)."""
    if seg_rect_hit(p, q, r):
        return 0.0

    def pt_seg2(pt, a, b):
        vx, vy = b.x - a.x, b.y - a.y
        wx, wy = pt.x - a.x, pt.y - a.y
        L = vx * vx + vy * vy or 1.0
        tt = max(0.0, min(1.0, (wx * vx + wy * vy) / L))
        return ((a.x + tt * vx) - pt.x) ** 2 + ((a.y + tt * vy) - pt.y) ** 2

    corners = [r.tl, r.tr, r.br, r.bl]
    edges = [(r.tl, r.tr), (r.tr, r.br), (r.br, r.bl), (r.bl, r.tl)]
    best = min(pt_seg2(pt, cn, cn2) for pt in (p, q)
               for cn in corners for cn2 in corners)     # corner-to-corner proxy
    for (e0, e1) in edges:
        best = min(best, pt_seg2(p, e0, e1), pt_seg2(q, e0, e1))
    for cn in corners:
        best = min(best, pt_seg2(cn, p, q))
    return best ** 0.5


def audit(pdf, verbose=True):
    doc = fitz.open(pdf)
    page = doc[0]
    spans, segs, frames = [], [], []
    for blk in page.get_text("dict")["blocks"]:
        for ln in blk.get("lines", []):
            for sp in ln["spans"]:
                if sp["text"].strip():
                    spans.append({"t": sp["text"], "size": sp["size"], "box": ink_box(sp),
                                  "em": fitz.Rect(sp["bbox"])})
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                segs.append((it[1], it[2]))
            elif it[0] == "re":
                r = fitz.Rect(it[1])
                for a, b in ((r.tl, r.tr), (r.tr, r.br), (r.br, r.bl), (r.bl, r.tl)):
                    segs.append((a, b))
        # a filled, card-sized drawing is a container: its outline counts as a
        # border (rounded corners arrive as curves, so the bbox stands in for it)
        r = fitz.Rect(d["rect"])
        if d.get("fill") is not None and r.width > 40 and r.height > 16:
            frames.append(r)
            for a, b in ((r.tl, r.tr), (r.tr, r.br), (r.br, r.bl), (r.bl, r.tl)):
                segs.append((a, b))
    segs = [(p, q) for (p, q) in segs
            if (abs(p.x - q.x) + abs(p.y - q.y)) > 0.1]

    tt = [(a["t"], b["t"]) for i, a in enumerate(spans) for b in spans[i + 1:]
          if a["box"].intersects(b["box"])
          and (a["box"] & b["box"]).width > PAD and (a["box"] & b["box"]).height > PAD]
    lt, mind = [], 1e9
    for (p, q) in segs:
        for sp in spans:
            d = seg_rect_dist(p, q, sp["box"])
            if d <= 0.0:
                lt.append((sp["t"], [round(p.x, 1), round(p.y, 1)],
                           [round(q.x, 1), round(q.y, 1)]))
            mind = min(mind, d)
    cross = [sp["t"] for sp in spans
             if any((not f.contains(sp["em"])) and (f & sp["em"]).width > 1.5
                    and (f & sp["em"]).height > 1.5 for f in frames)]

    rep = {"pdf": pdf.split("/")[-1], "size_pt": [round(page.rect.width, 1),
                                                 round(page.rect.height, 1)],
           "words": len(spans), "segments": len(segs), "frames": len(frames),
           "text_text_overlaps": len(tt), "line_ink_hits": len(lt),
           "words_crossing_a_border": len(cross),
           "min_line_to_ink_clearance_pt": round(mind, 2)}
    rep["ok"] = not (tt or lt or cross)
    if verbose:
        print(json.dumps(rep, ensure_ascii=False))
        for a, b in tt[:4]:
            print("   overlap:", a, "<->", b)
        for t, p, q in lt[:6]:
            print("   line through ink:", repr(t), p, q)
        for t in cross[:6]:
            print("   crosses border:", repr(t))
    return rep


if __name__ == "__main__":
    ok = True
    for f in sys.argv[1:]:
        r = audit(f)
        print("   -->", "PASS" if r["ok"] else "FAIL", f, "\n")
        ok = ok and r["ok"]
    sys.exit(0 if ok else 1)
