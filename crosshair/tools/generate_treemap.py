"""
Render the measured CrossHair support corpus as an SVG.

Two modes, both nested by builtin type / ``builtins`` functions / stdlib module,
each cell colored by its support result (green/yellow/red/grey):

  * default -- a fixed-size grid (every operation the same size).
  * ``--weights usage.json`` -- a single area-weighted treemap, each cell sized by
    how widely the operation is used (from ``mine_usage``); rarely-used ops shrink
    away.  This is what the docs ship.

Cells carry a plain-English hover and a click-through to a runnable crosshair-web
demo.  Reads the merged JSON emitted by ``measure_support surface``/``funcs``.

    python -m crosshair.tools.generate_treemap \\
        --measured surface.json,funcs.json --weights usage.json \\
        --out doc/source/support_treemap.svg
"""

import argparse
import json
import math
from pathlib import Path
from urllib.parse import quote_plus
from xml.sax.saxutils import escape

from crosshair.tools import measure_support as R

CH_WEB = "https://crosshair-web.org"

# verdict -> (fill, label) -- distinct, colorblind-ish, matches the emoji palette
COLORS = {
    "green": "#3fa650",
    "yellow": "#e8b400",
    "red": "#d5503a",
    "black": "#222222",
    "?": "#c9ced6",
    "grey": "#c9ced6",
}
LEGEND = [
    ("green", "CrossHair handles it well"),
    ("yellow", "works for small inputs, slower as they grow"),
    ("red", "CrossHair struggles here"),
    ("black", "gives a wrong answer (bug)"),
    ("?", "not measured"),
]
# plain-English hover, aimed at someone meeting CrossHair for the first time
HOVER = {
    "green": "CrossHair can usually find an input that makes this produce a given result.",
    "yellow": "CrossHair can work this out for small inputs, but gets slow as inputs grow.",
    "red": "CrossHair has trouble reasoning about this one.",
    "black": "CrossHair gives a wrong answer here (a soundness bug).",
    "?": "Not measured here.",
    "grey": "Not measured here.",
}

CELL = 11  # leaf square edge (px)
GAP = 2  # gap between cells
PAD = 5  # padding inside a group box
HEADER = 14  # group label height
TIERGAP = 10  # vertical gap between tiers
TIER_H = 22  # tier label band height
MARGIN = 12  # canvas margin
WIDTH = 1180  # content width (groups shelf-pack to this)
ASPECT = 1.7  # group grid wider than tall (cols vs rows bias)
FONT = "font-family='-apple-system,Segoe UI,Roboto,sans-serif'"


def _color(rec):
    if rec and rec.get("color") in COLORS:
        return COLORS[rec["color"]]
    return COLORS["grey"]


def _group_layout(n):
    """(cols, rows) for n equal cells, biased a bit wide."""
    cols = max(1, round(math.sqrt(n * ASPECT)))
    rows = math.ceil(n / cols)
    return cols, rows


def _group_size(n):
    cols, rows = _group_layout(n)
    w = PAD * 2 + cols * CELL + (cols - 1) * GAP
    h = HEADER + PAD + rows * CELL + (rows - 1) * GAP + PAD
    return max(w, 46), h  # floor width so the label fits


def _collect(measured):
    """[(tier, [(group, [(op, rec), ...]), ...]), ...] in display order."""
    tiers = []

    def leaf(op, key):  # (op name, cell key, measured record)
        return (op, key, measured.get(key))

    types = [
        (
            t.__name__,
            [leaf(m, f"builtins.{t.__name__}_{m}_method") for m in R.surface(t)],
        )
        for t in R.TYPES
    ]
    tiers.append(("builtin types", types))

    bfuncs = R.func_surface("builtins")
    if bfuncs:
        tiers.append(
            (
                "builtin functions",
                [("builtins", [leaf(f, f"builtins.{f}") for f in bfuncs])],
            )
        )

    mods = sorted(
        {
            k.rsplit(".", 1)[0]
            for k in measured
            if "." in k and not k.startswith("builtins.")
        }
    )
    stdlib = []
    for mod in mods:
        leaves = [leaf(f, f"{mod}.{f}") for f in R.func_surface(mod)]
        if leaves:
            stdlib.append((mod, leaves))
    if stdlib:
        tiers.append(("standard library", stdlib))
    return tiers


def _shelf_pack(groups, width):
    """Place group boxes left-to-right, wrapping; return (placements, total_h).
    placements: [(group, leaves, x, y, w, h), ...] with y relative to tier start."""
    placements, x, y, shelf_h = [], 0, 0, 0
    for group, leaves in groups:
        w, h = _group_size(len(leaves))
        if x > 0 and x + w > width:  # wrap
            y += shelf_h + GAP * 2
            x, shelf_h = 0, 0
        placements.append((group, leaves, x, y, w, h))
        x += w + GAP * 2
        shelf_h = max(shelf_h, h)
    return placements, y + shelf_h


def render(measured):
    tiers = _collect(measured)
    counts = {"green": 0, "yellow": 0, "red": 0, "black": 0, "?": 0, "grey": 0}
    for _t, groups in tiers:
        for _g, leaves in groups:
            for _op, _key, rec in leaves:
                counts[
                    rec["color"] if rec and rec.get("color") in COLORS else "grey"
                ] += 1

    svg = []
    y = MARGIN
    # title + legend
    svg.append(
        f"<text x='{MARGIN}' y='{y + 12}' {FONT} font-size='15' font-weight='bold'>"
        f"CrossHair symbolic-reasoning support</text>"
    )
    y += 24
    lx = MARGIN
    for key, desc in LEGEND:
        svg.append(
            f"<rect x='{lx}' y='{y}' width='11' height='11' rx='2' fill='{COLORS[key]}'/>"
        )
        svg.append(
            f"<text x='{lx + 16}' y='{y + 10}' {FONT} font-size='11' fill='#333'>{escape(desc)}</text>"
        )
        lx += 20 + len(desc) * 6.3 + 16
    y += 22

    body = []
    for tier, groups in tiers:
        n = sum(len(lv) for _g, lv in groups)
        body.append(
            f"<text x='{MARGIN}' y='{y + 15}' {FONT} font-size='13' font-weight='bold' "
            f"fill='#222'>{escape(tier)} <tspan fill='#888' font-weight='normal'>({n})</tspan></text>"
        )
        y += TIER_H
        placements, tier_h = _shelf_pack(groups, WIDTH)
        for group, leaves, gx, gy, gw, gh in placements:
            ox, oy = MARGIN + gx, y + gy
            body.append(
                f"<rect x='{ox}' y='{oy}' width='{gw}' height='{gh}' rx='3' "
                f"fill='#f7f8fa' stroke='#dfe2e7'/>"
            )
            body.append(
                f"<text x='{ox + PAD}' y='{oy + 10}' {FONT} font-size='10' fill='#444'>"
                f"{escape(group)} <tspan fill='#999'>{len(leaves)}</tspan></text>"
            )
            cols, _rows = _group_layout(len(leaves))
            for i, (op, _key, rec) in enumerate(leaves):
                cx = ox + PAD + (i % cols) * (CELL + GAP)
                cy = oy + HEADER + PAD + (i // cols) * (CELL + GAP)
                example = rec.get("example") if rec else None
                key = (
                    rec["color"]
                    if (rec and rec.get("color") in ("green", "yellow", "red", "black"))
                    else "?"
                )
                hint = HOVER[key] + (" Click to try it live." if example else "")
                title = escape(f"{group}.{op}() — {hint}")
                cell = (
                    f"<rect x='{cx}' y='{cy}' width='{CELL}' height='{CELL}' rx='1.5' "
                    f"fill='{_color(rec)}'><title>{title}</title></rect>"
                )
                if (
                    example
                ):  # clickable: run this op's auto-generated demo on crosshair-web
                    url = escape(
                        f"{CH_WEB}/?source={quote_plus(example)}", {'"': "&quot;"}
                    )
                    cell = (
                        f"<a href=\"{url}\" target='_blank' rel='noopener'>{cell}</a>"
                    )
                body.append(cell)
        y += tier_h + TIERGAP

    total_h = y + MARGIN
    full_w = WIDTH + 2 * MARGIN
    # responsive: scale to the container width (viewBox keeps aspect), capped at
    # native size; rendered inline so the per-cell <title> hovers stay live.
    head = (
        f"<svg xmlns='http://www.w3.org/2000/svg' "
        f"width='100%' viewBox='0 0 {full_w} {total_h}' "
        f"style='max-width:{full_w}px;height:auto;font-size:11px'>"
    )
    bg = f"<rect width='100%' height='100%' fill='white'/>"
    return head + bg + "".join(svg) + "".join(body) + "</svg>", counts


# ---------------------------------------------------------------------------
# weighted (area-proportional) treemap -- squarified, sized by a usage prior
# ---------------------------------------------------------------------------
def _squarify(sizes, x, y, dx, dy):
    """Canonical squarified treemap (Bruls et al.); returns rects {x,y,dx,dy} in
    input order.  ``sizes`` must already sum to dx*dy (use _normalize)."""
    sizes = [float(s) for s in sizes]
    if not sizes:
        return []
    if len(sizes) == 1:
        return _layout(sizes, x, y, dx, dy)
    i = 1
    while i < len(sizes) and _worst(sizes[:i], dx, dy) >= _worst(
        sizes[: i + 1], dx, dy
    ):
        i += 1
    cur, rest = sizes[:i], sizes[i:]
    lx, ly, ldx, ldy = _leftover(cur, x, y, dx, dy)
    return _layout(cur, x, y, dx, dy) + _squarify(rest, lx, ly, ldx, ldy)


def _layout(sizes, x, y, dx, dy):
    if dx >= dy:
        w = sum(sizes) / dy if dy else 0
        out, yy = [], y
        for s in sizes:
            h = s / w if w else 0
            out.append({"x": x, "y": yy, "dx": w, "dy": h})
            yy += h
        return out
    h = sum(sizes) / dx if dx else 0
    out, xx = [], x
    for s in sizes:
        w = s / h if h else 0
        out.append({"x": xx, "y": y, "dx": w, "dy": h})
        xx += w
    return out


def _worst(sizes, dx, dy):
    rects = _layout(sizes, 0, 0, dx, dy)
    return max(
        max(r["dx"] / r["dy"], r["dy"] / r["dx"]) for r in rects if r["dx"] and r["dy"]
    )


def _leftover(sizes, x, y, dx, dy):
    if dx >= dy:
        w = sum(sizes) / dy if dy else 0
        return x + w, y, dx - w, dy
    h = sum(sizes) / dx if dx else 0
    return x, y + h, dx, dy - h


def _normalize(sizes, dx, dy):
    total = sum(sizes)
    if total <= 0:
        return [0.0] * len(sizes)
    area = dx * dy
    return [s * area / total for s in sizes]


def _cell_svg(op, group, rec, lr):
    """One leaf cell: colored rect + op-name label (if it fits) + hover + link."""
    if lr["dx"] < 1 or lr["dy"] < 1:
        return ""
    ex = rec.get("example") if rec else None
    ck = (
        rec["color"]
        if (rec and rec.get("color") in ("green", "yellow", "red", "black"))
        else "?"
    )
    hint = HOVER[ck] + (" Click to try it live." if ex else "")
    title = escape(f"{group}.{op}() — {hint}")
    inner = (
        f"<rect x='{lr['x']:.1f}' y='{lr['y']:.1f}' width='{lr['dx']:.1f}' "
        f"height='{lr['dy']:.1f}' fill='{_color(rec)}' stroke='white' "
        f"stroke-width='0.5'><title>{title}</title></rect>"
    )
    if lr["dx"] > 26 and lr["dy"] > 9:  # room for an op-name label
        label = op.strip("_") if op.startswith("__") else op
        if len(label) * 5.3 > lr["dx"] - 3:
            label = label[: max(1, int((lr["dx"] - 3) / 5.3))]
        # white label on the dark "black" (buggy) cells; near-black elsewhere
        text_fill = "#ffffff" if (rec and rec.get("color") == "black") else "#1a1a1a"
        inner += (
            f"<text x='{lr['x'] + 2:.1f}' y='{lr['y'] + 9:.1f}' {FONT} font-size='8' "
            f"fill='{text_fill}'>{escape(label)}</text>"
        )
    if ex:
        url = escape(f"{CH_WEB}/?source={quote_plus(ex)}", {'"': "&quot;"})
        return f"<a href=\"{url}\" target='_blank' rel='noopener'>{inner}</a>"
    return inner


def _draw_leaves(leaves, x, y, w, h, body):
    rects = _squarify(_normalize([lf[3] for lf in leaves], w, h), x, y, w, h)
    for (op, _key, rec, _wt, group), lr in zip(leaves, rects):
        body.append(_cell_svg(op, group, rec, lr))


def _inset(r, p):
    return {
        "x": r["x"] + p,
        "y": r["y"] + p,
        "dx": max(0.0, r["dx"] - 2 * p),
        "dy": max(0.0, r["dy"] - 2 * p),
    }


# spacing for the weighted treemap (bigger margins -> the grouping reads clearer)
_MOD_GAP = 4  # gap between sibling boxes
_MOD_PAD = 5  # interior padding inside a box (below its label)


_SCALE = {
    "linear": lambda x: x,
    "sqrt": math.sqrt,
    "log": math.log1p,  # log1p so a 1-package op still has nonzero area
}


def render_weighted(
    measured, weights, metric="packages", min_weight=1.0, scale="linear"
):
    """Single area-weighted treemap: every cell's area is proportional to its usage
    prior.  Top-level boxes are each builtin TYPE, the builtin ``builtins``
    functions, and each stdlib module -- all siblings (so no single ``builtins``
    box dwarfs the standard library, which is ~12% of all usage).  Ops used by
    fewer than ``min_weight`` packages (after type-apportionment) are dropped, so
    the long tail shrinks away.  Cells keep their support color, an op-name label
    where it fits, a hover, and a click-through to a runnable demo.

    ``scale`` shapes area vs. usage: ``linear`` (area == packages, the honest
    "what matters most" view) or ``sqrt``/``log`` to compress the huge head so the
    long tail of rarer ops stays legible.  The ``min_weight`` cutoff always
    applies to the RAW package count, not the scaled area."""
    transform = _SCALE[scale]

    def raw_wt(key):
        return float((weights.get(key) or {}).get(metric, 0.0))

    def wt(key):  # area-driving weight (possibly compressed)
        return transform(raw_wt(key))

    by_tier = {t: g for t, g in _collect(measured)}

    def leaves_of(groupname, raw):  # leaf = (op, key, rec, weight, group_for_hover)
        return [
            (op, key, rec, wt(key), groupname)
            for op, key, rec in raw
            if raw_wt(key) >= min_weight
        ]

    modules = []  # (name, leaves, total_wt) -- one flat level
    for typ, raw in by_tier.get("builtin types", []):
        lv = leaves_of(typ, raw)
        if lv:
            modules.append((typ, lv, sum(lf[3] for lf in lv)))
    for _g, raw in by_tier.get("builtin functions", []):
        lv = leaves_of("builtins", raw)
        if lv:
            modules.append(("builtins", lv, sum(lf[3] for lf in lv)))
    for mod, raw in by_tier.get("standard library", []):
        lv = leaves_of(mod, raw)
        if lv:
            modules.append((mod, lv, sum(lf[3] for lf in lv)))
    modules.sort(key=lambda m: m[2], reverse=True)

    W, H = 960, 1120  # a little taller than wide
    parts = [
        f"<text x='{MARGIN}' y='{MARGIN + 12}' {FONT} font-size='15' font-weight='bold'>"
        f"CrossHair support × real-world usage</text>"
    ]
    lx = MARGIN
    # compact one-row legend (the full descriptions live in the page prose)
    wlegend = [
        ("green", "handles it well"),
        ("yellow", "slower as inputs grow"),
        ("red", "struggles"),
        ("black", "wrong answer (bug)"),
        ("?", f"box area = packages using it ({metric})"),
    ]
    for key, desc in wlegend:
        parts.append(
            f"<rect x='{lx}' y='{MARGIN + 22}' width='11' height='11' rx='2' fill='{COLORS[key]}'/>"
        )
        parts.append(
            f"<text x='{lx + 16}' y='{MARGIN + 32}' {FONT} font-size='11' fill='#333'>{escape(desc)}</text>"
        )
        lx += 20 + len(desc) * 6.3 + 16
    top = MARGIN + 44

    body = []
    mrects = _squarify(
        _normalize([m[2] for m in modules], W, H - top - MARGIN),
        MARGIN,
        top,
        W,
        H - top - MARGIN,
    )
    for (name, leaves, _w), mr0 in zip(modules, mrects):
        mr = _inset(mr0, _MOD_GAP / 2)  # gap between boxes
        if mr["dx"] < 4 or mr["dy"] < 4:
            continue
        body.append(
            f"<rect x='{mr['x']:.1f}' y='{mr['y']:.1f}' width='{mr['dx']:.1f}' "
            f"height='{mr['dy']:.1f}' rx='2' fill='#e7eaee' stroke='#aab2bd'/>"
        )
        head_h = 13 if (mr["dy"] > 30 and mr["dx"] > 26) else 0
        if head_h:
            body.append(
                f"<text x='{mr['x'] + _MOD_PAD:.1f}' y='{mr['y'] + 10:.1f}' {FONT} font-size='9.5' "
                f"font-weight='bold' fill='#2a2f36'>{escape(name)}</text>"
            )
        ix, iy = mr["x"] + _MOD_PAD, mr["y"] + head_h
        iw, ih = mr["dx"] - 2 * _MOD_PAD, mr["dy"] - head_h - _MOD_PAD
        if iw >= 2 and ih >= 2:
            _draw_leaves(leaves, ix, iy, iw, ih, body)

    full_w = W + 2 * MARGIN
    head = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='100%' "
        f"viewBox='0 0 {full_w} {H}' style='max-width:{full_w}px;height:auto;font-size:11px'>"
    )
    return (
        head
        + "<rect width='100%' height='100%' fill='white'/>"
        + "".join(parts)
        + "".join(body)
        + "</svg>"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Render the support corpus as an SVG treemap."
    )
    ap.add_argument(
        "--measured", required=True, help="comma-separated measured JSON files (merged)"
    )
    ap.add_argument("--out", required=True, help="output .svg path")
    ap.add_argument(
        "--weights",
        help="usage JSON from mine_usage; if given, render an "
        "area-weighted treemap instead of the fixed-size grid",
    )
    ap.add_argument(
        "--metric",
        default="packages",
        choices=["packages", "sites"],
        help="usage metric to size cells by (default: packages)",
    )
    ap.add_argument(
        "--min-weight",
        type=float,
        default=1.0,
        dest="min_weight",
        help="drop ops whose weight is below this (default 1.0 = <1 package)",
    )
    ap.add_argument(
        "--scale",
        default="linear",
        choices=["linear", "sqrt", "log"],
        help="area vs. usage: linear (honest 'what matters most') or "
        "sqrt/log to compress the head so the long tail stays legible",
    )
    args = ap.parse_args()
    measured = {}
    for p in args.measured.split(","):
        measured.update(json.loads(Path(p.strip()).read_text()))
    if args.weights:
        weights = json.loads(Path(args.weights).read_text())
        Path(args.out).write_text(
            render_weighted(measured, weights, args.metric, args.min_weight, args.scale)
            + "\n"
        )
        print(
            f"wrote {args.out}  (area-weighted by {args.metric}, "
            f"scale={args.scale}, min_weight={args.min_weight})"
        )
        return
    svg, counts = render(measured)
    Path(args.out).write_text(svg + "\n")
    print(
        f"wrote {args.out}  ({sum(counts.values())} cells: "
        + " ".join(f"{k}={v}" for k, v in counts.items())
        + ")"
    )


if __name__ == "__main__":
    main()
