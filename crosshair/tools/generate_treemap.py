"""
Render the measured CrossHair support corpus as an SVG.

Two modes, both nested by builtin type / ``builtins`` functions / stdlib module,
each cell colored by its support result (green/yellow/red/black/grey):

  * default -- a fixed-size grid (every operation the same size).
  * ``--weights usage.json`` -- a single area-weighted treemap, each cell sized by
    how widely the operation is used (from ``mine_usage``); rarely-used ops shrink
    away.  This is what the docs ship.

Cells carry a plain-English hover and a click-through to a runnable crosshair-web
demo.  ``--measured`` takes one or more JSON files from ``measure_support measure``
(comma-separated; later files win on key collisions).

================================================================================
Regenerating the shipped map  (doc/source/support_treemap.svg)
================================================================================
The map joins two JSON inputs: a MEASUREMENT (how well CrossHair reasons about
each op -- from ``measure_support``) and a USAGE PRIOR (how widely each op is
used -- from ``mine_usage``).  The usage prior (``doc/source/usage_prior.json``)
is checked in and changes rarely, so the everyday refresh is just steps 2 + 4:

    # 1. (occasional) fetch a corpus of top-PyPI wheels -- .py text only, no exec
    python -m crosshair.tools.fetch_corpus --n 200 --out /tmp/pypi_corpus

    # 2. measure the whole operation catalog -> measured.json
    #    ~22k ops, but ~19k are statically classified (out-of-scope / no inputs /
    #    side effect / probe hazard) and return instantly; only ~3k get a real
    #    symbolic sweep.  Keep --jobs at or below your core count (each worker
    #    pins a core with a z3 solve).  Run it from a SCRATCH dir: the fuzzer
    #    writes a .hypothesis/ cache into the cwd.
    python -m crosshair.tools.measure_support measure --jobs 8 --json measured.json

    # 3. (occasional) re-mine the usage prior from the corpus.  Keyed to the shared
    #    operation catalog -- INDEPENDENT of measurement, so this can run before,
    #    after, or in parallel with step 2; both artifacts join by construction.
    python -m crosshair.tools.mine_usage \\
        --corpus /tmp/pypi_corpus --out doc/source/usage_prior.json

    # 4. render the shipped SVG: full catalog, area-weighted by package usage.
    #    The committed map uses the DEFAULTS below -- do not pass --metric/--scale/
    #    --min-weight unless you intend to change what ships.
    python -m crosshair.tools.generate_treemap --measured measured.json \\
        --weights doc/source/usage_prior.json --out doc/source/support_treemap.svg

The committed map is the area-weighted treemap with ``--metric packages
--scale linear --min-weight 1.0`` (the defaults): each cell's area is the number
of packages that use the op, and ops used by fewer than one package are dropped.
Omit ``--weights`` entirely for the fixed-size grid instead of the treemap.

Scoped re-measurement (step 2) -- for re-running one wedged module or a single
tier in its own killable invocation, then merging the JSONs at step 4:
    python -m crosshair.tools.measure_support measure --modules math,json --json m1.json
    python -m crosshair.tools.measure_support measure --tiers builtin-methods,functions ...
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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


def _color(rec: Optional[Dict[str, Any]]) -> str:
    if rec and rec.get("color") in COLORS:
        return COLORS[rec["color"]]
    return COLORS["?"]


def _cell_title(
    group: str, op: str, rec: Optional[Dict[str, Any]], example: Optional[str]
) -> str:
    """Escaped <title> hover for a cell: plain-English verdict (unmeasured -> '?')
    plus a click-to-run nudge when a runnable demo exists."""
    key = (
        rec["color"]
        if (rec and rec.get("color") in ("green", "yellow", "red", "black"))
        else "?"
    )
    hint = HOVER[key] + (" Click to try it live." if example else "")
    return escape(f"{group}.{op}() — {hint}")


def _maybe_link(inner: str, example: Optional[str]) -> str:
    """Wrap a cell's SVG in a crosshair-web demo link when an example exists."""
    if not example:
        return inner
    url = escape(f"{CH_WEB}/?source={quote_plus(example)}", {'"': "&quot;"})
    return f"<a href=\"{url}\" target='_blank' rel='noopener'>{inner}</a>"


def _group_layout(n: int) -> Tuple[int, int]:
    """(cols, rows) for n equal cells, biased a bit wide."""
    cols = max(1, round(math.sqrt(n * ASPECT)))
    rows = math.ceil(n / cols)
    return cols, rows


def _group_size(n: int) -> Tuple[int, int]:
    cols, rows = _group_layout(n)
    w = PAD * 2 + cols * CELL + (cols - 1) * GAP
    h = HEADER + PAD + rows * CELL + (rows - 1) * GAP + PAD
    return max(w, 46), h  # floor width so the label fits


def _collect(
    measured: Dict[str, Any],
) -> List[
    Tuple[str, List[Tuple[str, List[Tuple[str, str, Optional[Dict[str, Any]]]]]]]
]:
    """[(tier, [(group, [(op, key, rec), ...]), ...]), ...] in display order.

    Built straight from the shared operation catalog (``crosshair.inputgen.catalog``,
    via measure_support) -- the SAME surface measure_support measures and mine_usage
    weights.  No re-derived surface here, so the map can't drift from the catalog:
    every catalogued op gets a cell (builtin ``complex``/``memoryview``/exception
    classes included), and its support record is looked up from ``measured``."""

    def leaf(op: str, key: str) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        return (op, key, measured.get(key))  # (op name, cell key, record)

    # Four tiers, mirroring the catalog's own method/func x builtins/stdlib split.
    builtin_types: Dict[str, list] = {}  # class name -> [leaf]
    builtin_funcs: list = []  # [leaf]
    stdlib_funcs: Dict[str, list] = {}  # module -> [leaf]
    stdlib_methods: Dict[str, list] = {}  # "module.Class" -> [leaf]
    for op in R.catalog(probe=False):
        lf = leaf(op.name, op.key)
        if op.module == "builtins":
            if op.kind == "method":
                builtin_types.setdefault(op.owner, []).append(lf)
            else:
                builtin_funcs.append(lf)
        elif op.kind == "method":
            stdlib_methods.setdefault(f"{op.module}.{op.owner}", []).append(lf)
        else:
            stdlib_funcs.setdefault(op.module, []).append(lf)

    tiers: list = []
    if builtin_types:
        tiers.append(
            ("builtin types", [(g, builtin_types[g]) for g in sorted(builtin_types)])
        )
    if builtin_funcs:
        tiers.append(("builtin functions", [("builtins", builtin_funcs)]))
    if stdlib_funcs:
        tiers.append(
            ("standard library", [(m, stdlib_funcs[m]) for m in sorted(stdlib_funcs)])
        )
    if stdlib_methods:
        tiers.append(
            (
                "standard library methods",
                [(g, stdlib_methods[g]) for g in sorted(stdlib_methods)],
            )
        )
    return tiers


def _shelf_pack(
    groups: List[Tuple[str, Any]], width: int
) -> Tuple[List[Tuple[str, Any, int, int, int, int]], int]:
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


def render(measured: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    tiers = _collect(measured)
    counts = {"green": 0, "yellow": 0, "red": 0, "black": 0, "?": 0}
    for _t, groups in tiers:
        for _g, leaves in groups:
            for _op, _key, rec in leaves:
                counts[rec["color"] if rec and rec.get("color") in COLORS else "?"] += 1

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
                title = _cell_title(group, op, rec, example)
                cell = (
                    f"<rect x='{cx}' y='{cy}' width='{CELL}' height='{CELL}' rx='1.5' "
                    f"fill='{_color(rec)}'><title>{title}</title></rect>"
                )
                body.append(_maybe_link(cell, example))
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
def _squarify(
    sizes: List[float], x: float, y: float, dx: float, dy: float
) -> List[Dict[str, float]]:
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


def _layout(
    sizes: List[float], x: float, y: float, dx: float, dy: float
) -> List[Dict[str, float]]:
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


def _worst(sizes: List[float], dx: float, dy: float) -> float:
    rects = _layout(sizes, 0, 0, dx, dy)
    return max(
        max(r["dx"] / r["dy"], r["dy"] / r["dx"]) for r in rects if r["dx"] and r["dy"]
    )


def _leftover(
    sizes: List[float], x: float, y: float, dx: float, dy: float
) -> Tuple[float, float, float, float]:
    if dx >= dy:
        w = sum(sizes) / dy if dy else 0
        return x + w, y, dx - w, dy
    h = sum(sizes) / dx if dx else 0
    return x, y + h, dx, dy - h


def _normalize(sizes: List[float], dx: float, dy: float) -> List[float]:
    total = sum(sizes)
    if total <= 0:
        return [0.0] * len(sizes)
    area = dx * dy
    return [s * area / total for s in sizes]


def _cell_svg(
    op: str, group: str, rec: Optional[Dict[str, Any]], lr: Dict[str, float]
) -> str:
    """One leaf cell: colored rect + op-name label (if it fits) + hover + link."""
    if lr["dx"] < 1 or lr["dy"] < 1:
        return ""
    ex = rec.get("example") if rec else None
    title = _cell_title(group, op, rec, ex)
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
    return _maybe_link(inner, ex)


def _draw_leaves(
    leaves: List[Tuple[str, str, Optional[Dict[str, Any]], float, str]],
    x: float,
    y: float,
    w: float,
    h: float,
    body: List[str],
) -> None:
    rects = _squarify(_normalize([lf[3] for lf in leaves], w, h), x, y, w, h)
    for (op, _key, rec, _wt, group), lr in zip(leaves, rects):
        body.append(_cell_svg(op, group, rec, lr))


def _inset(r: Dict[str, float], p: float) -> Dict[str, float]:
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
    measured: Dict[str, Any],
    weights: Dict[str, Any],
    metric: str = "packages",
    min_weight: float = 1.0,
    scale: str = "linear",
) -> Tuple[str, List[str]]:
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
    applies to the RAW package count, not the scaled area.

    Returns ``(svg, missing)`` where ``missing`` is the sorted keys the usage prior
    wants drawn but has no support result for (grey cells / stale usage keys) -- the
    caller warns on these so a stale or under-measured JSON doesn't pass silently."""
    transform = _SCALE[scale]

    def raw_wt(key: str) -> float:
        return float((weights.get(key) or {}).get(metric, 0.0))

    def wt(key: str) -> float:  # area-driving weight (possibly compressed)
        return transform(raw_wt(key))

    by_tier = {t: g for t, g in _collect(measured)}

    def leaves_of(
        groupname: str, raw: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> List[
        Tuple[str, str, Optional[Dict[str, Any]], float, str]
    ]:  # leaf = (op, key, rec, weight, group_for_hover)
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
    for group, raw in by_tier.get("standard library methods", []):
        lv = leaves_of(group, raw)
        if lv:
            modules.append((group, lv, sum(lf[3] for lf in lv)))
    modules.sort(key=lambda m: m[2], reverse=True)

    # Cells the usage prior wants drawn but for which we have NO support result --
    # they render as indistinguishable grey "?".  Two flavors: a cell that IS drawn
    # but has no measured record (grey), and a usage key above the cutoff that never
    # became a cell at all (stale key: op dropped from the catalog surface).  Both
    # mean "usage says this matters, but the map can't say how CrossHair does on it".
    drawn_keys = {lf[1] for _n, lv, _w in modules for lf in lv}
    missing = sorted(
        {lf[1] for _n, lv, _w in modules for lf in lv if lf[2] is None}
        | {k for k in weights if raw_wt(k) >= min_weight and k not in drawn_keys}
    )

    W, H = 960, 1120  # a little taller than wide
    parts = [
        f"<text x='{MARGIN}' y='{MARGIN + 12}' {FONT} font-size='15' font-weight='bold'>"
        f"CrossHair support × real-world usage</text>"
    ]
    lx = MARGIN
    # compact one-row legend (the full descriptions live in the page prose).
    # The swatches are the support VERDICT colors only -- grey is the unmeasured
    # "?" cells.  Area is a separate SIZE channel, so it gets a trailing note
    # rather than a grey swatch that collides with the "not measured" grey.
    wlegend = [
        ("green", "handles it well"),
        ("yellow", "slower as inputs grow"),
        ("red", "struggles"),
        ("black", "wrong answer (bug)"),
        ("?", "not measured"),
    ]
    for key, desc in wlegend:
        # a hairline keeps the pale grey/black swatches visible against white
        parts.append(
            f"<rect x='{lx}' y='{MARGIN + 22}' width='11' height='11' rx='2' "
            f"fill='{COLORS[key]}' stroke='#b7bec8' stroke-width='0.5'/>"
        )
        parts.append(
            f"<text x='{lx + 16}' y='{MARGIN + 32}' {FONT} font-size='11' fill='#333'>{escape(desc)}</text>"
        )
        lx += 20 + len(desc) * 6.3 + 16
    parts.append(
        f"<text x='{lx + 4}' y='{MARGIN + 32}' {FONT} font-size='11' fill='#777' "
        f"font-style='italic'>cell area = packages using it ({escape(metric)})</text>"
    )
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
            f"height='{mr['dy']:.1f}' rx='2' fill='#f2f4f7' stroke='#c4ccd6'/>"
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
    svg = (
        head
        + "<rect width='100%' height='100%' fill='white'/>"
        + "".join(parts)
        + "".join(body)
        + "</svg>"
    )
    return svg, missing


def main() -> None:
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
        svg, missing = render_weighted(
            measured, weights, args.metric, args.min_weight, args.scale
        )
        Path(args.out).write_text(svg + "\n")
        print(
            f"wrote {args.out}  (area-weighted by {args.metric}, "
            f"scale={args.scale}, min_weight={args.min_weight})"
        )
        if missing:
            # These render as grey "?" but are used enough to be drawn -- usually a
            # stale usage key (op left the catalog) or a JSON that under-measured its
            # surface.  Warn so a wrong-looking map is explained, not mysterious.
            head = ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else "")
            print(
                f"WARNING: {len(missing)} drawn cell(s) have no support info "
                f"(grey): {head}",
                file=sys.stderr,
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
