"""
Fetch a corpus of top-PyPI packages for usage mining -- DOWNLOAD + UNZIP ONLY.

SAFETY: this never runs any downloaded code.  It does not use ``pip`` (which would
execute ``setup.py`` for sdists); it downloads pre-built **wheels** (plain zip
archives) over https and extracts their ``.py`` files with ``zipfile``.  No import,
no exec, no build step.  The downstream miner (``mine_usage``) is likewise pure
``ast.parse``.

    python -m crosshair.tools.fetch_corpus --n 200 --out /tmp/pypi_corpus

Each package's ``.py`` files land under ``<out>/<project>/`` (one package per dir),
ready for ``mine_usage --corpus <out>``.
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

_TOP = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"
_UA = {"User-Agent": "crosshair-usage-miner (static AST analysis; no code execution)"}


def _get(url, timeout=60):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=_UA), timeout=timeout
    ).read()


def top_projects(n):
    rows = json.loads(_get(_TOP)).get("rows", [])
    return [r["project"] for r in rows[:n]]


def _wheel_url(project):
    try:
        meta = json.loads(_get(f"https://pypi.org/pypi/{project}/json", timeout=30))
    except Exception:
        return None
    wheels = [f for f in meta.get("urls", []) if f["filename"].endswith(".whl")]
    if not wheels:
        return None
    # prefer the pure-python any-wheel; else the first (we only read .py text)
    wheels.sort(key=lambda f: ("py3-none-any" not in f["filename"], f["filename"]))
    return wheels[0]["url"]


def _extract_py(wheel_bytes, dest):
    """Extract .py members from a wheel (zip) into dest.  zip-slip guarded; runs nothing."""
    dest = dest.resolve()
    count = 0
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as z:
        for name in z.namelist():
            if not name.endswith(".py") or ".dist-info/" in name or ".data/" in name:
                continue
            target = (dest / name).resolve()
            if dest != target and dest not in target.parents:  # path traversal guard
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(
        description="Download top-PyPI wheels and extract .py (no code execution)."
    )
    ap.add_argument("--n", type=int, default=200, help="number of top packages")
    ap.add_argument("--out", required=True, help="corpus output directory")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    projects = top_projects(args.n)
    ok = nowheel = fail = 0
    for i, proj in enumerate(projects, 1):
        safe = proj.replace("/", "_")
        if (out / safe).exists():
            ok += 1
            continue
        url = _wheel_url(proj)
        if not url:
            nowheel += 1
            print(f"[{i}/{len(projects)}] {proj}: no wheel, skip")
            continue
        try:
            n_py = _extract_py(_get(url), out / safe)
            ok += 1
            print(f"[{i}/{len(projects)}] {proj}: {n_py} .py files")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(projects)}] {proj}: FAILED ({type(e).__name__})")
    print(f"\ndone: {ok} fetched, {nowheel} no-wheel, {fail} failed -> {out}")


if __name__ == "__main__":
    main()
