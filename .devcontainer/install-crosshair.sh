#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

# Quick sanity check that the CLI and C extension built for this interpreter.
crosshair -h >/dev/null
python -c "import _crosshair_tracers"
