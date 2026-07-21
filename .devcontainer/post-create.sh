#!/usr/bin/env bash
set -euo pipefail

# A fresh named volume mounts root-owned; make these dirs writable by us.
sudo chown -R "$(id -u):$(id -g)" /home/vscode/.commandhistory
sudo chown -R "$(id -u):$(id -g)" /home/vscode/.claude


pyenv update
pyenv install -s 3.13
pyenv local 3.13
# Also set a global default so `python`/`pip`/`crosshair` resolve even when the
# shell cwd is outside the repo (otherwise pyenv falls back to `system`, which
# has no Python, and bare `python` reports "command not found").
pyenv global 3.13

bash "$(dirname "$0")/install-crosshair.sh"

# Pre-fetch the pinned code-quality tools (black/isort/flake8/mypy at the versions
# in .pre-commit-config.yaml) so `pre-commit run` matches CI exactly -- and can't
# drift from it even when a stale interpreter has an older pip-installed black.
# Deliberately NOT `pre-commit install`: we don't wire a git hook, so commits stay
# free of hidden reformats.  Run checks explicitly with `pre-commit run` (or
# `pre-commit run --all-files`); CI enforces them on push.
pre-commit install-hooks

echo "CrossHair devcontainer is ready (Python 3.13)."
echo "Use 'switch-python 3.11' (etc.) to test other CI versions."
