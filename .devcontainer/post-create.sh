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

echo "CrossHair devcontainer is ready (Python 3.13)."
echo "Use 'switch-python 3.11' (etc.) to test other CI versions."
