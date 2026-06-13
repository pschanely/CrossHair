#!/usr/bin/env bash
set -euo pipefail

# A fresh named volume mounts root-owned; make the history dir writable by us.
sudo chown -R "$(id -u):$(id -g)" /home/vscode/.commandhistory

pyenv update
pyenv install -s 3.13
pyenv local 3.13

bash "$(dirname "$0")/install-crosshair.sh"
pre-commit install --install-hooks

echo "CrossHair devcontainer is ready (Python 3.13)."
echo "Use 'switch-python 3.11' (etc.) to test other CI versions."
