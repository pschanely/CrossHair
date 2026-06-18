#!/usr/bin/env bash
set -euo pipefail

# Versions exercised in .github/workflows/test_crosshair.yml
CI_PYTHON_VERSIONS=(3.9 3.10 3.11 3.12 3.13 3.14)

usage() {
  cat <<EOF
Usage:
  switch-python <version>       Switch Python and reinstall CrossHair
  switch-python --install-all   Preinstall all CI Python versions (slow, one-time)
  switch-python --list          Show installed and CI target versions

Examples:
  switch-python 3.11
  switch-python 3.13.2

If a version isn't found, run 'pyenv update' to refresh the known list.

CI matrix: ${CI_PYTHON_VERSIONS[*]}
EOF
}

list_versions() {
  echo "CI versions: ${CI_PYTHON_VERSIONS[*]}"
  echo "Installed:"
  pyenv versions
}

install_all() {
  pyenv update
  for version in "${CI_PYTHON_VERSIONS[@]}"; do
    echo "Installing Python ${version}..."
    pyenv install -s "${version}"
  done
  echo "All CI Python versions are installed."
}

switch_version() {
  local version="$1"
  pyenv install -s "${version}"
  pyenv local "${version}"
  bash "$(dirname "$0")/install-crosshair.sh"
  python --version
  echo "CrossHair is ready on $(python --version)."
}

case "${1:-}" in
  --help|-h)
    usage
    ;;
  --list)
    list_versions
    ;;
  --install-all)
    install_all
    ;;
  "")
    usage
    exit 1
    ;;
  *)
    switch_version "$1"
    ;;
esac
