# CrossHair devcontainer

A reproducible dev environment that matches CI — dev dependencies, pre-commit
hooks, and pyenv-managed Python — keeping the project's Python toolchain
separate from whatever you have on your host.

## Getting started

Open the repo in the container with **Dev Containers: Reopen in Container**
(VS Code / Cursor). The first build installs Python 3.13 via pyenv, runs
`pip install -e .[dev]`, and installs the pre-commit hooks.

## Python versions

CrossHair is version-sensitive. Switch interpreters with `switch-python 3.11`;
`switch-python --install-all` preinstalls them all.
Run `switch-python --list` to see what's installed.

## What's shared with your host

The container isolates the **Python toolchain**, but it is not a security
sandbox. Your host `~/.claude` directory (Claude Code credentials and sessions)
is bind-mounted in so that state persists across rebuilds, and shell history
lives in a named volume. Don't treat the container as a trust boundary for
anything you wouldn't already expose to your host credentials.
