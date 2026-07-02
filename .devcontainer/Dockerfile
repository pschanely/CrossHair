# CrossHair dev environment: isolated from the host for safer agent work.
# trixie (Debian 13, glibc >= 2.38) so newer z3-solver aarch64 wheels
# (manylinux_2_38) install instead of falling back to a failing sdist build.
FROM mcr.microsoft.com/devcontainers/base:trixie

# Build deps for pyenv/CPython and the _crosshair_tracers C extension.
# nodejs + npm are here (not left to the claude-code feature) because on trixie
# the feature installs node without npm, so `npm install -g` fails; providing
# both up front makes the feature skip its own Node install.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        build-essential curl git \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
        libffi-dev liblzma-dev \
        nodejs npm \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

USER vscode
ENV PYENV_ROOT="/home/vscode/.pyenv"
ENV PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"

RUN curl -fsSL https://pyenv.run | bash
