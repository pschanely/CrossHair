# CrossHair dev environment: isolated from the host for safer agent work.
FROM mcr.microsoft.com/devcontainers/base:bookworm

# Build deps for pyenv/CPython and the _crosshair_tracers C extension.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        build-essential curl git \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
        libffi-dev liblzma-dev \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

USER vscode
ENV PYENV_ROOT="/home/vscode/.pyenv"
ENV PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"

RUN curl -fsSL https://pyenv.run | bash
