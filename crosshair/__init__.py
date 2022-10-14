"""Analyze Python code for correctness using symbolic execution."""

import sys

from crosshair.core import (
    SymbolicFactory,
    deep_realize,
    patch_to_return,
    realize,
    register_patch,
    register_type,
    with_realized_args,
)
from crosshair.statespace import StateSpace
from crosshair.tracers import NoTracing, ResumedTracing
from crosshair.util import IgnoreAttempt, debug

__version__ = "0.0.32"  # Do not forget to update in setup.py!
__author__ = "Phillip Schanely"
__license__ = "MIT"
__status__ = "Alpha"


def env_info() -> str:
    python_ver = sys.version.split(" ")[0]
    return f"CrossHair v{__version__} on {sys.platform}, Python {python_ver}"


__all__ = [
    "debug",
    "deep_realize",
    "patch_to_return",
    "realize",
    "register_patch",
    "register_type",
    "with_realized_args",
    "IgnoreAttempt",
    "NoTracing",
    "ResumedTracing",
    "StateSpace",
    "SymbolicFactory",
]
