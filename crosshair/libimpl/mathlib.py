import math

from crosshair import register_patch, NoTracing
from crosshair.libimpl.builtinslib import SymbolicFloat


def _isfinite(x):
    with NoTracing():
        if isinstance(x, SymbolicFloat):
            return True
        else:
            return math.isfinite(x)


def make_registrations():
    register_patch(math, _isfinite, "isfinite")
