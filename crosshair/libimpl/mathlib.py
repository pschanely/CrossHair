import math

from crosshair import debug, register_patch, register_type, StateSpace
from crosshair.libimpl.builtinslib import SymbolicFloat

_orig_isfinite = math.isfinite


def _isfinite(x):
    if isinstance(x, SymbolicFloat):
        return True
    else:
        return _orig_isfinite(x)


def make_registrations():
    register_patch(math, _isfinite, "isfinite")
