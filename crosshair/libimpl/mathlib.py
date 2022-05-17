import math

from crosshair import NoTracing, register_patch
from crosshair.libimpl.builtinslib import SymbolicNumberAble


def _isfinite(x):
    with NoTracing():
        if isinstance(x, SymbolicNumberAble):
            return True
        else:
            return math.isfinite(x)


def make_registrations():
    register_patch(math.isfinite, _isfinite)
