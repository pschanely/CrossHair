import math
import sys

from crosshair import NoTracing, register_patch
from crosshair.core import with_realized_args
from crosshair.libimpl.builtinslib import SymbolicNumberAble


def _isfinite(x):
    with NoTracing():
        if isinstance(x, SymbolicNumberAble):
            return True
        else:
            return math.isfinite(x)


def _isnan(x):
    with NoTracing():
        if isinstance(x, SymbolicNumberAble):
            return False
        else:
            return math.isnan(x)


def _isinf(x):
    with NoTracing():
        if isinstance(x, SymbolicNumberAble):
            return False
        else:
            return math.isinf(x)


_FUNCTIONS_WITH_REALIZATION = [
    # TODO: we could attempt to implement some of these in the SMT solver
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "comb",
    "copysign",
    "cos",
    "cosh",
    "degrees",
    "dist",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "fabs",
    "factorial",
    "floor",
    "fmod",
    "frexp",
    "fsum",
    "gamma",
    "gcd",
    "hypot",
    "isclose",
    "isqrt",
    "ldexp",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "modf",
    "perm",
    "pow",
    "prod",
    "radians",
    "remainder",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
]

if sys.version_info >= (3, 9):
    _FUNCTIONS_WITH_REALIZATION.extend(
        [
            "lcm",
            "nextafter",
            "ulp",
        ]
    )

if sys.version_info >= (3, 11):
    _FUNCTIONS_WITH_REALIZATION.extend(
        [
            "cbrt",
            "exp2",
        ]
    )

if sys.version_info >= (3, 12):
    _FUNCTIONS_WITH_REALIZATION.append("sumprod")


def make_registrations():
    register_patch(math.isfinite, _isfinite)
    register_patch(math.isnan, _isnan)
    register_patch(math.isinf, _isinf)
    for fn_name in _FUNCTIONS_WITH_REALIZATION:
        fn = getattr(math, fn_name)
        register_patch(fn, with_realized_args(fn))
