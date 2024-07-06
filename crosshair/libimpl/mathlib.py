import math

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
    "cbrt",
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
    "exp2",
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
    "lcm",
    "ldexp",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "modf",
    "nextafter",
    "perm",
    "pow",
    "prod",
    "radians",
    "remainder",
    "sin",
    "sinh",
    "sqrt",
    "sumprod",
    "tan",
    "tanh",
    "trunc",
    "ulp",
]


def make_registrations():
    register_patch(math.isfinite, _isfinite)
    register_patch(math.isnan, _isnan)
    register_patch(math.isinf, _isinf)
    for fn_name in _FUNCTIONS_WITH_REALIZATION:
        fn = getattr(math, fn_name)
        register_patch(fn, with_realized_args(fn))
