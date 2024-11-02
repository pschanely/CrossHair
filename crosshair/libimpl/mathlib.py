import math
import sys
from numbers import Real

import z3  # type: ignore

from crosshair import NoTracing, register_patch
from crosshair.core import with_realized_args
from crosshair.libimpl.builtinslib import (
    PreciseIeeeSymbolicFloat,
    RealBasedSymbolicFloat,
    SymbolicBool,
    SymbolicIntable,
    SymbolicValue,
)
from crosshair.tracers import ResumedTracing
from crosshair.util import name_of_type
from crosshair.z3util import z3Not, z3Or


def _copysign(x, y):
    if not isinstance(x, Real):
        raise TypeError(f"must be real number, not {name_of_type(type(x))}")
    if not isinstance(y, Real):
        raise TypeError(f"must be real number, not {name_of_type(type(y))}")
    with NoTracing():
        # Find the sign of y:
        if isinstance(y, SymbolicValue):
            if isinstance(y, PreciseIeeeSymbolicFloat):
                y_is_positive = not SymbolicBool(z3.fpIsNegative(y.var))
            else:
                with ResumedTracing():
                    y_is_positive = y >= 0
        else:
            y_is_positive = math.copysign(1, y) == 1
        # then invert as needed:
        if isinstance(x, PreciseIeeeSymbolicFloat):
            if y_is_positive:
                return PreciseIeeeSymbolicFloat(z3.If(z3.fpIsNegative(x.var), -x, x))
            else:
                return PreciseIeeeSymbolicFloat(z3.If(z3.fpIsNegative(x.var), x, -x))
        with ResumedTracing():
            if y_is_positive:
                return x if x >= 0 else -x
            else:
                return -x if x >= 0 else x


if sys.version_info >= (3, 9):

    def _gcd(a=0, b=0):
        while b:
            a, b = b, a % b
        return abs(a)

else:  # (arguments were required in Python <= 3.8)

    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)


def _isfinite(x):
    with NoTracing():
        if isinstance(x, (SymbolicIntable, RealBasedSymbolicFloat)):
            return True
        elif isinstance(x, PreciseIeeeSymbolicFloat):
            return SymbolicBool(z3Not(z3Or(z3.fpIsNaN(x.var), z3.fpIsInf(x.var))))
        else:
            return math.isfinite(x)


def _isnan(x):
    with NoTracing():
        if isinstance(x, (SymbolicIntable, RealBasedSymbolicFloat)):
            return False
        elif isinstance(x, PreciseIeeeSymbolicFloat):
            return SymbolicBool(z3.fpIsNaN(x.var))
        else:
            return math.isnan(x)


def _isinf(x):
    with NoTracing():
        if isinstance(x, (SymbolicIntable, RealBasedSymbolicFloat)):
            return False
        elif isinstance(x, PreciseIeeeSymbolicFloat):
            return SymbolicBool(z3.fpIsInf(x.var))
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
    register_patch(math.copysign, _copysign)
    register_patch(math.gcd, _gcd)
    register_patch(math.isfinite, _isfinite)
    register_patch(math.isnan, _isnan)
    register_patch(math.isinf, _isinf)
    for fn_name in _FUNCTIONS_WITH_REALIZATION:
        fn = getattr(math, fn_name)
        register_patch(
            fn, with_realized_args(fn, deep=True)
        )  # deep realization needed for Fraction instances
