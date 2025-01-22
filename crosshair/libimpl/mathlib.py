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
    smt_xor,
)
from crosshair.tracers import ResumedTracing
from crosshair.util import name_of_type
from crosshair.z3util import z3Not, z3Or


def _is_positive(x):
    if isinstance(x, SymbolicValue):
        if isinstance(x, PreciseIeeeSymbolicFloat):
            return SymbolicBool(z3Not(z3.fpIsNegative(x.var)))
        elif isinstance(x, RealBasedSymbolicFloat):
            return SymbolicBool(x.var >= 0)
        else:
            with ResumedTracing():
                return x >= 0
    else:
        return math.copysign(1, x) == 1


def _copysign(x, y):
    if not isinstance(x, Real):
        raise TypeError(f"must be real number, not {name_of_type(type(x))}")
    if not isinstance(y, Real):
        raise TypeError(f"must be real number, not {name_of_type(type(y))}")
    with NoTracing():
        x_is_positive = _is_positive(x)
        y_is_positive = _is_positive(y)
        # then invert as needed:
    invert = smt_xor(x_is_positive, y_is_positive)
    with NoTracing():
        if isinstance(invert, SymbolicBool) and isinstance(
            x, (PreciseIeeeSymbolicFloat, RealBasedSymbolicFloat)
        ):
            return type(x)(z3.If(invert.var, -x.var, x.var))
        with ResumedTracing():
            return -x if invert else x


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
