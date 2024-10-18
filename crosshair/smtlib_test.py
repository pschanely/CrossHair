import math

import z3  # type: ignore

from crosshair.smtlib import parse_smtlib_literal


def test_parse_smtlib_literal():
    assert parse_smtlib_literal(z3.FPVal(1.23, z3.Float64).sexpr()) == 1.23
    assert math.isnan(parse_smtlib_literal(z3.FPVal(math.nan, z3.Float64).sexpr()))
    assert parse_smtlib_literal(z3.FPVal(-math.inf, z3.Float64).sexpr()) <= -math.inf
    negzero = parse_smtlib_literal(z3.FPVal(-0.0, z3.Float64).sexpr())
    assert negzero == 0
    assert math.copysign(42, negzero) == -42
