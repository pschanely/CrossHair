import math

from crosshair import smt
from crosshair.smtlib import parse_smtlib_literal


def test_parse_smtlib_literal():
    assert parse_smtlib_literal(smt.FPVal(1.23, smt.Float64()).sexpr()) == 1.23
    assert math.isnan(parse_smtlib_literal(smt.FPVal(math.nan, smt.Float64()).sexpr()))
    assert (
        parse_smtlib_literal(smt.FPVal(-math.inf, smt.Float64()).sexpr()) <= -math.inf
    )
    negzero = parse_smtlib_literal(smt.FPVal(-0.0, smt.Float64()).sexpr())
    assert negzero == 0
    assert math.copysign(42, negzero) == -42
