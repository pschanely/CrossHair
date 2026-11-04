import math

from crosshair import smt
from crosshair.smtlib import parse_smtlib_literal


def test_parse_smtlib_literal():
    assert parse_smtlib_literal(smt.fp_val(1.23, smt.float64_sort).sexpr()) == 1.23
    assert math.isnan(
        parse_smtlib_literal(smt.fp_val(math.nan, smt.float64_sort).sexpr())
    )
    assert (
        parse_smtlib_literal(smt.fp_val(-math.inf, smt.float64_sort).sexpr())
        <= -math.inf
    )
    negzero = parse_smtlib_literal(smt.fp_val(-0.0, smt.float64_sort).sexpr())
    assert negzero == 0
    assert math.copysign(42, negzero) == -42
