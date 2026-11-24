from enum import IntEnum

import pytest
import z3  # type: ignore

from crosshair import smt
from crosshair.util import CrosshairUnsupported


class IntSubClass(IntEnum):
    FIRST = 1


def test_int_val_on_int_enum():
    smt.int_val(IntSubClass.FIRST)


def test_constructors_reject_python_values():
    with pytest.raises(AttributeError):
        smt.ite(smt.true, 1, 0)
    with pytest.raises(AttributeError):
        smt.and_(smt.true, True)


def test_not_unwraps_negation():
    x = smt.const("x", smt.bool_sort)
    assert smt.not_(smt.not_(x)).eq(x)
    polarity, inner = smt.pop_not(smt.not_(x))
    assert polarity is False
    assert inner.eq(x)


def test_quantifiers():
    x = smt.const("x", smt.int_sort)
    assert smt.forall([x], smt.ge(x, x)).is_forall()
    assert smt.exists([x], smt.gt(x, smt.int_val(0))).is_exists()


def test_declare_datatype():
    maybe = smt.declare_datatype(
        "maybe_int", {"none": [], "some": [("val", smt.int_sort)]}
    )
    assert maybe.num_constructors() == 2
    some = smt.apply(maybe.some, smt.int_val(3))
    assert smt.apply(maybe.val, some).sort() == smt.int_sort


@pytest.mark.parametrize(
    "value,digits",
    [(v, str(v)) for v in (10**700 + 123, -(10**700 + 123), 7 * 10**2000 - 1)],
    ids=["positive", "negative", "deep"],
)
def test_int_numerals_beyond_digit_limit(small_int_str_digit_limit, value, digits):
    with pytest.raises(ValueError):
        str(value)
    literal = smt.int_val(value)
    assert literal.as_string() == digits
    assert smt.numeral_to_int(literal) == value


def test_to_python_AlgebraicNumRef():
    # Tests that z3.AlgebraicNumRef is handled properly.
    # See https://github.com/pschanely/CrossHair/issues/242
    rt2 = z3.simplify(z3.Sqrt(2))
    assert type(rt2) == z3.AlgebraicNumRef
    smt.to_python(rt2)


def test_to_python_ArithRef():
    # Tests that a plain z3.ArithRef can be exported as Python
    # See https://github.com/pschanely/CrossHair/issues/381
    rt2 = smt.to_int(2 ** smt.const("x", smt.int_sort))
    assert type(rt2) == z3.ArithRef
    smt.to_python(rt2)


def test_to_python_unfolded_power():
    # z3's simplifier leaves a ground power above max_degree=64 unevaluated;
    # realization must refold it rather than return None.
    unfolded = z3.simplify(smt.to_int(smt.int_val(7) ** smt.int_val(200)))
    assert not z3.is_int_value(unfolded)
    assert smt.to_python(unfolded) == 7**200


def test_to_python_refuses_astronomical_int():
    huge = smt.to_int(smt.int_val(2) ** smt.int_val(5_000_000))
    with pytest.raises(CrosshairUnsupported):
        smt.to_python(huge)
