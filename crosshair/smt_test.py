from enum import IntEnum

import pytest

from crosshair import smt


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
