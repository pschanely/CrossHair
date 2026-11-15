import subprocess
import sys
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


@pytest.mark.skipif(
    not hasattr(sys, "set_int_max_str_digits"), reason="No integer string limit"
)
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_int_val_exceeds_decimal_digit_limit(sign):
    original_limit = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        value = sign * (10**700 + 123)
        with pytest.raises(ValueError):
            str(value)
        literal = smt.int_val(value)
        expected = ("-" if sign < 0 else "") + "1" + "0" * 697 + "123"
        assert literal.as_string() == expected
        assert sys.get_int_max_str_digits() == 640
    finally:
        sys.set_int_max_str_digits(original_limit)


@pytest.mark.parametrize("sign", [1, -1])
def test_int_val_without_decimal_accelerator(sign):
    script = """
import sys
sys.modules["_decimal"] = None
from crosshair import smt
sys.set_int_max_str_digits(640)
sign = int(sys.argv[1])
value = sign * (10**700 + 123)
literal = smt.int_val(value)
expected = ("-" if sign < 0 else "") + "1" + "0" * 697 + "123"
assert literal.as_string() == expected
assert sys.get_int_max_str_digits() == 640
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(sign)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
