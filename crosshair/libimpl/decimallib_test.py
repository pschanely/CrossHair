from decimal import (
    Decimal,
    DivisionByZero,
    ExtendedContext,
    InvalidOperation,
    localcontext,
)

import pytest

from crosshair.core import proxy_for_type, standalone_statespace
from crosshair.libimpl.decimallib import Decimal as PyDecimal
from crosshair.tracers import NoTracing
from crosshair.util import debug


def test_mixed_decimal_addition() -> None:
    d1 = Decimal("1.05")
    with standalone_statespace:
        d2 = proxy_for_type(Decimal, "d2")
        debug("type(d2)", type(d2))
        d1 + d2


def test_external_decimal_context() -> None:
    with localcontext(ExtendedContext):
        Decimal("43.4") / 0  # does not raise
    with pytest.raises(DivisionByZero):
        Decimal("43.4") / 0
    with pytest.raises(InvalidOperation):
        Decimal("0") / 0
    with pytest.raises(DivisionByZero):
        PyDecimal("43.4") / 0
    with standalone_statespace as space:
        with NoTracing():
            d1 = proxy_for_type(Decimal, "d1")
        if d1 == 0:
            d1 += 1
        with pytest.raises(DivisionByZero):
            d1 / 0
        with pytest.raises(InvalidOperation):
            (d1 - d1) / 0


def test_context_method_on_symbolic():
    with standalone_statespace:
        ExtendedContext.exp(proxy_for_type(Decimal, "d"))
        ExtendedContext.divide_int(Decimal(12), proxy_for_type(Decimal, "d"))
        ExtendedContext.divide_int(Decimal(12), Decimal(2))


def test_precision():
    """post: _"""
    d1, d2 = Decimal("3.4445"), Decimal("1.0023")
    expected = Decimal("4.45")
    assert d1 + d2 != expected
    with standalone_statespace:
        with localcontext() as ctx:
            ctx.prec = 3
            assert d1 + d2 == expected
        with localcontext() as ctx:
            ctx.prec = 1
            assert +expected == Decimal("4")


# Still working on this! (rn, issue with z3 int exponent vars becoming reals)
# def test_decimal_end_to_end():
#     def add_tax(price: Decimal):
#         """post: _ != Decimal('1.05')"""
#         ctx = ExtendedContext.copy()
#         ctx.prec = 3
#         with localcontext(ctx):
#             return price + Decimal("0.05")
#             # return price * Decimal("1.05")

#     check_states(add_tax, POST_FAIL)
