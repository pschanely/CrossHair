import operator
import sys
from decimal import BasicContext, Decimal, ExtendedContext, localcontext
from typing import Union

import pytest  # type: ignore

from crosshair.behavior_compare import (
    ResultComparison,
    compare_results,
    compare_returns,
)
from crosshair.libimpl.decimallib import C_DECIMAL_IS_ACTIVE
from crosshair.test_util import assert_comparisons_hold

pytestmark = pytest.mark.skipif(
    not C_DECIMAL_IS_ACTIVE,
    reason="decimal is unmodeled unless the C _decimal extension is active",
)


def _binary_op_under_context(ctx, op):
    def run_op(d1, d2):
        with localcontext(ctx):
            return op(d1, d2)

    return run_op


def check_division(
    decimal1: Decimal, decimal2: Union[Decimal, int, float]
) -> ResultComparison:
    """post: _"""
    return compare_returns(operator.truediv, decimal1, decimal2)


def check_pow(decimal1: Decimal, decimal2: Decimal) -> ResultComparison:
    """post: _"""
    return compare_results(operator.pow, decimal1, decimal2)


def check_extended_context(
    decimal1: Decimal, decimal2: Union[Decimal, int, float]
) -> ResultComparison:
    """post: _"""
    return compare_results(
        _binary_op_under_context(ExtendedContext, operator.truediv), decimal1, decimal2
    )


def check_basic_context(
    decimal1: Decimal, decimal2: Union[Decimal, int, float]
) -> ResultComparison:
    """post: _"""
    return compare_results(
        _binary_op_under_context(BasicContext, operator.truediv), decimal1, decimal2
    )


def check_div_using_context_method(
    decimal1: Decimal, decimal2: Union[Decimal, int, float]
) -> ResultComparison:
    """post: _"""
    return compare_returns(BasicContext.divide, decimal1, decimal2)


def check_remainder_near_using_context_parameter(
    decimal1: Decimal, decimal2: Union[Decimal, int, float]
) -> ResultComparison:
    """post: _"""
    return compare_returns(
        lambda d1, d2: d1.remainder_near(d2, context=BasicContext), decimal1, decimal2
    )


def check_create_decimal_from_float(float_number: float):
    """post: _"""
    return compare_results(BasicContext.create_decimal_from_float, float_number)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    fn = getattr(sys.modules[__name__], fn_name)
    assert_comparisons_hold(fn)
