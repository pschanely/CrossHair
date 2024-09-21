import operator
import sys
from decimal import BasicContext, Decimal, ExtendedContext, localcontext
from typing import Union

import pytest  # type: ignore

from crosshair.core_and_libs import MessageType, analyze_function, run_checkables
from crosshair.test_util import ResultComparison, compare_results, compare_returns


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


def check_div_using_context_parameter(
    decimal1: Decimal, decimal2: Union[Decimal, int, float]
) -> ResultComparison:
    """post: _"""
    return compare_returns(
        lambda d1, d2: d1.divide(d2, context=BasicContext), decimal1, decimal2
    )


def check_create_decimal_from_float(float_number: float):
    """post: _"""
    return compare_results(BasicContext.create_decimal_from_float, float_number)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    this_module = sys.modules[__name__]
    messages = run_checkables(analyze_function(getattr(this_module, fn_name)))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
