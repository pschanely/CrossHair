import math
import sys
from typing import Union

import pytest  # type: ignore

from crosshair.core_and_libs import MessageType, analyze_function, run_checkables
from crosshair.test_util import compare_results


def check_copysign(a: Union[int, float], b: Union[int, float]):
    """post: _"""
    return compare_results(math.copysign, a, b)


def check_gcd(a: int, b: int):
    """
    pre: all([-10 < a, a < 10, -10 < b, b < 10])  # for performance
    post: _
    """
    return compare_results(math.gcd, a, b)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    this_module = sys.modules[__name__]
    messages = run_checkables(analyze_function(getattr(this_module, fn_name)))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
