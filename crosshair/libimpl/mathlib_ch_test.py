import math
import sys
from typing import Union

import pytest  # type: ignore

from crosshair.behavior_compare import compare_results
from crosshair.core import realize
from crosshair.options import AnalysisOptionSet
from crosshair.test_util import assert_comparisons_hold


def check_copysign(
    a: Union[int, float], b: Union[int, float], realize_a: bool, realize_b: bool
):
    """post: _"""
    if realize_a:
        a = realize(a)
    if realize_b:
        b = realize(b)
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
    fn = getattr(sys.modules[__name__], fn_name)
    assert_comparisons_hold(fn, AnalysisOptionSet(max_uninteresting_iterations=50))
