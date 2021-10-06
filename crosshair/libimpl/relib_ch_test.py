import re
import sys
from typing import Optional, Sequence

import pytest  # type: ignore

from crosshair.core_and_libs import analyze_function
from crosshair.core_and_libs import run_checkables
from crosshair.core_and_libs import MessageType
from crosshair.options import AnalysisOptionSet
from crosshair.test_util import compare_results
from crosshair.test_util import ResultComparison


def groups(match: Optional[re.Match]) -> Optional[Sequence]:
    if match is None:
        return None
    return match.groups()


def check_inverted_categories(text: str) -> ResultComparison:
    """
    pre: len(text) == 3
    post: _
    """
    return compare_results(lambda t: groups(re.fullmatch(r"\W\S\D", t)), text)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    opts = AnalysisOptionSet(
        max_iterations=10, per_condition_timeout=30, per_path_timeout=4
    )
    this_module = sys.modules[__name__]
    fn = getattr(this_module, fn_name)
    messages = run_checkables(analyze_function(fn, opts))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
