import binascii
import sys

import pytest

from crosshair.core import analyze_function, run_checkables
from crosshair.statespace import MessageType
from crosshair.test_util import compare_results


def check_b2a_base64(byts: bytes, newline: bool):
    """post: _"""
    return compare_results(binascii.b2a_base64, byts, newline=newline)


def check_a2b_base64(byts: bytes, strict_mode: bool):
    """post: _"""
    kw = {"strict_mode": strict_mode} if sys.version_info >= (3, 11) else {}
    return compare_results(binascii.a2b_base64, byts, **kw)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    this_module = sys.modules[__name__]
    fn = getattr(this_module, fn_name)
    messages = run_checkables(analyze_function(fn))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
