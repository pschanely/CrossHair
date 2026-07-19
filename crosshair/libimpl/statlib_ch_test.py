import stat
import sys

import pytest

from crosshair.behavior_compare import compare_results
from crosshair.core import analyze_function, run_checkables
from crosshair.statespace import MessageType


def check_S_ISDIR(mode: int):
    """post: _"""
    return compare_results(stat.S_ISDIR, mode)


def check_S_ISREG(mode: int):
    """post: _"""
    return compare_results(stat.S_ISREG, mode)


def check_S_IFMT(mode: int):
    """post: _"""
    return compare_results(stat.S_IFMT, mode)


def check_S_IMODE(mode: int):
    """post: _"""
    return compare_results(stat.S_IMODE, mode)


def check_filemode(mode: int):
    """post: _"""
    return compare_results(stat.filemode, mode)


@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    this_module = sys.modules[__name__]
    fn = getattr(this_module, fn_name)
    messages = run_checkables(analyze_function(fn))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == [], [(m.state, m.message) for m in errors]
