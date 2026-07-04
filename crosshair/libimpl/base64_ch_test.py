import base64
import sys

import pytest

from crosshair.behavior_compare import compare_results
from crosshair.core import analyze_function, run_checkables
from crosshair.statespace import MessageType

# The base64 module is powered entirely by the binascii patches.  These checks
# exercise it at the module level (the binascii tests only call the functions
# directly), which is the coverage gap that let the 3.15 keyword-argument
# regression slip past.


def check_b64encode(byts: bytes):
    """post: _"""
    return compare_results(base64.b64encode, byts)


def check_b64encode_altchars(byts: bytes):
    """post: _"""
    return compare_results(lambda b: base64.b64encode(b, altchars=b"-_"), byts)


def check_b64decode(byts: bytes):
    """post: _"""
    return compare_results(base64.b64decode, byts)


def check_b64_roundtrip(byts: bytes):
    """post: _"""
    return compare_results(lambda b: base64.b64decode(base64.b64encode(b)), byts)


def check_urlsafe_b64encode(byts: bytes):
    """post: _"""
    return compare_results(base64.urlsafe_b64encode, byts)


def check_urlsafe_roundtrip(byts: bytes):
    """post: _"""
    return compare_results(
        lambda b: base64.urlsafe_b64decode(base64.urlsafe_b64encode(b)), byts
    )


def check_encodebytes(byts: bytes):
    """post: _"""
    return compare_results(base64.encodebytes, byts)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    this_module = sys.modules[__name__]
    fn = getattr(this_module, fn_name)
    messages = run_checkables(analyze_function(fn))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == [], [(m.state, m.message) for m in errors]
