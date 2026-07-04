import binascii
import sys

import pytest

from crosshair.behavior_compare import compare_results
from crosshair.core import analyze_function, run_checkables
from crosshair.statespace import MessageType


def check_b2a_base64(byts: bytes, newline: bool):
    """post: _"""
    return compare_results(binascii.b2a_base64, byts, newline=newline)


def check_a2b_base64(byts: bytes, strict_mode: bool):
    """post: _"""
    kw = {"strict_mode": strict_mode} if sys.version_info >= (3, 11) else {}
    return compare_results(binascii.a2b_base64, byts, **kw)


def check_a2b_base64_unpadded(byts: bytes):
    """post: _"""
    # 3.15 added ``padded``; older versions always require padding.
    kw = {"padded": False} if sys.version_info >= (3, 15) else {}
    return compare_results(binascii.a2b_base64, byts, **kw)


def check_a2b_base64_alphabet(byts: bytes):
    """post: _"""
    # 3.15 added a custom decode ``alphabet``; fall back to the standard one on
    # older versions so the check stays meaningful everywhere.
    if sys.version_info >= (3, 15):
        return compare_results(
            binascii.a2b_base64, byts, alphabet=binascii.URLSAFE_BASE64_ALPHABET
        )
    return compare_results(binascii.a2b_base64, byts)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    this_module = sys.modules[__name__]
    fn = getattr(this_module, fn_name)
    messages = run_checkables(analyze_function(fn))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == [], [(m.state, m.message) for m in errors]
