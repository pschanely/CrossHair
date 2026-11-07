import base64
import binascii
import sys

import pytest

from crosshair.behavior_compare import compare_results
from crosshair.test_util import assert_comparisons_hold


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


def check_b2a_base32(byts: bytes):
    """post: _"""
    if sys.version_info >= (3, 15):
        return compare_results(binascii.b2a_base32, byts)
    return compare_results(base64.b32encode, byts)


def check_b2a_base32_unpadded(byts: bytes):
    """post: _"""
    if sys.version_info >= (3, 15):
        return compare_results(binascii.b2a_base32, byts, padded=False)
    return compare_results(base64.b32encode, byts)


def check_b2a_base32_wrapped(byts: bytes):
    """post: _"""
    if sys.version_info >= (3, 15):
        return compare_results(binascii.b2a_base32, byts, wrapcol=8)
    return compare_results(base64.b32encode, byts)


def check_a2b_base32(byts: bytes):
    """post: _"""
    if sys.version_info >= (3, 15):
        return compare_results(binascii.a2b_base32, byts)
    return compare_results(base64.b32decode, byts)


def check_a2b_base32_unpadded(byts: bytes):
    """post: _"""
    if sys.version_info >= (3, 15):
        return compare_results(binascii.a2b_base32, byts, padded=False)
    return compare_results(base64.b32decode, byts)


def check_a2b_base32_hex_alphabet(byts: bytes):
    """post: _"""
    if sys.version_info >= (3, 15):
        return compare_results(
            binascii.a2b_base32, byts, alphabet=binascii.BASE32HEX_ALPHABET
        )
    return compare_results(base64.b32hexdecode, byts)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    fn = getattr(sys.modules[__name__], fn_name)
    assert_comparisons_hold(fn)
