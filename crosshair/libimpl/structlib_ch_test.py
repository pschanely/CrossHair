import struct
import sys

import pytest

from crosshair.behavior_compare import compare_results
from crosshair.test_util import assert_comparisons_hold


def check_calcsize(fmt: str):
    """post: _"""
    return compare_results(struct.calcsize, fmt)


def check_pack_h(x: int):
    """pre: -32768 <= x <= 32767
    post: _
    """
    return compare_results(struct.pack, ">h", x)


def check_pack_i(x: int):
    """pre: -2147483648 <= x <= 2147483647
    post: _
    """
    return compare_results(struct.pack, ">i", x)


def check_unpack_h(byts: bytes):
    """post: _"""
    return compare_results(struct.unpack, ">h", byts)


def check_unpack_i(byts: bytes):
    """post: _"""
    return compare_results(struct.unpack, ">i", byts)


def _roundtrip_h(x: int):
    return struct.unpack(">h", struct.pack(">h", x))[0]


def check_roundtrip_h(x: int):
    """pre: -32768 <= x <= 32767
    post: _
    """
    return compare_results(_roundtrip_h, x)


# This is the only real test definition.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    fn = getattr(sys.modules[__name__], fn_name)
    assert_comparisons_hold(fn)
