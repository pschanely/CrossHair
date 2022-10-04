import codecs

import pytest

from crosshair.core_and_libs import ResumedTracing
from crosshair.libimpl.builtinslib import LazyIntSymbolicStr, SymbolicBytes, SymbolicInt
from crosshair.options import AnalysisOptionSet
from crosshair.statespace import POST_FAIL, MessageType
from crosshair.test_util import check_states


def test_encode_strict(space):
    with ResumedTracing():
        with pytest.raises(UnicodeError):
            codecs.encode("Â", "ascii")


def test_encode_utf8_literal(space):
    with ResumedTracing():
        assert codecs.encode("Â", "utf-8") == b"\xc3\x82"


def test_encode_utf8_symbolic_char(space):
    cp = SymbolicInt("cp")
    space.add(cp.var >= ord("a"))
    space.add(cp.var <= ord("z"))
    with ResumedTracing():
        encoded = codecs.encode(chr(cp), "utf-8")
    assert isinstance(encoded, SymbolicBytes)
    byte_value = encoded[0]
    assert space.is_possible(byte_value.var == ord("a"))
    assert space.is_possible(byte_value.var == ord("b"))


def test_decode_utf8_symbolic_char(space):
    cp = SymbolicInt("cp")
    space.add(cp.var >= ord("a"))
    space.add(cp.var <= ord("z"))
    with ResumedTracing():
        decoded = codecs.decode(SymbolicBytes([cp]), "utf-8")
    assert isinstance(decoded, LazyIntSymbolicStr)
    assert space.is_possible(decoded._codepoints[0].var == ord("a"))
    assert space.is_possible(decoded._codepoints[0].var == ord("b"))


def test_decode_e2e():
    def f(byts: bytes) -> str:
        """
        post: _ != 'Â'
        raises: UnicodeDecodeError
        """
        return byts.decode("utf-8", errors="strict")

    opts = AnalysisOptionSet(max_iterations=100, per_condition_timeout=20)
    check_states(f, POST_FAIL, opts)
