import codecs

import pytest

from crosshair.core_and_libs import ResumedTracing
from crosshair.libimpl.builtinslib import SymbolicInt, SymbolicBytes


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
