import codecs
import io
import sys

import pytest

from crosshair.core import proxy_for_type
from crosshair.core_and_libs import ResumedTracing, standalone_statespace
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
    with ResumedTracing():
        space.add(cp >= ord("a"))
        space.add(cp <= ord("z"))
        encoded = codecs.encode(chr(cp), "utf-8")
    assert isinstance(encoded, SymbolicBytes)
    byte_value = encoded[0]
    with ResumedTracing():
        assert space.is_possible(byte_value == ord("a"))
        assert space.is_possible(byte_value == ord("b"))


def test_decode_utf8_symbolic_char(space):
    cp = SymbolicInt("cp")
    with ResumedTracing():
        space.add(cp >= ord("a"))
        space.add(cp <= ord("z"))
        decoded = codecs.decode(SymbolicBytes([cp]), "utf-8")
    assert isinstance(decoded, LazyIntSymbolicStr)
    with ResumedTracing():
        assert space.is_possible(decoded._codepoints[0] == ord("a"))
        assert space.is_possible(decoded._codepoints[0] == ord("b"))


def test_unsupported_codec_encode(space):
    s = proxy_for_type(str, "s")
    with ResumedTracing():
        s.encode("cp858")


def test_unsupported_codec_streamwriter(space):
    s = proxy_for_type(str, "s")
    buf = bytearray()
    with ResumedTracing():
        codecs.getwriter("cp858")(io.BytesIO(buf)).write(s)


@pytest.mark.xfail(reason="not yet implemented")
def test_supported_codec_streamwriter(space):
    s = proxy_for_type(str, "s")
    with ResumedTracing():
        space.add(len(s) == 1)
        buf = bytearray()
        codecs.getwriter("ascii")(io.BytesIO(buf)).write(s)
        space.is_possible(buf[0] == ord("x"))


@pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="Need to intercept UnicodeDecodeError.str in 3.13+",
)
def test_decode_e2e():
    def f(byts: bytes) -> str:
        """
        post: _ != 'Â'
        raises: UnicodeDecodeError
        """
        return byts.decode("utf-8", errors="strict")

    check_states(f, POST_FAIL)
