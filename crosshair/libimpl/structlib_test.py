import re
import struct

import pytest

from crosshair.core import proxy_for_type, standalone_statespace
from crosshair.core_and_libs import NoTracing
from crosshair.libimpl.structlib import (
    _calcsize,
    _get_item_size,
    _iter_unpack,
    _pack,
    _pack_arg_count,
    _pack_into,
    _parse_format,
    _unpack,
    _unpack_from,
)
from crosshair.tracers import ResumedTracing


@pytest.mark.parametrize(
    "fmt",
    [
        ">h",
        "<i",
        ">q",
        ">h i",
        ">2h",
        ">hH",
        ">5s",
        ">10p",
        ">b",
        ">B",
        ">?",
        ">f",
        ">d",
        ">e",
        "=h",
        "!h",
        ">3h 2H",
    ],
)
def test_calcsize_matches_parse_format(fmt: str) -> None:
    """calcsize should match size computed from _parse_format + _get_item_size."""
    expected = struct.calcsize(fmt)
    prefix, items = _parse_format(fmt)
    computed = sum(_get_item_size(fc, count, prefix) for fc, count in items)
    assert computed == expected, f"Format {fmt!r}: computed {computed} != {expected}"


def test_pack_arg_count_ignores_padding() -> None:
    _, items = _parse_format(">h3x")
    assert _pack_arg_count(items) == 1
    _, items = _parse_format("3x")
    assert _pack_arg_count(items) == 0


def test_calcsize_with_symbolic_format(space) -> None:
    """calcsize with symbolic format string - should not explode (realize format)."""
    fmt = proxy_for_type(str, "fmt")
    with ResumedTracing():
        space.add(fmt == ">h")
        result = struct.calcsize(fmt)
    assert result == 2


def test_pack_with_padding_and_symbolic_int(space) -> None:
    """Padding 'x' does not consume pack arguments; size includes pad bytes."""
    x = proxy_for_type(int, "x")
    with ResumedTracing():
        space.add(x >= -32768)
        space.add(x < 32768)
        packed = struct.pack(">h3x", x)
    assert len(packed) == 5
    assert struct.pack(">h3x", 42) == _pack(">h3x", 42)


def test_pack_with_symbolic_int_bounded(space) -> None:
    """pack with bounded symbolic int (2-byte short) - should produce SymbolicBytes."""
    # x in range(-32768, 32768) always yields 2 bytes for ">h"
    x = proxy_for_type(int, "x")
    with ResumedTracing():
        space.add(x >= -32768)
        space.add(x < 32768)
        packed = struct.pack(">h", x)
        assert len(packed) == 2
        assert space.is_possible(packed[0] == 42)
        assert space.is_possible(packed[0] == 43)
    # Round-trip: unpack should give back equivalent value
    unpacked = struct.unpack(">h", packed)
    assert len(unpacked) == 1
    with ResumedTracing():
        assert not space.is_possible(unpacked[0] != x)


def test_unpack_int(space) -> None:
    """unpack on SymbolicBytes - should produce symbolic int"""
    byts = proxy_for_type(bytes, "byts")
    with ResumedTracing():
        space.add(len(byts) == 2)
        unpacked = struct.unpack(">h", byts)
        assert len(unpacked) == 1
        unpacked_int = unpacked[0]
        with NoTracing():
            assert not isinstance(unpacked_int, int)
        assert space.is_possible(unpacked_int == 42)
        assert not space.is_possible(unpacked_int >= 32768)
        assert not space.is_possible(unpacked_int < -32768)
        # Verify result is usable - round-trip pack/unpack preserves value
        repacked = struct.pack(">h", unpacked_int)
        assert len(repacked) == 2
        assert not space.is_possible(repacked != byts)


def test_unpack_char(space) -> None:
    """unpack 'c' format on SymbolicBytes - should produce symbolic bytes (chunk)."""
    byts = proxy_for_type(bytes, "byts")
    with ResumedTracing():
        space.add(len(byts) == 1)
        unpacked = struct.unpack(">c", byts)
        chunk = unpacked[0]
        assert len(unpacked) == 1
    assert not isinstance(chunk, bytes)
    with ResumedTracing():
        assert space.is_possible(len(chunk) == 1)


def test_unpack_multiple_ints(space) -> None:
    """unpack '>2h' on SymbolicBytes - should produce tuple of symbolic ints."""
    byts = proxy_for_type(bytes, "byts")
    with ResumedTracing():
        space.add(len(byts) == 4)
        unpacked = struct.unpack(">2h", byts)
        assert len(unpacked) == 2
        a, b = unpacked[0], unpacked[1]
    assert not isinstance(a, int)
    assert not isinstance(b, int)
    with ResumedTracing():
        assert space.is_possible(a == 42)
        assert space.is_possible(-32768 <= b <= 32767)


def test_unpack_from(space) -> None:
    """unpack_from on SymbolicBytes with offset - should produce symbolic int."""
    byts = proxy_for_type(bytes, "byts")
    with ResumedTracing():
        space.add(len(byts) == 4)
        unpacked = struct.unpack_from(">h", byts, 2)
        assert len(unpacked) == 1
    assert not isinstance(unpacked[0], int)
    with ResumedTracing():
        assert space.is_possible(unpacked[0] == 42)


def test_iter_unpack(space) -> None:
    """iter_unpack on SymbolicBytes - should yield tuples of symbolic ints."""
    byts = proxy_for_type(bytes, "byts")
    with ResumedTracing():
        space.add(len(byts) == 4)
        items = list(struct.iter_unpack(">h", byts))
        a, b = items[0][0], items[1][0]
        assert len(items) == 2
    assert not isinstance(a, int)
    assert not isinstance(b, int)
    with ResumedTracing():
        assert space.is_possible(a == 42)
        assert space.is_possible(-32768 <= b <= 32767)


def test_pack_into(space) -> None:
    """pack_into into SymbolicByteArray - should retain symbolic state."""
    buf = proxy_for_type(bytearray, "buf")
    x = proxy_for_type(int, "x")
    with ResumedTracing():
        space.add(len(buf) == 4)
        space.add(x >= -32768)
        space.add(x < 32768)
        struct.pack_into(">h", buf, 0, x)
        b0, b1 = buf[0], buf[1]
    assert not isinstance(b0, int)
    assert not isinstance(b1, int)


@pytest.mark.parametrize(
    "fn,args,exc_type,exc_message",
    [
        (
            struct.unpack,
            (1, b"a"),
            TypeError,
            "Struct() argument 1 must be a str or bytes object, not int",
        ),
        (
            struct.unpack,
            ("i", 1),
            TypeError,
            "a bytes-like object is required, not 'int'",
        ),
        (
            struct.pack_into,
            ("i", 1, 0, 1),
            TypeError,
            "argument must be read-write bytes-like object, not int",
        ),
        (
            struct.unpack_from,
            ("i", 1, 0),
            TypeError,
            "a bytes-like object is required, not 'int'",
        ),
        (
            struct.iter_unpack,
            ("b", 1),
            TypeError,
            "a bytes-like object is required, not 'int'",
        ),
        (
            struct.pack_into,
            ("i", bytearray(4), "0", 1),
            TypeError,
            "'str' object cannot be interpreted as an integer",
        ),
        (
            struct.unpack_from,
            ("i", b"abcd", "0"),
            TypeError,
            "'str' object cannot be interpreted as an integer",
        ),
    ],
)
def test_type_errors_match_struct(fn, args, exc_type, exc_message) -> None:
    with pytest.raises(exc_type, match=re.escape(exc_message)):
        fn(*args)


def test_positional_only_signature_rejections():
    with pytest.raises(TypeError):
        _calcsize(fmt="i")
    with pytest.raises(TypeError):
        _pack(fmt="i")
    with pytest.raises(TypeError):
        _unpack(fmt="i", buffer=b"1234")
    with pytest.raises(TypeError):
        _iter_unpack(fmt="i", buffer=b"1234")


def test_unpack_from_keyword_shape_matches_struct():
    # buffer/offset are keyword-acceptable in CPython; format is positional-only.
    assert _unpack_from("i", buffer=b"1234") == struct.unpack_from("i", buffer=b"1234")
    assert _unpack_from("i", b"1234", offset=0) == struct.unpack_from(
        "i", b"1234", offset=0
    )
    with pytest.raises(TypeError):
        _unpack_from(format="i", buffer=b"1234")


def test_invalid_format_validated_before_pack_arity() -> None:
    """Like C struct.pack, reject bad format codes before counting arguments."""
    msg = "bad char in struct format"
    with pytest.raises(struct.error, match=re.escape(msg)):
        struct.pack("a")
    with pytest.raises(struct.error, match=re.escape(msg)):
        _pack("a")
    buf = bytearray(4)
    with pytest.raises(struct.error, match=re.escape(msg)):
        struct.pack_into("a", buf, 0)
    with pytest.raises(struct.error, match=re.escape(msg)):
        _pack_into("a", buf, 0)
