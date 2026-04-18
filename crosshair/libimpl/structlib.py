"""
Symbolic implementation of Python's struct module.

Format strings and floats are realized. Symbolic support focuses on
integers (pack/unpack) and byte buffers (unpack).
"""

import struct
from operator import index
from typing import Any, List, Literal, Tuple, Union

from crosshair.core import deep_realize, realize, register_patch
from crosshair.libimpl.builtinslib import SymbolicByteArray, SymbolicBytes
from crosshair.tracers import NoTracing, ResumedTracing
from crosshair.util import name_of_type

_MISSING = object()


def _is_symbolic_buffer(buf: object) -> bool:
    with NoTracing():
        return isinstance(buf, (SymbolicBytes, SymbolicByteArray))


def _normalize_format_for_parse(fmt: Union[str, bytes]) -> str:
    """Match CPython: bytes format strings are interpreted like Latin-1."""
    if isinstance(fmt, bytes):
        # TODO: likely less work for CrossHair if we normalized to bytes instead
        return fmt.decode("latin-1")
    return fmt


# Format char -> (size in bytes for standard formats, signed for ints)
# For @ native, sizes vary; we use struct.calcsize for those.
_STANDARD_SIZES: dict[str, Tuple[int, bool]] = {
    "x": (1, False),  # pad
    "c": (1, False),  # char
    "b": (1, True),
    "B": (1, False),
    "?": (1, False),  # bool
    "h": (2, True),
    "H": (2, False),
    "i": (4, True),
    "I": (4, False),
    "l": (4, True),
    "L": (4, False),
    "q": (8, True),
    "Q": (8, False),
    "e": (2, False),  # half float
    "f": (4, False),
    "d": (8, False),
    "n": (0, True),  # ssize_t - native only
    "N": (0, False),  # size_t - native only
    "s": (-1, False),  # char[] - count required
    "p": (-1, False),  # pascal string - count required
    "P": (0, False),  # pointer - native only
    "F": (8, False),  # float complex
    "D": (16, False),  # double complex
}


def _parse_format(fmt: str) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Parse struct format string. Returns (byte_order_prefix, [(format_char, count), ...]).
    """
    fmt = fmt.strip()
    if not fmt:
        raise struct.error("bad char in struct format")
    if isinstance(fmt, bytes):
        fmt = fmt.decode("latin-1")
    idx = 0
    prefix = "@"
    if fmt[0] in "@=<>!":
        prefix = fmt[0]
        idx = 1
    items: List[Tuple[str, int]] = []
    while idx < len(fmt):
        while idx < len(fmt) and fmt[idx] in " \t\n\r\v":
            idx += 1
        if idx >= len(fmt):
            break
        count_str = ""
        while idx < len(fmt) and fmt[idx].isdigit():
            count_str += fmt[idx]
            idx += 1
        count = int(count_str) if count_str else 1
        if idx >= len(fmt):
            raise struct.error("bad char in struct format")
        fc = fmt[idx]
        idx += 1
        if fc in "sp":
            if count == 0 and not count_str:
                count = 1
            items.append((fc, count))
        elif fc in _STANDARD_SIZES.keys():
            for _ in range(count):
                items.append((fc, 1))
        else:
            raise struct.error("bad char in struct format")
    return (prefix, items)


def _get_item_size(fc: str, count: int, prefix: str) -> int:
    """Get size in bytes for a format item."""
    if fc in _STANDARD_SIZES:
        size, _ = _STANDARD_SIZES[fc]
        if size == 0:
            return struct.calcsize(prefix + fc)
        if size == -1:
            return count
        return size * count
    raise struct.error("bad char in struct format")


def _struct_items_total_size(prefix: str, items: List[Tuple[str, int]]) -> int:
    """Validate format and return total packed size."""
    total = 0
    for fc, count in items:
        total += _get_item_size(fc, count, prefix)
    return total


def _pack_arg_count(items: List[Tuple[str, int]]) -> int:
    """Number of values struct.pack / pack_into expect (padding 'x' has no argument)."""
    return sum(1 for fc, _ in items if fc != "x")


def _is_int_format(fc: str) -> bool:
    return fc in "bBhHiIlLqQnN"


def _is_float_format(fc: str) -> bool:
    return fc in "efdFD"


def _byteorder_for_int(prefix: str) -> Literal["little", "big"]:
    return "little" if prefix == "<" else "big"


def _calcsize(fmt: Union[str, bytes], /) -> int:
    _check_format_arg(fmt)
    return struct.calcsize(realize(fmt))


def _check_format_arg(fmt: object) -> None:
    if not isinstance(fmt, (str, bytes)):
        raise TypeError(
            "Struct() argument 1 must be a str or bytes object, not "
            + name_of_type(type(fmt))
        )


def _check_readable_buffer_arg(buffer: object) -> None:
    if not isinstance(buffer, (bytes, bytearray, SymbolicBytes, SymbolicByteArray)):
        raise TypeError(
            f"a bytes-like object is required, not '{name_of_type(type(buffer))}'"
        )


def _check_writable_buffer_arg(buffer: object) -> None:
    if not isinstance(buffer, (bytearray, SymbolicByteArray)):
        raise TypeError(
            f"argument must be read-write bytes-like object, not {name_of_type(type(buffer))}"
        )


def _pack_subfmt(prefix: str, fc: str, count: int) -> str:
    if fc in "sp":
        return prefix + str(count) + fc
    if count != 1:
        return prefix + str(count) + fc
    return prefix + fc


def _pack(fmt, /, *args) -> Union[bytes, SymbolicBytes]:
    _check_format_arg(fmt)
    with NoTracing():
        fmt_arg = realize(fmt)
        # Match CPython Struct errors (e.g. UnicodeEncodeError) before our parse/arity checks.
        struct.calcsize(fmt_arg)
        fmt_s = _normalize_format_for_parse(fmt_arg)
        prefix, items = _parse_format(fmt_s)
        _struct_items_total_size(prefix, items)
        parts: List[Union[bytes, SymbolicBytes]] = []
        expected_args = _pack_arg_count(items)
        if len(args) != expected_args:
            raise struct.error(
                f"pack expected {expected_args} items for packing (got {len(args)})"
            )
        byteorder = _byteorder_for_int(prefix)
        argi = 0
        for fc, count in items:
            if fc == "x":
                parts.append(b"\x00" * count)
                continue
            val = args[argi]
            argi += 1
            if fc == "c":
                val = realize(val)
                b = val if isinstance(val, (bytes, bytearray)) else bytes([val])
                if len(b) != 1:
                    raise struct.error(
                        "char format requires a bytes object of length 1"
                    )
                parts.append(bytes(b))
            elif fc == "?":
                with ResumedTracing():
                    parts.append(val.to_bytes(1, byteorder, signed=False))
            elif _is_int_format(fc):
                size = _get_item_size(fc, count, prefix)
                signed = fc in "bhlqin"
                try:
                    with ResumedTracing():
                        parts.append(val.to_bytes(size, byteorder, signed=signed))
                except OverflowError as e:
                    raise struct.error(str(e)) from e
            else:
                subfmt = _pack_subfmt(prefix, fc, count)
                parts.append(struct.pack(subfmt, realize(val)))
        result: List[int] = []
        has_symbolic = False
        for p in parts:
            if isinstance(p, SymbolicBytes):
                result.extend(p)
                has_symbolic = True
            else:
                result.extend(p)
        return SymbolicBytes(result) if has_symbolic else bytes(result)


def _unpack(
    fmt: Union[str, bytes],
    buffer: Union[bytes, bytearray, SymbolicBytes, SymbolicByteArray],
    /,
) -> Tuple[Any, ...]:
    with NoTracing():
        fmt_arg = deep_realize(fmt)
        if not _is_symbolic_buffer(buffer):
            return struct.unpack(fmt_arg, buffer)
        fmt_s = _normalize_format_for_parse(fmt_arg)
        prefix, items = _parse_format(fmt_s)
        need = _struct_items_total_size(prefix, items)
        with ResumedTracing():
            buf_len = len(buffer)
            if buf_len < need:
                raise struct.error(f"unpack requires a buffer of {need} bytes")
        offset = 0
        results: List[Any] = []
        byteorder = _byteorder_for_int(prefix)
        for fc, count in items:
            size = _get_item_size(fc, count, prefix)
            with ResumedTracing():
                chunk = buffer[offset : offset + size]
            offset += size
            if fc == "x":
                continue
            if fc == "c":
                results.append(chunk)
            elif fc == "?":
                chunk = deep_realize(chunk)
                subfmt = _pack_subfmt(prefix, "?", 1)
                results.append(struct.unpack(subfmt, chunk)[0])
            elif _is_int_format(fc):
                signed = fc in "bhlqin"
                with ResumedTracing():
                    unpacked_int = int.from_bytes(chunk, byteorder, signed=signed)
                results.append(unpacked_int)
            elif _is_float_format(fc) or fc in "sp":
                chunk = deep_realize(chunk)
                subfmt = _pack_subfmt(prefix, fc, count)
                results.append(_unpack(subfmt, chunk)[0])
            else:
                chunk = deep_realize(chunk)
                subfmt = _pack_subfmt(prefix, fc, count)
                results.append(_unpack(subfmt, chunk)[0])
        return tuple(results)


# def pack_into(fmt: str | bytes, buffer: WriteableBuffer, offset: int, /, *v: Any) -> None:
def _pack_into(
    fmt: Union[str, bytes],
    buffer=_MISSING,
    offset=_MISSING,
    *args: Any,
) -> None:
    _check_format_arg(fmt)
    fmt_arg = realize(fmt)
    with NoTracing():
        struct.calcsize(fmt_arg)
        fmt_s = _normalize_format_for_parse(fmt_arg)
        prefix, items = _parse_format(fmt_s)
    if buffer is _MISSING:
        raise struct.error("pack_into expected buffer argument")
    if offset is _MISSING:
        raise struct.error("pack_into expected offset argument")
    _check_writable_buffer_arg(buffer)
    with NoTracing():
        _struct_items_total_size(prefix, items)
        expected_args = _pack_arg_count(items)
        if len(args) != expected_args:
            raise struct.error(
                f"pack_into expected {expected_args} items for packing (got {len(args)})"
            )
    packed = struct.pack(fmt_arg, *args)
    size = len(packed)
    buffer[offset : offset + size] = packed


def _unpack_from(
    fmt: Union[str, bytes],
    /,
    buffer: Union[bytes, bytearray, SymbolicBytes, SymbolicByteArray],
    offset: int = 0,
) -> Tuple[Any, ...]:
    _check_format_arg(fmt)
    fmt_arg = realize(fmt)
    with NoTracing():
        # Match CPython: Struct/calcsize reject non-ASCII str formats with UnicodeEncodeError
        # before buffer or offset checks.
        struct.calcsize(fmt_arg)
    _check_readable_buffer_arg(buffer)
    offset = index(offset)
    return struct.unpack(fmt_arg, buffer[offset:])


def _iter_unpack(
    fmt: Union[str, bytes],
    buffer: Union[bytes, bytearray, SymbolicBytes, SymbolicByteArray],
    /,
):
    _check_format_arg(fmt)
    with NoTracing():
        fmt_arg = deep_realize(fmt)
        _normalize_format_for_parse(fmt_arg)
        size = _calcsize(fmt_arg)
    _check_readable_buffer_arg(buffer)
    with NoTracing():
        offset = 0
        buf_len = len(buffer)
        while offset + size <= buf_len:
            if _is_symbolic_buffer(buffer):
                yield _unpack(fmt_arg, buffer[offset : offset + size])
            else:
                yield struct.unpack(fmt_arg, buffer[offset : offset + size])
            offset += size


def make_registrations() -> None:
    register_patch(struct.calcsize, _calcsize)
    register_patch(struct.pack, _pack)
    register_patch(struct.unpack, _unpack)
    register_patch(struct.pack_into, _pack_into)
    register_patch(struct.unpack_from, _unpack_from)
    register_patch(struct.iter_unpack, _iter_unpack)
