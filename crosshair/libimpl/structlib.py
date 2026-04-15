"""
Symbolic implementation of Python's struct module.

Format strings and floats are realized. Symbolic support focuses on
integers (pack/unpack) and byte buffers (unpack).
"""

import re
import struct
from typing import Any, List, Literal, Tuple, Union

from crosshair.core import deep_realize, realize, register_patch
from crosshair.libimpl.builtinslib import SymbolicByteArray, SymbolicBytes
from crosshair.tracers import NoTracing, ResumedTracing


def _is_symbolic_buffer(buf: object) -> bool:
    with NoTracing():
        return isinstance(buf, (SymbolicBytes, SymbolicByteArray))


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

# Byte order chars that use standard sizes (no alignment)
_STANDARD_PREFIXES = frozenset("<>=!")


def _parse_format(fmt: str) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Parse struct format string. Returns (byte_order_prefix, [(format_char, count), ...]).
    """
    fmt = fmt.strip()
    if not fmt:
        raise struct.error("bad char in struct format")
    idx = 0
    prefix = "@"
    if fmt[0] in "@=<>!":
        prefix = fmt[0]
        idx = 1
    items: List[Tuple[str, int]] = []
    while idx < len(fmt):
        # Skip whitespace
        while idx < len(fmt) and fmt[idx] in " \t\n\r\v":
            idx += 1
        if idx >= len(fmt):
            break
        # Read optional repeat count
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
            # s and p require a count
            if count == 0 and not count_str:
                count = 1
            items.append((fc, count))
        else:
            for _ in range(count):
                items.append((fc, 1))
    return (prefix, items)


def _get_item_size(fc: str, count: int, prefix: str) -> int:
    """Get size in bytes for a format item."""
    if fc in "nNP" and prefix != "@":
        raise struct.error("format " + repr(fc) + " requires native byte order")
    if fc in _STANDARD_SIZES:
        size, _ = _STANDARD_SIZES[fc]
        if size == 0:
            # Native size - use struct.calcsize for single item
            return struct.calcsize(prefix + fc)
        if size == -1:
            return count  # s or p
        return size * count
    raise struct.error("bad char in struct format " + repr(fc))


def _is_int_format(fc: str) -> bool:
    return fc in "bBhHiIlLqQnN?"


def _is_float_format(fc: str) -> bool:
    return fc in "efdFD"


def _calcsize(fmt: str) -> int:
    return struct.calcsize(realize(fmt))


def _pack(fmt: str, *args: Any) -> Union[bytes, SymbolicBytes]:
    with NoTracing():
        fmt = realize(fmt)
        prefix, items = _parse_format(fmt)
        parts: List[Union[bytes, SymbolicBytes]] = []
        if len(args) != len(items):
            raise struct.error(
                f"pack expected {len(items)} items for packing (got {len(args)})"
            )
        for (fc, count), val in zip(items, args):
            if fc == "x":
                parts.append(b"\x00" * count)
                continue
            if fc == "c":
                val = realize(val)
                b = val if isinstance(val, (bytes, bytearray)) else bytes([val])
                if len(b) != 1:
                    raise struct.error(
                        "char format requires a bytes object of length 1"
                    )
                parts.append(bytes(b))
            elif fc == "?":
                parts.append(val.to_bytes(1, "big", signed=False))
            elif _is_int_format(fc):
                # Use int.to_bytes for symbolic int support (needs tracing for symbolic)
                size = _get_item_size(fc, count, prefix)
                signed = fc in "bhlqin"
                byteorder = "little" if prefix == "<" else "big"
                try:
                    with ResumedTracing():
                        parts.append(val.to_bytes(size, byteorder, signed=signed))
                except OverflowError as e:
                    raise struct.error(str(e)) from e
            else:
                # Realize and delegate
                parts.append(struct.pack(f"{prefix}{fc},{count}", realize(val)))
        # Concatenate
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
    fmt: str, buffer: Union[bytes, bytearray, SymbolicBytes, SymbolicByteArray]
) -> Tuple[Any, ...]:
    with NoTracing():
        fmt = str(deep_realize(fmt))
        if not _is_symbolic_buffer(buffer):
            return struct.unpack(fmt, buffer)
        # Symbolic buffer: use indexing to avoid buffer protocol (which realizes)
        prefix, items = _parse_format(fmt)
        offset = 0
        results: List[Any] = []
        byteorder: Literal["little", "big"] = "little" if prefix == "<" else "big"
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
                results.append(struct.unpack("?", chunk)[0])
            elif _is_int_format(fc):
                signed = fc in "bhlqin"
                with ResumedTracing():
                    unpacked_int = int.from_bytes(chunk, byteorder, signed=signed)
                results.append(unpacked_int)
            elif _is_float_format(fc) or fc in "sp":
                chunk = deep_realize(chunk)
                sub_fmt = prefix + (f"{count}{fc}" if fc in "sp" else fc * count)
                results.append(_unpack(sub_fmt, chunk)[0])
            else:
                chunk = deep_realize(chunk)
                sub_fmt = prefix + fc * count
                results.append(_unpack(sub_fmt, chunk)[0])
        return tuple(results)


def _pack_into(
    fmt: str, buffer: Union[bytearray, SymbolicByteArray], offset: int, *args: Any
) -> None:
    packed = struct.pack(fmt, *args)
    size = len(packed)
    buffer[offset : offset + size] = packed


def _unpack_from(
    fmt: str,
    buffer: Union[bytes, bytearray, SymbolicBytes, SymbolicByteArray],
    offset: int = 0,
) -> Tuple[Any, ...]:
    return struct.unpack(fmt, buffer[offset:])


def _iter_unpack(
    fmt: str, buffer: Union[bytes, bytearray, SymbolicBytes, SymbolicByteArray]
):
    with NoTracing():
        fmt = deep_realize(fmt)
        size = _calcsize(fmt)
        offset = 0
        buf_len = len(buffer)
        while offset + size <= buf_len:
            if _is_symbolic_buffer(buffer):
                yield _unpack(fmt, buffer[offset : offset + size])
            else:
                yield struct.unpack(fmt, buffer[offset : offset + size])
            offset += size


def make_registrations() -> None:
    register_patch(struct.calcsize, _calcsize)
    register_patch(struct.pack, _pack)
    register_patch(struct.unpack, _unpack)
    register_patch(struct.pack_into, _pack_into)
    register_patch(struct.unpack_from, _unpack_from)
    register_patch(struct.iter_unpack, _iter_unpack)
