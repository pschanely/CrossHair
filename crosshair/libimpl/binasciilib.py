import binascii
from collections.abc import ByteString
from functools import partial
from typing import Dict, Iterable, Tuple

from crosshair.core import register_patch
from crosshair.libimpl.builtinslib import _ALL_BYTES_TYPES, SymbolicBytes
from crosshair.util import name_of_type

_ORD_OF_NEWLINE = ord("\n")
_ORD_OF_EQUALS = ord("=")


def _bit_chunk_iter(
    seq: Iterable[int], input_bit_size: int, output_bit_size: int
) -> Iterable[int]:
    """Turns a sequence of N-bit integers into a sequence of M-bit integers"""
    accumulated = 0
    bits_accumulated = 0
    input_size = 1 << input_bit_size
    for chunk in seq:
        accumulated = accumulated * input_size + chunk
        bits_accumulated += input_bit_size
        while bits_accumulated >= output_bit_size:
            remaining_bits = bits_accumulated - output_bit_size
            if remaining_bits == 0:
                yield accumulated
                accumulated = bits_accumulated = 0
            else:
                top, bottom = divmod(accumulated, 1 << remaining_bits)
                yield top
                accumulated = bottom
                bits_accumulated = remaining_bits
    if bits_accumulated > 0:
        zero_bits = output_bit_size - bits_accumulated
        yield accumulated * (1 << zero_bits)


def _remap(
    table: Dict[Tuple[int, int], int],
    input_ints: Iterable[int],
    error_type=None,
    error_message=None,
):
    output_ints = []
    padding = 0
    for ch in input_ints:
        for (minval, maxval), target_start in table.items():
            if all([minval <= ch, ch <= maxval]):
                if minval != 0:
                    ch -= minval
                if target_start != 0:
                    ch += target_start
                output_ints.append(ch)
                if padding != 0:
                    raise binascii.Error("Discontinuous padding not allowed")
                break
        else:
            if ch == _ORD_OF_EQUALS:
                padding += 1
            elif error_message:
                raise (error_type or binascii.Error)(error_message)
    return output_ints, padding


def _reverse_map(table: Dict[Tuple[int, int], int]):
    ret = {}
    for (minval, maxval), target_start in table.items():
        ret[(target_start, target_start + maxval - minval)] = minval
    return ret


_DECODE_BASE64_MAP = {
    (ord("A"), ord("Z")): 0,
    (ord("a"), ord("z")): 26,
    (ord("0"), ord("9")): 52,
    (ord("+"), ord("+")): 62,
    (ord("/"), ord("/")): 63,
}
_ENCODE_BASE64_MAP = _reverse_map(_DECODE_BASE64_MAP)

_DECODE_MAPPER_BASE64 = partial(_remap, _DECODE_BASE64_MAP)
_DECODE_MAPPER_BASE64_STRICT = partial(
    _remap,
    _DECODE_BASE64_MAP,
    error_type=binascii.Error,
    error_message="Only base64 data is allowed",
)
_ENCODE_MAPPER_BASE64 = partial(_remap, _ENCODE_BASE64_MAP)


def make_bytes(arg: object) -> ByteString:
    if isinstance(arg, (bytes, bytearray, memoryview)):
        return arg
    if isinstance(arg, str):
        try:
            return arg.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError("string argument should contain only ASCII characters")
    else:
        raise TypeError(
            f"a bytes-like object is required, not '{name_of_type(type(arg))}'"
        )


def _b2a_base64(data, /, *, newline=True):  # encode
    if not isinstance(data, _ALL_BYTES_TYPES):
        raise TypeError(
            f"a bytes-like object is required, not '{name_of_type(type(data))}'"
        )
    output_ints, _padding_count = _ENCODE_MAPPER_BASE64(
        _bit_chunk_iter(data, input_bit_size=8, output_bit_size=6)
    )
    spillover = len(output_ints) % 4
    if spillover > 0:
        output_ints.extend([_ORD_OF_EQUALS for _ in range(4 - spillover)])
    if newline:
        output_ints.append(_ORD_OF_NEWLINE)
    return SymbolicBytes(output_ints)


def _a2b_base64(data, /, *, strict_mode: bool = False):
    data = make_bytes(data)
    input_ints, padding_count = (
        _DECODE_MAPPER_BASE64_STRICT if strict_mode else _DECODE_MAPPER_BASE64
    )(data)

    data_char_count = len(input_ints)
    uneven = data_char_count % 4 != 0
    if uneven:
        if data_char_count % 4 == 1:
            raise binascii.Error(
                f"Invalid base64-encoded string: number of data characters ({len(input_ints)}) cannot be 1 more than a multiple of 4"
            )
        expected_padding = 4 - (data_char_count % 4)
        if padding_count < expected_padding:
            raise binascii.Error("Incorrect padding")
        elif strict_mode and padding_count > expected_padding:
            raise binascii.Error("Excess data after padding")
    output_ints = [
        byt for byt in _bit_chunk_iter(input_ints, input_bit_size=6, output_bit_size=8)
    ]
    if uneven:
        output_ints = output_ints[:-1]
    return SymbolicBytes(output_ints)


def make_registrations():
    # NOTE: these power the base64 module, so we get that for free
    register_patch(binascii.b2a_base64, _b2a_base64)
    register_patch(binascii.a2b_base64, _a2b_base64)
