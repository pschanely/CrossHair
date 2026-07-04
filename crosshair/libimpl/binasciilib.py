import binascii
from typing import Dict, Iterable, Tuple, Union

from crosshair.core import deep_realize, register_patch
from crosshair.libimpl.builtinslib import _ALL_BYTES_TYPES, SymbolicBytes
from crosshair.tracers import NoTracing
from crosshair.util import name_of_type

_ORD_OF_NEWLINE = ord("\n")
_ORD_OF_EQUALS = ord("=")

# The standard base64 alphabet (6-bit value -> output character).  Named
# binascii.BASE64_ALPHABET in 3.15+, but we keep our own copy so the default
# argument works on older Pythons too.
_STD_BASE64_ALPHABET = (
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
)


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


def make_bytes(arg: object) -> Union[bytes, bytearray, memoryview]:
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


def _encode_map_from_alphabet(alphabet) -> Dict[Tuple[int, int], int]:
    """Build a ``_remap`` table (6-bit value range -> output char code) for a
    custom base64 ``alphabet``.  Consecutive values whose characters are also
    consecutive are collapsed into a single range so the mapping stays a small
    set of arithmetic offsets -- which keeps it symbolic-friendly, exactly like
    the standard ``_ENCODE_BASE64_MAP``."""
    alphabet = bytes(alphabet)
    if len(alphabet) != 64:
        raise ValueError("alphabet must have length 64")
    table = {}
    v = 0
    while v < 64:
        start = v
        while v + 1 < 64 and alphabet[v + 1] == alphabet[start] + (v + 1 - start):
            v += 1
        table[(start, v)] = alphabet[start]
        v += 1
    return table


def _b2a_base64(
    data, /, *, newline=True, wrapcol=0, padded=True, alphabet=_STD_BASE64_ALPHABET
):  # encode
    if not isinstance(data, _ALL_BYTES_TYPES):
        raise TypeError(
            f"a bytes-like object is required, not '{name_of_type(type(data))}'"
        )
    if alphabet is _STD_BASE64_ALPHABET or bytes(alphabet) == _STD_BASE64_ALPHABET:
        encode_map = _ENCODE_BASE64_MAP
    else:
        encode_map = _encode_map_from_alphabet(alphabet)
    output_ints, _padding_count = _remap(
        encode_map, _bit_chunk_iter(data, input_bit_size=8, output_bit_size=6)
    )
    if padded:
        spillover = len(output_ints) % 4
        if spillover > 0:
            output_ints.extend([_ORD_OF_EQUALS for _ in range(4 - spillover)])
    if wrapcol and output_ints:
        # 3.15+: insert a newline after every `wrapcol` output characters.
        # CPython first rounds the width down to a whole number of 4-char groups
        # (minimum 4) and never emits a trailing wrap-newline; see
        # Modules/binascii.c:binascii_b2a_base64_impl.
        width = 4 if wrapcol < 4 else (wrapcol // 4) * 4
        wrapped = []
        for idx, ch in enumerate(output_ints):
            if idx and idx % width == 0:
                wrapped.append(_ORD_OF_NEWLINE)
            wrapped.append(ch)
        output_ints = wrapped
    if newline:
        output_ints.append(_ORD_OF_NEWLINE)
    return SymbolicBytes(output_ints)


def _a2b_base64(
    data,
    /,
    *,
    strict_mode: bool = False,
    padded: bool = True,
    alphabet=_STD_BASE64_ALPHABET,
    ignorechars=b"",
    canonical: bool = False,
):  # decode
    # 3.15+ forwards padded/alphabet/ignorechars/canonical through the base64
    # module; older Pythons never pass them, so the extra keywords keep defaults
    # that reproduce the pre-3.15 behavior exactly.
    if canonical or ignorechars:
        # Rare strict-validation paths (RFC 4648 canonical padding bits, and
        # skip-these-characters filtering).  Defer to the real implementation on
        # realized data; the common paths below stay fully symbolic.
        with NoTracing():
            real_alphabet = bytes(deep_realize(alphabet))
            kwargs: dict = {"strict_mode": strict_mode, "padded": padded}
            if real_alphabet != _STD_BASE64_ALPHABET:
                kwargs["alphabet"] = real_alphabet
            if ignorechars:
                kwargs["ignorechars"] = bytes(deep_realize(ignorechars))
            if canonical:
                kwargs["canonical"] = True
            return binascii.a2b_base64(bytes(deep_realize(data)), **kwargs)
    data = make_bytes(data)
    if alphabet is _STD_BASE64_ALPHABET or bytes(alphabet) == _STD_BASE64_ALPHABET:
        decode_map = _DECODE_BASE64_MAP
    else:
        decode_map = _reverse_map(_encode_map_from_alphabet(alphabet))
    input_ints, padding_count = _remap(
        decode_map,
        data,
        error_type=binascii.Error if strict_mode else None,
        error_message="Only base64 data is allowed" if strict_mode else None,
    )

    data_char_count = len(input_ints)
    uneven = data_char_count % 4 != 0
    if uneven:
        if data_char_count % 4 == 1:
            raise binascii.Error(
                f"Invalid base64-encoded string: number of data characters ({data_char_count}) cannot be 1 more than a multiple of 4"
            )
        if padded:
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
