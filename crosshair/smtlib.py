import math
import re
import struct

# Python refuses decimal conversions beyond a configurable digit count
# (no smaller than 640); convert at most this many digits at a time:
_MAX_DIRECT_DECIMAL_DIGITS = 600
_MAX_DIRECT_DECIMAL = 10**_MAX_DIRECT_DECIMAL_DIGITS


def _decimal_string(magnitude: int) -> str:
    if magnitude < _MAX_DIRECT_DECIMAL:
        return magnitude.__str__()
    low_digits = magnitude.bit_length() * 3 // 20
    high, low = divmod(magnitude, 10**low_digits)
    return _decimal_string(high) + _decimal_string(low).zfill(low_digits)


def _decimal_magnitude(digits: str) -> int:
    if len(digits) <= _MAX_DIRECT_DECIMAL_DIGITS:
        return int(digits)
    low_digits = len(digits) // 2
    high, low = digits[:-low_digits], digits[-low_digits:]
    return _decimal_magnitude(high) * 10**low_digits + _decimal_magnitude(low)


def format_int_numeral(value: int) -> str:
    """Format an int (or int subtype, such as an enum) in decimal, at any size."""
    value = value.__index__()
    return ("-" if value < 0 else "") + _decimal_string(abs(value))


def parse_int_numeral(digits: str) -> int:
    """Parse a decimal integer, at any size."""
    if digits.startswith("-"):
        return -_decimal_magnitude(digits[1:])
    return _decimal_magnitude(digits)


_SL2_PARSE_FLOAT_RE = re.compile(r"\(fp #(b\d+) #(b\d+) #(x.+)\)")
LITERAL_CONSTS = {
    "(_ -zero 11 53)": -0.0,
    "(_ +zero 11 53)": 0.0,
    "(_ -oo 11 53)": -math.inf,
    "(_ +oo 11 53)": math.inf,
    "(_ NaN 11 53)": math.nan,
}


def parse_smtlib_literal(input: str):
    literal_const = LITERAL_CONSTS.get(input, None)
    if literal_const is not None:
        return literal_const
    match = _SL2_PARSE_FLOAT_RE.fullmatch(input)
    if match:
        sign, exp, significand = (int("0" + g, base=0) for g in match.groups())
        return struct.unpack(
            "d", struct.pack("Q", sign << 63 | exp << 52 | significand)
        )[0]
