import math
import re
import struct

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
