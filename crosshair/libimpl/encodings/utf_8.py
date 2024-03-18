import codecs
from typing import List, Optional, Tuple, Union

from crosshair.libimpl.builtinslib import SymbolicBytes
from crosshair.libimpl.encodings._encutil import (
    ChunkError,
    MidChunkError,
    StemEncoder,
    UnexpectedEndError,
)


def _encode_codepoint(codepoint: int) -> Tuple[int, ...]:
    if codepoint <= 0b01111111:
        return (codepoint,)
    elif codepoint <= 0b111_11111111:
        return (
            (0b11000000 + ((codepoint >> 6) & 0b00011111)),
            (0b10000000 + (codepoint & 0b00111111)),
        )
    elif codepoint <= 0b11111111_11111111:
        return (
            (0b11100000 + ((codepoint >> 12) & 0b00001111)),
            (0b10000000 + ((codepoint >> 6) & 0b00111111)),
            (0b10000000 + (codepoint & 0b00111111)),
        )
    else:
        return (
            (0b11110000 + ((codepoint >> 18) & 0b00000111)),
            (0b10000000 + ((codepoint >> 12) & 0b00111111)),
            (0b10000000 + ((codepoint >> 6) & 0b00111111)),
            (0b10000000 + (codepoint & 0b00111111)),
        )


class Utf8StemEncoder(StemEncoder):
    encoding_name = "utf-8"

    @classmethod
    def _encode_chunk(
        cls, string: str, start: int
    ) -> Tuple[Union[bytes, SymbolicBytes], int, Optional[ChunkError]]:
        byte_ints: List[int] = []
        for ch in string[start:]:
            byte_ints.extend(_encode_codepoint(ord(ch)))
        return (SymbolicBytes(byte_ints), len(string), None)

    @classmethod
    def _decode_chunk(
        cls, byts: bytes, start: int
    ) -> Tuple[str, int, Optional[ChunkError]]:
        num_bytes = len(byts)
        byt = byts[start]
        end = start + 1
        if byt >= 0b11000000:
            end += 1
            if byt >= 0b11100000:
                end += 1
                if byt >= 0b11110000:
                    if byt > 0b11110111:
                        return ("", start, MidChunkError(f"can't decode byte"))
                    end += 1
                    cp = byt & 0b00000111
                    mincp, maxcp = 0x10000, 0x10FFFF
                else:
                    if byt > 0b11101111:
                        return ("", start, MidChunkError(f"can't decode byte"))
                    cp = byt & 0b00001111
                    mincp, maxcp = 0x0800, 0xFFFF
            else:
                if byt > 0b11011111:
                    return ("", start, MidChunkError(f"can't decode byte"))
                cp = byt & 0b00011111
                mincp, maxcp = 0x0080, 0x07FF
        else:
            cp = byt
            mincp, maxcp = 0, 0x007F
        if end > num_bytes:
            return ("", start, UnexpectedEndError())
        for idx in range(start + 1, end):
            byt = byts[idx]
            if 0b10_000000 <= byt <= 0b10_111111:
                cp = (cp * 64) + (byts[idx] - 0b10_000000)
            else:
                return ("", start, MidChunkError(f"can't decode byte"))
        if mincp <= cp <= maxcp:
            return (chr(cp), end, None)
        else:
            return ("", start, MidChunkError(f"invalid start byte"))


def getregentry() -> codecs.CodecInfo:
    return Utf8StemEncoder.getregentry()
