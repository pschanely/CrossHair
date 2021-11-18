import codecs

from typing import List, Tuple

from crosshair.core import realize, NoTracing
from crosshair.libimpl.builtinslib import SymbolicBytes


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


def _encode(input: str, errors: str) -> Tuple[bytes, int]:
    byte_ints: List[int] = []
    for ch in input:
        byte_ints.extend(_encode_codepoint(ord(ch)))
    return (SymbolicBytes(byte_ints), len(input))  # type: ignore


def _decode(input: bytes, errors="strict") -> Tuple[str, int]:
    return codecs.utf_8_decode(realize(input), errors, True)  # type: ignore


class IncrementalEncoder(codecs.IncrementalEncoder):
    def encode(self, input, final=False):
        return _encode(input, self.errors)[0]


class IncrementalDecoder(codecs.BufferedIncrementalDecoder):
    def _buffer_decode(self, input, errors, final):
        return codecs.utf_8_decode(realize(input), errors, final)


class StreamWriter(codecs.StreamWriter):
    def encode(self, input: str, errors: str = "strict") -> Tuple[bytes, int]:
        return _encode(input, errors)


class StreamReader(codecs.StreamReader):
    def decode(self, input: bytes, errors: str = "strict") -> Tuple[str, int]:
        return _decode(input, errors)


def getregentry():
    return codecs.CodecInfo(
        name="utf-8",
        encode=_encode,
        decode=_decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
