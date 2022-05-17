import codecs
from typing import List, Optional, Tuple, Union

from crosshair.libimpl.builtinslib import SymbolicBytes
from crosshair.libimpl.encodings._encutil import ChunkError, MidChunkError, StemEncoder


class AsciiStemEncoder(StemEncoder):
    encoding_name = "ascii"

    @classmethod
    def _encode_chunk(
        cls, string: str, start: int
    ) -> Tuple[Union[bytes, SymbolicBytes], int, Optional[ChunkError]]:
        byte_ints: List[int] = []
        for idx, ch in enumerate(string):
            cp = ord(ch)
            if cp >= 0x80:
                return (
                    SymbolicBytes(byte_ints),
                    idx,
                    MidChunkError("ordinal not in range"),
                )
            else:
                byte_ints.append(cp)
        return (SymbolicBytes(byte_ints), len(string), None)

    @classmethod
    def _decode_chunk(
        cls, byts: bytes, start: int
    ) -> Tuple[str, int, Optional[ChunkError]]:
        chars: List[str] = []
        for (idx, cp) in enumerate(byts):
            if cp >= 0x80:
                return ("".join(chars), idx, MidChunkError("ordinal not in range"))
            else:
                chars.append(chr(cp))
        return ("".join(chars), len(byts), None)  # type: ignore


def getregentry() -> codecs.CodecInfo:
    return AsciiStemEncoder.getregentry()
