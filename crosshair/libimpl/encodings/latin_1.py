import codecs
from typing import List, Optional, Tuple, Union

from crosshair.libimpl.builtinslib import SymbolicBytes
from crosshair.libimpl.encodings._encutil import ChunkError, MidChunkError, StemEncoder


class Latin1StemEncoder(StemEncoder):
    encoding_name = "iso8859-1"

    @classmethod
    def _encode_chunk(
        cls, string: str, start: int
    ) -> Tuple[Union[bytes, SymbolicBytes], int, Optional[ChunkError]]:
        byte_ints: List[int] = []
        for idx, ch in enumerate(string):
            cp = ord(ch)
            if cp < 256:
                byte_ints.append(cp)
            else:
                return (
                    SymbolicBytes(byte_ints),
                    idx,
                    MidChunkError("bytes must be in range(0, 256)"),
                )
        return (SymbolicBytes(byte_ints), len(string), None)

    @classmethod
    def _decode_chunk(
        cls, byts: bytes, start: int
    ) -> Tuple[str, int, Optional[ChunkError]]:
        chars: List[str] = []
        for cp in byts:
            chars.append(chr(cp))
        return ("".join(chars), len(byts), None)


def getregentry() -> codecs.CodecInfo:
    return Latin1StemEncoder.getregentry()
