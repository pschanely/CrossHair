import codecs
from collections.abc import ByteString
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

from crosshair.core import realize
from crosshair.libimpl.builtinslib import AnySymbolicStr, SymbolicBytes


class ChunkError:
    def reason(self) -> str:
        raise NotImplementedError


class UnexpectedEndError(ChunkError):
    def reason(self) -> str:
        return "unexpected end of data"


@dataclass
class MidChunkError(ChunkError):
    _reason: str
    # _errlen: int = 1
    def reason(self) -> str:
        return self._reason


class _UnicodeDecodeError(UnicodeDecodeError):
    def __init__(self, enc, byts, start, end, reason):
        UnicodeDecodeError.__init__(self, enc, b"", start, end, reason)
        self.object = byts

    def __ch_deep_realize__(self, memo) -> object:
        enc, obj, reason = self.encoding, self.object, self.reason
        start, end = self.start, self.end
        return UnicodeDecodeError(
            realize(enc), realize(obj), realize(start), realize(end), realize(reason)
        )

    def __repr__(self):
        enc, obj, reason = self.encoding, self.object, self.reason
        start, end = self.start, self.end
        return f"UnicodeDecodeError({enc!r}, {obj!r}, {start!r}, {end!r}, {reason!r})"


class StemEncoder:

    encoding_name: str

    @classmethod
    def _encode_chunk(
        cls, intput: str, start: int
    ) -> Tuple[Union[bytes, SymbolicBytes], int, Optional[ChunkError]]:
        raise NotImplementedError

    @classmethod
    def _decode_chunk(
        cls, intput: bytes, start: int
    ) -> Tuple[Union[str, AnySymbolicStr], int, Optional[ChunkError]]:
        raise NotImplementedError

    @classmethod
    def encode(
        cls, input: str, errors: str = "strict"
    ) -> Tuple[Union[bytes, SymbolicBytes], int]:
        if not (isinstance(input, str) and isinstance(errors, str)):
            raise TypeError
        parts: List[bytes] = []
        idx = 0
        inputlen = len(input)
        while idx < inputlen:
            out, idx, err = cls._encode_chunk(input, idx)
            parts.append(out)  # type: ignore
            if err is not None:
                realized_input = realize(input)  # TODO: avoid realization here.
                # (which possibly requires implementing the error handlers in python)
                exc = UnicodeEncodeError(
                    cls.encoding_name, realized_input, idx, idx + 1, err.reason()
                )
                replacement, idx = codecs.lookup_error(errors)(exc)
                if isinstance(replacement, str):
                    replacement = codecs.encode(replacement, cls.encoding_name)
                parts.append(replacement)
        return b"".join(parts), idx

    @classmethod
    def decode(
        cls, input: bytes, errors: str = "strict"
    ) -> Tuple[Union[str, AnySymbolicStr], int]:
        if not (isinstance(input, ByteString) and isinstance(errors, str)):
            raise TypeError
        parts: List[Union[str, AnySymbolicStr]] = []
        idx = 0
        inputlen = len(input)
        while idx < inputlen:
            out, idx, err = cls._decode_chunk(input, idx)
            parts.append(out)
            if err is not None:
                # 1. Handle some well-known error modes directly:
                if errors == "strict":
                    raise _UnicodeDecodeError(
                        cls.encoding_name, input, idx, idx + 1, err.reason()
                    )
                # TODO: continuation after erros seems poorly tested right now
                if errors == "ignore":
                    idx += 1
                    continue
                if errors == "replace":
                    idx += 1
                    parts.append("\uFFFD")
                    continue

                # 2. Then fall back to native implementations if necessary:
                exc = UnicodeDecodeError(
                    cls.encoding_name, realize(input), idx, idx + 1, err.reason()
                )
                replacement, idx = codecs.lookup_error(errors)(exc)
                if isinstance(replacement, bytes):
                    replacement = codecs.decode(replacement, cls.encoding_name)
                parts.append(replacement)
        return "".join(parts), idx  # type: ignore

    @classmethod
    def getregentry(cls) -> codecs.CodecInfo:
        return _getregentry(cls)


def _getregentry(stem_encoder: Type[StemEncoder]):
    class StemIncrementalEncoder(codecs.BufferedIncrementalEncoder):
        def _buffer_encode(self, input: str, errors: str, final: bool) -> bytes:
            enc_name = stem_encoder.encoding_name
            out, idx, err = stem_encoder._encode_chunk(input, 0)
            assert isinstance(out, bytes)
            if not err:
                return out
            if isinstance(err, UnexpectedEndError) or not final:
                return out
            exc = UnicodeEncodeError(enc_name, input, idx, idx + 1, err.reason())
            replacement, idx = codecs.lookup_error(errors)(exc)
            if isinstance(replacement, str):
                replacement = codecs.encode(replacement, enc_name)
            return out + replacement

    class StemIncrementalDecoder(codecs.BufferedIncrementalDecoder):
        def _buffer_decode(
            self, input: bytes, errors: str, final: bool
        ) -> Tuple[str, int]:
            enc_name = stem_encoder.encoding_name
            out, idx, err = stem_encoder._decode_chunk(input, 0)
            assert isinstance(out, str)
            if not err:
                return out, idx
            if isinstance(err, UnexpectedEndError) or not final:
                return out, idx
            exc = UnicodeDecodeError(enc_name, input, idx, idx + 1, err.reason())
            replacement, idx = codecs.lookup_error(errors)(exc)
            if isinstance(replacement, bytes):
                replacement = codecs.decode(replacement, enc_name)
            return (out + replacement, idx)

    class StemStreamWriter(codecs.StreamWriter):
        def encode(self, input: str, errors: str = "strict") -> Tuple[bytes, int]:
            raise Exception  # TODO implement

    class StemStreamReader(codecs.StreamReader):
        def decode(self, input: bytes, errors: str = "strict") -> Tuple[str, int]:
            raise Exception

    return codecs.CodecInfo(
        name=stem_encoder.encoding_name,
        encode=stem_encoder.encode,  # type: ignore
        decode=stem_encoder.decode,  # type: ignore
        incrementalencoder=StemIncrementalEncoder,
        incrementaldecoder=StemIncrementalDecoder,
        streamreader=StemStreamReader,
        streamwriter=StemStreamWriter,
    )
