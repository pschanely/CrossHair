import codecs

from crosshair import NoTracing, register_patch
from crosshair.core import realize
from crosshair.libimpl.encodings import codec_search


def _decode(obj, encoding="utf-8", errors="strict"):
    errors = realize(errors)
    (out, _len_consumed) = _getdecoder(encoding)(obj, errors)
    return out


def _encode(obj, encoding="utf-8", errors="strict"):
    with NoTracing():
        errors = realize(errors)
        if "\x00" in errors:
            raise ValueError
    (out, _len_consumed) = _getencoder(encoding)(obj, errors)
    return out


def _getencoder(encoding):
    return _lookup(encoding).encode


def _getdecoder(encoding):
    return _lookup(encoding).decode


def _getincrementaldecoder(encoding):
    return _lookup(encoding).incrementaldecoder


def _getincrementalencoder(encoding):
    return _lookup(encoding).incrementalencoder


def _getreader(encoding):
    return _lookup(encoding).streamreader


def _getwriter(encoding):
    return _lookup(encoding).streamwriter


def _lookup(encoding: str) -> codecs.CodecInfo:
    with NoTracing():
        encoding = realize(encoding)
        try:
            return codecs.lookup("crosshair_" + encoding)
        except LookupError:
            return codecs.lookup(encoding)


def make_registrations() -> None:
    codecs.register(codec_search)
    register_patch(codecs.decode, _decode)
    register_patch(codecs.getencoder, _getencoder)
    register_patch(codecs.getdecoder, _getdecoder)
    register_patch(codecs.getincrementalencoder, _getincrementalencoder)
    register_patch(codecs.getincrementaldecoder, _getincrementaldecoder)
    register_patch(codecs.getreader, _getreader)
    register_patch(codecs.getwriter, _getwriter)
    register_patch(codecs.encode, _encode)
    register_patch(codecs.lookup, _lookup)
