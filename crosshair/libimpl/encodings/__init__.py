import codecs
import importlib
from encodings import normalize_encoding
from encodings.aliases import aliases
from typing import Optional


def codec_search(encoding: str) -> Optional[codecs.CodecInfo]:
    enc_prefix = "crosshair_"
    if not encoding.startswith(enc_prefix):
        return None

    encoding = normalize_encoding(encoding[len(enc_prefix) :])
    encoding = aliases.get(encoding, encoding)

    try:
        module = importlib.import_module(f"crosshair.libimpl.encodings.{encoding}")
    except ImportError:
        pass
    else:
        if callable(getattr(module, "getregentry", None)):
            return module.getregentry()  # type: ignore
    return None
