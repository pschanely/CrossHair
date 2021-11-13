import importlib
import sys


def make_registrations() -> None:
    # CPython already ships with a pure-Python implementation of this module.
    # We just need to do some hacks to prevent it from using the optimized C versions
    # of some functions:
    import json.encoder
    import json.decoder
    import json.scanner  # type: ignore
    import json

    sys.modules["_json"] = None  # type: ignore
    importlib.reload(json.encoder)
    importlib.reload(json.decoder)
    importlib.reload(json.scanner)
    importlib.reload(json)
