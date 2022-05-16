import importlib
import re
import sys

from crosshair import NoTracing, ResumedTracing, register_patch
from crosshair.libimpl.builtinslib import CrossHairValue, SymbolicBool, SymbolicInt
from crosshair.tracers import ResumedTracing


def _jsonint(self):
    with NoTracing():
        if isinstance(self, CrossHairValue):
            if isinstance(self, SymbolicBool):
                return "true" if self else "false"
            elif isinstance(self, SymbolicInt):
                with ResumedTracing():
                    return self._symbolic_repr()
            return self.__repr__()
    return int.__repr__(self)


def _jsonfloat(self):
    with NoTracing():
        if isinstance(self, CrossHairValue):
            return self.__repr__()
    return float.__repr__(self)


def make_registrations() -> None:
    # CPython already ships with a pure-Python implementation of this module.
    # We just need to do some hacks to prevent it from using the optimized C versions
    # of some functions:
    import json
    import json.decoder
    import json.encoder
    import json.scanner  # type: ignore

    sys.modules["_json"] = None  # type: ignore

    importlib.reload(json.scanner)

    # The pure python scanner accepts unicode digits, but the C-based scanner does not.
    # Patch json.scanner.NUMBER_RE to include the re.ASCII flag:
    assert json.scanner.NUMBER_RE is not None
    json.scanner.NUMBER_RE = re.compile(
        r"(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?",
        (re.VERBOSE | re.MULTILINE | re.DOTALL | re.ASCII),
    )

    importlib.reload(json.encoder)
    importlib.reload(json.decoder)
    importlib.reload(json)

    # Normally we can rely on symbolic classes just overriding __repr__().
    # However, json.encoder._make_iterencode() explicitly invokes the __repr__ of
    # specific classes, like int and float; so we inject some customized encoders here:
    def _make_iterencode(markers, _default, _encoder, _indent, _floatstr, *a, **kw):
        return json.encoder._make_iterencode(
            markers, _default, _encoder, _indent, _jsonfloat, *a, **kw, _intstr=_jsonint
        )

    register_patch(json.encoder._make_iterencode, _make_iterencode)  # type: ignore
