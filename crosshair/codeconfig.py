"""Configure analysis options at different levels."""
import inspect
import re
import sys
import tokenize
from typing import Any, List, Tuple

from crosshair.util import sourcelines

_COMMENT_TOKEN_RE = re.compile(r"^\#\s*crosshair\s*\:\s*(.*?)\s*$")


def get_directives(lines: List[str]) -> List[Tuple[int, str]]:
    tokens = tokenize.generate_tokens(lines.__iter__().__next__)
    ret = []
    for (toktyp, tokval, begin, _, _) in tokens:
        print(toktyp, tokenize.COMMENT, tokval)
        if toktyp == tokenize.COMMENT:
            tokval = _COMMENT_TOKEN_RE.sub(r"\1", tokval)
            ret.append((begin[0], tokval))
    return ret


def collect_options(thing: Any):
    _file, start, lines = sourcelines(thing)
    if hasattr(thing, "__module__"):
        collect_options(sys.modules[thing.__module__])
    elif hasattr(thing, "__package__"):
        if thing.__package__ and thing.__package__ != thing.__name__:
            collect_options(sys.modules[thing.__package__])
