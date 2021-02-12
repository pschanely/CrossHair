"""Configure analysis options at different levels."""
import inspect
import re
import sys
import tokenize
from typing import Any, List, Tuple

from crosshair.options import AnalysisOptionSet
from crosshair.util import memo
from crosshair.util import sourcelines

_COMMENT_TOKEN_RE = re.compile(r"^\#\s*crosshair\s*\:\s*(.*?)\s*$")


def get_directives(lines: List[str]) -> List[Tuple[int, str]]:
    """
    Extract directive text from source lines.

    :returns: a list of (line number, directive text) pairs

    >>> get_directives(["pass", "# crosshair: foo=bar"])
    [(2, 'foo=bar')]
    """
    tokens = tokenize.generate_tokens(lines.__iter__().__next__)
    ret = []
    for (toktyp, tokval, begin, _, _) in tokens:
        if toktyp == tokenize.COMMENT:
            tokval = _COMMENT_TOKEN_RE.sub(r"\1", tokval)
            ret.append((begin[0], tokval))
    return ret


class InvalidDirective(Exception):
    pass


def parse_directives(directive_lines: List[Tuple[int, str]]) -> AnalysisOptionSet:
    """
    Parse options from directives in comments.

    >>> parse_directives([(1, "off")]).enabled
    False
    """
    result = AnalysisOptionSet()
    for lineno, directive in directive_lines:
        for part in directive.split():
            if part == "on":
                part = "enabled=1"
            if part == "off":
                part = "enabled="
            pair = part.split("=", 2)
            if len(pair) != 2:
                raise InvalidDirective(f'Malformed option: "{part}"', lineno)
            key, strvalue = pair
            if key not in AnalysisOptionSet.directive_fields:
                raise InvalidDirective(f'Unknown option: "{key}"', lineno)
            value = AnalysisOptionSet.parse_field(key, strvalue)
            if value is None:
                raise InvalidDirective(
                    f'"{strvalue}" is not a valid "{key}" value', lineno
                )
            if getattr(result, key) is not None:
                raise InvalidDirective(
                    f'Option "{key}" is set multiple times at the same scope', lineno
                )
            result = result.overlay(AnalysisOptionSet(**{key: value}))
    return result


@memo
def collect_options(thing: Any) -> AnalysisOptionSet:
    parent_opts = AnalysisOptionSet()
    if hasattr(thing, "__module__"):
        parent_opts = collect_options(sys.modules[thing.__module__])
    elif hasattr(thing, "__package__"):
        if thing.__package__ and thing.__package__ != thing.__name__:
            parent_opts = collect_options(sys.modules[thing.__package__])

    _file, start, lines = sourcelines(thing)
    my_opts = parse_directives(get_directives(lines))
    return parent_opts.overlay(my_opts)
