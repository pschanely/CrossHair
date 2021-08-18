"""Configure analysis options at different levels."""
import importlib
import inspect
import re
import sys
import tokenize
from typing import *

from crosshair.options import AnalysisOptionSet
from crosshair.util import debug
from crosshair.util import memo
from crosshair.util import sourcelines

_COMMENT_TOKEN_RE = re.compile(r"^\#\s*crosshair\s*\:\s*(.*?)\s*$")


def get_directives(lines: Iterable[str]) -> Iterable[Tuple[int, int, str]]:
    """
    Extract directive text from source lines.

    :returns: a list of (line number, column number, directive text) tuples

    >>> get_directives(["pass", "# crosshair: foo=bar"])
    [(2, 0, 'foo=bar')]
    """
    tokens = tokenize.generate_tokens(lines.__iter__().__next__)
    ret = []
    for (toktyp, tokval, begin, _, _) in tokens:
        linenum, colnum = begin
        if toktyp == tokenize.COMMENT:
            directive = _COMMENT_TOKEN_RE.sub(r"\1", tokval)
            if tokval != directive:
                ret.append((linenum, colnum, directive))
    return ret


class InvalidDirective(Exception):
    pass


def parse_directives(
    directive_lines: Iterable[Tuple[int, int, str]]
) -> AnalysisOptionSet:
    """
    Parse options from directives in comments.

    >>> parse_directives([(1, 0, "off")]).enabled
    False
    """
    result = AnalysisOptionSet()
    for lineno, _colno, directive in directive_lines:
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
    is_package = thing.__name__ == getattr(thing, "__package__", None)
    if getattr(thing, "__module__", None):
        parent_opts = collect_options(sys.modules[thing.__module__])
    elif getattr(thing, "__package__", None):
        if is_package:
            parent_pkg, _, _ = thing.__package__.rpartition(".")
        else:
            parent_pkg = thing.__package__
        if parent_pkg:
            parent_opts = collect_options(sys.modules[parent_pkg])

    lines: Iterable[str]
    if is_package:
        try:
            lines = importlib.resources.read_text(thing, "__init__.py").splitlines()
        except FileNotFoundError:
            lines = []
    else:
        _file, _start, lines = sourcelines(thing)
    directives = get_directives(lines)
    if inspect.ismodule(thing):
        # Only look at toplevel comments in modules
        # (we don't want to catch directives for functions inside it)
        # TODO: detect directives at other levels like classes etc and warn that they
        # will be ignored.
        directives = [(l, c, t) for (l, c, t) in directives if c == 0]
    my_opts = parse_directives(directives)
    return parent_opts.overlay(my_opts)
