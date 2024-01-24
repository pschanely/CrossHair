"""Configure analysis options at different levels."""
import importlib.resources
import inspect
import re
import sys
import tokenize
from io import StringIO
from typing import Any, Iterable, Tuple

from crosshair.options import AnalysisOptionSet
from crosshair.util import memo, sourcelines

_COMMENT_TOKEN_RE = re.compile(r"^\#\s*crosshair\s*\:\s*(.*?)\s*$")


def get_directives(source_text: str) -> Iterable[Tuple[int, int, str]]:
    r"""
    Extract directive text from source lines.

    :returns: a list of (line number, column number, directive text) tuples

    >>> get_directives("pass\n# crosshair: foo=bar")
    [(2, 0, 'foo=bar')]
    """
    ret = []
    tokens = tokenize.generate_tokens(StringIO(source_text).readline)
    # TODO catch tokenize.TokenError ... just in case?
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
                part = "enabled=yes"
            if part == "off":
                part = "enabled=no"
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

    source_text: str
    if is_package:
        try:
            if sys.version_info >= (3, 10):
                source_text = (
                    importlib.resources.files(thing).joinpath("__init__.py").read_text()
                )
            else:
                source_text = importlib.resources.read_text(thing, "__init__.py")
        except FileNotFoundError:
            source_text = ""
    else:
        _file, _start, lines = sourcelines(thing)
        source_text = "".join(lines)
    directives = get_directives(source_text)
    if inspect.ismodule(thing):
        # Only look at toplevel comments in modules
        # (we don't want to catch directives for functions inside it)
        # TODO: detect directives at other levels like classes etc and warn that they
        # will be ignored.
        directives = [(l, c, t) for (l, c, t) in directives if c == 0]
    my_opts = parse_directives(directives)
    return parent_opts.overlay(my_opts)
