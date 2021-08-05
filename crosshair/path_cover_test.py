import re

from crosshair.fnutil import FunctionInfo
from crosshair.options import DEFAULT_OPTIONS
from crosshair.path_cover import path_cover
from crosshair.core_and_libs import *


def _foo(x: int) -> int:
    if x > 100:
        return 100
    return x


def _regex(x: str) -> bool:
    compiled = re.compile("f(oo)+(bar)?")
    return bool(compiled.fullmatch(x))


foo = FunctionInfo.from_fn(_foo)
regex = FunctionInfo.from_fn(_regex)


def test_path_cover() -> None:
    paths = list(path_cover(foo, DEFAULT_OPTIONS.overlay(max_iterations=10)))
    assert len(paths) == 2
    small, large = sorted(paths, key=lambda p: p.result)  # type: ignore
    assert large.result == 100
    assert large.args.arguments["x"] > 100
    assert small.result == small.args.arguments["x"]


def test_path_cover_regex() -> None:
    paths = list(path_cover(regex, DEFAULT_OPTIONS.overlay(max_iterations=10)))
    # Someday, it'd be nice to support per-path coverage, which could generate a variety
    # of matching and non-matching examples here:
    assert len(paths) == 1
