import re

from crosshair.fnutil import FunctionInfo
from crosshair.options import DEFAULT_OPTIONS
from crosshair.path_cover import path_cover
from crosshair.path_cover import CoverageType


def _foo(x: int) -> int:
    if x > 100:
        return 100
    return x


def _regex(x: str) -> bool:
    compiled = re.compile("f(o)+")
    return bool(compiled.fullmatch(x))


OPTS = DEFAULT_OPTIONS.overlay(max_iterations=10, per_condition_timeout=10.0)
foo = FunctionInfo.from_fn(_foo)
regex = FunctionInfo.from_fn(_regex)


def test_path_cover() -> None:
    paths = list(path_cover(foo, OPTS, CoverageType.OPCODE))
    assert len(paths) == 2
    small, large = sorted(paths, key=lambda p: p.result)  # type: ignore
    assert large.result == 100
    assert large.args.arguments["x"] > 100
    assert small.result == small.args.arguments["x"]


def test_path_cover_regex() -> None:
    paths = list(path_cover(regex, OPTS, CoverageType.OPCODE))
    assert len(paths) == 1
    paths = list(path_cover(regex, OPTS, CoverageType.PATH))
    input_output = set((p.args.arguments["x"], p.result) for p in paths)
    assert ("fo", True) in input_output
    assert ("f", False) in input_output
