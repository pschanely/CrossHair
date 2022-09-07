import functools
import re
from io import StringIO

from crosshair.fnutil import FunctionInfo
from crosshair.options import DEFAULT_OPTIONS
from crosshair.path_cover import CoverageType, output_eval_exression_paths, path_cover


def _foo(x: int) -> int:
    if x > 100:
        return 100
    return x


def _regex(x: str) -> bool:
    compiled = re.compile("f(o)+")
    return bool(compiled.fullmatch(x))


def _exceptionex(x: int) -> int:
    if x == 42:
        raise ValueError
    return x


OPTS = DEFAULT_OPTIONS.overlay(max_iterations=10, per_condition_timeout=10.0)
foo = FunctionInfo.from_fn(_foo)
decorated_foo = FunctionInfo.from_fn(functools.lru_cache()(_foo))
regex = FunctionInfo.from_fn(_regex)
exceptionex = FunctionInfo.from_fn(_exceptionex)


def test_path_cover_foo() -> None:
    paths = list(path_cover(foo, OPTS, CoverageType.OPCODE))
    assert len(paths) == 2
    small, large = sorted(paths, key=lambda p: p.result)  # type: ignore
    assert large.result == 100
    assert large.args.arguments["x"] > 100
    assert small.result == small.args.arguments["x"]


def test_path_cover_decorated_foo() -> None:
    paths = list(path_cover(decorated_foo, OPTS, CoverageType.OPCODE))
    assert len(paths) == 2


def test_path_cover_regex() -> None:
    paths = list(path_cover(regex, OPTS, CoverageType.OPCODE))
    assert len(paths) == 1
    paths = list(path_cover(regex, OPTS, CoverageType.PATH))
    input_output = set((p.args.arguments["x"], p.result) for p in paths)
    assert ("foo", True) in input_output


def test_path_cover_exception_example() -> None:
    paths = list(path_cover(exceptionex, OPTS, CoverageType.OPCODE))
    out, err = StringIO(), StringIO()
    output_eval_exression_paths(_exceptionex, paths, out, err)
    assert "_exceptionex(42)" in out.getvalue()
