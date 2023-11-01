import functools
import re
import textwrap
from enum import Enum
from io import StringIO
from typing import Callable, Optional

from crosshair.fnutil import FunctionInfo
from crosshair.options import DEFAULT_OPTIONS
from crosshair.path_cover import (
    CoverageType,
    output_eval_exression_paths,
    output_pytest_paths,
    path_cover,
)
from crosshair.statespace import context_statespace
from crosshair.tracers import NoTracing


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


def _symbolic_exception_example(x: str) -> str:
    if x == "foobar":
        raise ValueError(x)
    return x


def _has_no_successful_paths(x: int) -> None:
    with NoTracing():
        context_statespace().defer_assumption("fail", lambda: False)


class Color(Enum):
    RED = 0


def _enum_identity(color: Color):
    return color


OPTS = DEFAULT_OPTIONS.overlay(max_iterations=10, per_condition_timeout=10.0)
foo = FunctionInfo.from_fn(_foo)
decorated_foo = FunctionInfo.from_fn(functools.lru_cache()(_foo))
regex = FunctionInfo.from_fn(_regex)
exceptionex = FunctionInfo.from_fn(_exceptionex)
symbolic_exception_example = FunctionInfo.from_fn(_symbolic_exception_example)
has_no_successful_paths = FunctionInfo.from_fn(_has_no_successful_paths)
enum_identity = FunctionInfo.from_fn(_enum_identity)


def test_path_cover_foo() -> None:
    paths = list(path_cover(foo, OPTS, CoverageType.OPCODE))
    assert len(paths) == 2
    small, large = sorted(paths, key=lambda p: p.result)  # type: ignore
    assert large.result == "100"
    assert large.args.arguments["x"] > 100
    assert small.result == repr(small.args.arguments["x"])


def test_path_cover_decorated_foo() -> None:
    paths = list(path_cover(decorated_foo, OPTS, CoverageType.OPCODE))
    assert len(paths) == 2


def test_path_cover_regex() -> None:
    paths = list(path_cover(regex, OPTS, CoverageType.OPCODE))
    assert len(paths) == 1
    paths = list(path_cover(regex, OPTS, CoverageType.PATH))
    input_output = set((p.args.arguments["x"], p.result) for p in paths)
    assert ("foo", "True") in input_output


def test_path_cover_exception_example() -> None:
    paths = list(path_cover(exceptionex, OPTS, CoverageType.OPCODE))
    out, err = StringIO(), StringIO()
    output_eval_exression_paths(_exceptionex, paths, out, err)
    assert "_exceptionex(42)" in out.getvalue()


def test_path_cover_symbolic_exception_message() -> None:
    paths = list(path_cover(symbolic_exception_example, OPTS, CoverageType.OPCODE))
    _imports, lines = output_pytest_paths(_symbolic_exception_example, paths)
    expected = textwrap.dedent(
        """\
        def test__symbolic_exception_example():
            with pytest.raises(ValueError, match='foobar'):
                _symbolic_exception_example('foobar')"""
    )
    assert expected in "\n".join(lines)


def test_has_no_successful_paths() -> None:
    assert list(path_cover(has_no_successful_paths, OPTS, CoverageType.OPCODE)) == []


def test_path_cover_lambda() -> None:
    def lambdaFn(a: Optional[Callable[[int], int]]):
        if a:
            return a(2) + 4
        else:
            return "hello"

    assert path_cover(FunctionInfo.from_fn(lambdaFn), OPTS, CoverageType.OPCODE)
    # TODO: more detailed assert?


def test_path_cover_pytest_output() -> None:
    paths = list(path_cover(enum_identity, OPTS, CoverageType.OPCODE))
    imports, lines = output_pytest_paths(_enum_identity, paths)
    assert imports == {
        "from crosshair.path_cover_test import _enum_identity",
    }
    assert lines == [
        "def test__enum_identity():",
        "    assert _enum_identity(Color.RED) == Color.RED",
        "",
    ]
