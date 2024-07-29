import ast
from inspect import BoundArguments
from typing import Callable, Optional

from crosshair.fnutil import FunctionInfo
from crosshair.options import DEFAULT_OPTIONS, AnalysisOptions, AnalysisOptionSet
from crosshair.path_search import OptimizationKind, path_search


def ten_over_difference(x: int, y: int) -> int:
    if x != 42:
        return 10 // (x - y)
    return 100


def do_path_search(
    fn: Callable,
    options: AnalysisOptions,
    argument_formatter: Optional[Callable[[BoundArguments], str]],
    optimization_kind: OptimizationKind,
    optimize_fn: Optional[Callable] = None,
) -> Optional[str]:
    fninfo = FunctionInfo.from_fn(fn)
    final_example: Optional[str] = None

    def on_example(example: str) -> None:
        nonlocal final_example
        final_example = example

    path_search(
        fninfo, options, argument_formatter, optimization_kind, optimize_fn, on_example
    )
    return final_example


def test_optimize_options() -> None:
    opts = DEFAULT_OPTIONS.overlay(AnalysisOptionSet(max_uninteresting_iterations=10))
    ret = do_path_search(
        ten_over_difference, opts, None, optimization_kind=OptimizationKind.SIMPLIFY
    )
    assert ret in ('{"x": 1, "y": 0}', '{"x": 0, "y": 1}')
    ret = do_path_search(
        ten_over_difference, opts, None, optimization_kind=OptimizationKind.MINIMIZE_INT
    )
    assert ret is not None
    parsed_ret = ast.literal_eval(ret)
    assert parsed_ret["x"] - parsed_ret["y"] > 10
    ret = do_path_search(
        ten_over_difference, opts, None, optimization_kind=OptimizationKind.NONE
    )
    assert ret is not None
    ast.literal_eval(ret)  # (just ensure the result is parseable)
