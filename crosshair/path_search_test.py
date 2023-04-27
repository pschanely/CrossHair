import ast

from crosshair.fnutil import FunctionInfo
from crosshair.options import DEFAULT_OPTIONS, AnalysisOptionSet
from crosshair.path_search import OptimizationKind, path_search


def ten_over_difference(x: int, y: int) -> int:
    if x != 42:
        return 10 // (x - y)
    return 100


def test_optimize_options() -> None:
    fninfo = FunctionInfo.from_fn(ten_over_difference)
    opts = DEFAULT_OPTIONS
    ret = path_search(fninfo, opts, optimization_kind=OptimizationKind.SIMPLIFY)
    assert ret in ('{"x": 1, "y": 0}', '{"x": 0, "y": 1}')
    ret = path_search(fninfo, opts, optimization_kind=OptimizationKind.MINIMIZE_INT)
    assert ret is not None
    parsed_ret = ast.literal_eval(ret)
    assert parsed_ret["x"] - parsed_ret["y"] > 10
    ret = path_search(fninfo, opts, optimization_kind=OptimizationKind.NONE)
    assert ret == '{"x": 42, "y": 0}'
