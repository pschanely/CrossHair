import enum
import traceback
from dataclasses import dataclass
from inspect import BoundArguments
from typing import Callable, List, Optional, Set, TextIO, Type

from crosshair.condition_parser import condition_parser
from crosshair.core import ExceptionFilter, deep_realize, explore_paths
from crosshair.fnutil import FunctionInfo
from crosshair.options import AnalysisOptions
from crosshair.statespace import RootNode, StateSpace
from crosshair.tracers import (
    COMPOSITE_TRACER,
    CoverageResult,
    CoverageTracingModule,
    NoTracing,
    PushedModule,
)
from crosshair.util import (
    debug,
    format_boundargs,
    format_boundargs_as_dictionary,
    name_of_type,
    test_stack,
)


class CoverageType(enum.Enum):
    OPCODE = "OPCODE"
    PATH = "PATH"


@dataclass
class PathSummary:
    args: BoundArguments
    result: object
    exc: Optional[Type[BaseException]]
    post_args: BoundArguments
    coverage: CoverageResult


def path_cover(
    ctxfn: FunctionInfo, options: AnalysisOptions, coverage_type: CoverageType
) -> List[PathSummary]:
    fn, sig = ctxfn.callable()
    while getattr(fn, "__wrapped__", None):
        # Usually we don't want to run decorator code. (and we certainly don't want
        # to measure coverage on the decorator rather than the real body) Unwrap:
        fn = fn.__wrapped__  # type: ignore
    search_root = RootNode()

    paths: List[PathSummary] = []
    coverage: CoverageTracingModule = CoverageTracingModule(fn)

    def run_path(args: BoundArguments):
        nonlocal coverage
        with NoTracing():
            coverage = CoverageTracingModule(fn)
        with PushedModule(coverage):
            return fn(*args.args, **args.kwargs)

    def on_path_complete(
        space: StateSpace,
        pre_args: BoundArguments,
        post_args: BoundArguments,
        ret,
        exc: Optional[Exception],
        exc_stack: Optional[traceback.StackSummary],
    ) -> bool:
        with ExceptionFilter() as efilter:
            space.detach_path()
            pre_args = deep_realize(pre_args)  # TODO: repr-aware realization?
            post_args = deep_realize(post_args)
            ret = deep_realize(ret)
            cov = coverage.get_results(fn)
            if exc is not None:
                debug(
                    "user-level exception found", type(exc), exc, test_stack(exc_stack)
                )
                paths.append(PathSummary(pre_args, ret, type(exc), post_args, cov))
            else:
                paths.append(PathSummary(pre_args, ret, None, post_args, cov))
            return False
        debug("Skipping path (failed to realize values)", efilter.user_exc)
        return False

    explore_paths(run_path, sig, options, search_root, on_path_complete)

    opcodes_found: Set[int] = set()
    selected: List[PathSummary] = []
    while paths:
        next_best = max(
            paths, key=lambda p: len(p.coverage.offsets_covered - opcodes_found)
        )
        cur_offsets = next_best.coverage.offsets_covered
        if coverage_type == CoverageType.OPCODE:
            debug("Next best path covers these opcode offsets:", cur_offsets)
            if len(cur_offsets - opcodes_found) == 0:
                break
        selected.append(next_best)
        opcodes_found |= cur_offsets
        paths = [p for p in paths if p is not next_best]
    return selected


def output_argument_dictionary_paths(
    fn: Callable, paths: List[PathSummary], stdout: TextIO, stderr: TextIO
) -> int:
    for path in paths:
        stdout.write(format_boundargs_as_dictionary(path.args) + "\n")
    stdout.flush()
    return 0


def output_eval_exression_paths(
    fn: Callable, paths: List[PathSummary], stdout: TextIO, stderr: TextIO
) -> int:
    for path in paths:
        stdout.write(fn.__name__ + "(" + format_boundargs(path.args) + ")\n")
    stdout.flush()
    return 0


def output_pytest_paths(
    fn: Callable, paths: List[PathSummary], stdout: TextIO, stderr: TextIO
) -> int:
    fn_name = fn.__name__
    lines: List[str] = []
    lines.append(f"from {fn.__module__} import {fn_name}")
    lines.append("")
    import_pytest = False
    for idx, path in enumerate(paths):
        test_name_suffix = "" if idx == 0 else "_" + str(idx + 1)
        exec_fn = f"{fn_name}({format_boundargs(path.args)})"
        lines.append(f"def test_{fn_name}{test_name_suffix}():")
        if path.exc is None:
            lines.append(f"    assert {exec_fn} == {repr(path.result)}")
        else:
            import_pytest = True
            lines.append(f"    with pytest.raises({name_of_type(path.exc)}):")
            lines.append(f"        {exec_fn}")
        lines.append("")
    if import_pytest:
        lines.insert(0, "import pytest")
    stdout.write("\n".join(lines) + "\n")
    stdout.flush()
    return 0
