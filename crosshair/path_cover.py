import copy
import enum
import time
from dataclasses import dataclass
from inspect import BoundArguments, Signature
from typing import Callable, List, Optional, Set, TextIO, Type

from crosshair.condition_parser import condition_parser
from crosshair.core import ExceptionFilter, Patched, deep_realize, gen_args
from crosshair.fnutil import FunctionInfo
from crosshair.options import AnalysisOptions
from crosshair.statespace import (
    CallAnalysis,
    NotDeterministic,
    RootNode,
    StateSpace,
    StateSpaceContext,
    VerificationStatus,
)
from crosshair.tracers import COMPOSITE_TRACER, NoTracing
from crosshair.util import (
    CoverageResult,
    IgnoreAttempt,
    UnexploredPath,
    debug,
    measure_fn_coverage,
    name_of_type,
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


def run_iteration(
    fn: Callable, sig: Signature, space: StateSpace
) -> Optional[PathSummary]:
    with NoTracing():
        args = gen_args(sig)
    pre_args = copy.deepcopy(args)
    ret = None
    with measure_fn_coverage(fn) as coverage, ExceptionFilter() as efilter:
        ret = fn(*args.args, **args.kwargs)
    if efilter.user_exc and isinstance(efilter.user_exc[0], NotDeterministic):
        raise NotDeterministic
    if efilter.ignore:
        return None
    with ExceptionFilter():
        space.detach_path()
        pre_args = deep_realize(pre_args)
        ret = deep_realize(ret)
        args = deep_realize(args)
        if efilter.user_exc is not None:
            exc = efilter.user_exc[0]
            debug("user-level exception found", type(exc), *efilter.user_exc[1])
            return PathSummary(pre_args, ret, type(exc), args, coverage(fn))
        else:
            return PathSummary(pre_args, ret, None, args, coverage(fn))
    debug("Skipping path (failed to realize values)")
    return None


def path_cover(
    ctxfn: FunctionInfo, options: AnalysisOptions, coverage_type: CoverageType
) -> List[PathSummary]:
    fn, sig = ctxfn.callable()
    while getattr(fn, "__wrapped__", None):
        # Usually we don't want to run decorator code. (and we certainly don't want
        # to measure coverage on the decorator rather than the real body) Unwrap:
        fn = fn.__wrapped__  # type: ignore
    search_root = RootNode()
    condition_start = time.monotonic()
    paths: List[PathSummary] = []
    for i in range(1, options.max_iterations):
        debug("Iteration ", i)
        itr_start = time.monotonic()
        if itr_start > condition_start + options.per_condition_timeout:
            debug(
                "Stopping due to --per_condition_timeout=",
                options.per_condition_timeout,
            )
            break
        per_path_timeout = options.get_per_path_timeout()
        space = StateSpace(
            execution_deadline=itr_start + per_path_timeout,
            model_check_timeout=per_path_timeout / 2,
            search_root=search_root,
        )
        with condition_parser(
            options.analysis_kind
        ), Patched(), COMPOSITE_TRACER, StateSpaceContext(space):
            summary = None
            try:
                summary = run_iteration(fn, sig, space)
                verification_status = VerificationStatus.CONFIRMED
            except IgnoreAttempt:
                verification_status = None
            except UnexploredPath:
                verification_status = VerificationStatus.UNKNOWN
            debug("Verification status:", verification_status)
            top_analysis, exhausted = space.bubble_status(
                CallAnalysis(verification_status)
            )
            debug("Path tree stats", search_root.stats())
            if summary:
                paths.append(summary)
            if exhausted:
                debug("Stopping due to code path exhaustion. (yay!)")
                break
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


def repr_boundargs(boundargs: BoundArguments) -> str:
    pieces = list(map(repr, boundargs.args))
    pieces.extend(f"{k}={repr(v)}" for k, v in boundargs.kwargs.items())
    return ", ".join(pieces)


def output_argument_dictionary_paths(
    fn: Callable, paths: List[PathSummary], stdout: TextIO, stderr: TextIO
) -> int:
    for path in paths:
        stdout.write("(" + repr_boundargs(path.args) + ")\n")
    stdout.flush()
    return 0


def output_eval_exression_paths(
    fn: Callable, paths: List[PathSummary], stdout: TextIO, stderr: TextIO
) -> int:
    for path in paths:
        stdout.write(fn.__name__ + "(" + repr_boundargs(path.args) + ")\n")
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
        exec_fn = f"{fn_name}({repr_boundargs(path.args)})"
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
