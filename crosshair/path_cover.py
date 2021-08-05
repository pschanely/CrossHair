import copy
from crosshair.condition_parser import condition_parser
from dataclasses import dataclass
from inspect import BoundArguments
from inspect import Signature
import time
from typing import Callable
from typing import List
from typing import Optional
from typing import Set
from typing import TextIO
from typing import Type

from crosshair.core import ExceptionFilter, deep_realize, gen_args, Patched
from crosshair.options import AnalysisOptions
from crosshair.fnutil import FunctionInfo
from crosshair.statespace import CallAnalysis
from crosshair.statespace import SinglePathNode
from crosshair.statespace import StateSpace
from crosshair.statespace import StateSpaceContext
from crosshair.statespace import VerificationStatus
from crosshair.tracers import NoTracing, COMPOSITE_TRACER
from crosshair.util import (
    CoverageResult,
    UnexploredPath,
    debug,
    measure_fn_coverage,
    name_of_type,
)


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
        # coverage = lambda _: CoverageResult(set(), set(), 1.0)
        # with ExceptionFilter() as efilter:
        ret = fn(*args.args, **args.kwargs)
    if efilter.user_exc is not None:
        exc = efilter.user_exc[0]
        debug("user-level exception found", repr(exc), *efilter.user_exc[1])
        return PathSummary(pre_args, ret, type(exc), args, coverage(fn))
    elif efilter.ignore:
        return None
    else:
        return PathSummary(
            deep_realize(pre_args),
            deep_realize(ret),
            None,
            deep_realize(args),
            coverage(fn),
        )


def path_cover(ctxfn: FunctionInfo, options: AnalysisOptions) -> List[PathSummary]:
    fn, sig = ctxfn.callable()
    search_root = SinglePathNode(True)
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
        options.incr("num_paths")
        space = StateSpace(
            execution_deadline=itr_start + options.per_path_timeout,
            model_check_timeout=options.per_path_timeout / 2,
            search_root=search_root,
        )
        with condition_parser(
            options.analysis_kind
        ), Patched(), COMPOSITE_TRACER, StateSpaceContext(space):
            summary = None
            try:
                summary = run_iteration(fn, sig, space)
                verification_status = VerificationStatus.CONFIRMED
            except UnexploredPath:
                verification_status = VerificationStatus.UNKNOWN
            debug("Verification status:", verification_status)
            top_analysis, _ = space.bubble_status(CallAnalysis(verification_status))
            if (
                top_analysis
                and top_analysis.verification_status == VerificationStatus.CONFIRMED
            ):
                debug("Stopping due to code path exhaustion. (yay!)")
                options.incr("exhaustion")
                break
            if summary:
                paths.append(summary)
    debug("stats", options.stats)
    opcodes_found: Set[int] = set()
    selected: List[PathSummary] = []
    while paths:
        next_best = max(
            paths, key=lambda p: len(p.coverage.offsets_covered - opcodes_found)
        )
        cur_offsets = next_best.coverage.offsets_covered
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
