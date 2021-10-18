import copy
import dataclasses
import dis
import inspect
import time
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union
from crosshair.fnutil import FunctionInfo
from crosshair.statespace import CallAnalysis
from crosshair.statespace import SinglePathNode
from crosshair.statespace import StateSpace
from crosshair.statespace import StateSpaceContext
from crosshair.statespace import VerificationStatus
from crosshair.condition_parser import condition_parser
from crosshair.core import gen_args
from crosshair.core import Patched
from crosshair.core import ExceptionFilter
from crosshair.options import AnalysisOptions
from crosshair.tracers import NoTracing
from crosshair.util import debug
from crosshair.util import UnexploredPath
from crosshair.util import measure_fn_coverage
from crosshair.util import CoverageResult


@dataclasses.dataclass
class Result:
    return_repr: str
    error: Optional[str]
    post_execution_args: Dict[str, str]

    def get_differing_arg_mutations(self, other: "Result") -> Set[str]:
        args1 = self.post_execution_args
        args2 = other.post_execution_args
        differing_args: Set[str] = set()
        for arg in set(args1.keys()) | args2.keys():
            if arg in args1 and arg in args2 and args1[arg] != args2[arg]:
                differing_args.add(arg)
        return differing_args

    def describe(self, args_to_show: Set[str]) -> str:
        ret = []
        if self.error is not None:
            ret.append(f"raises {self.error}")
        if self.return_repr != "None":
            ret.append(f"returns {self.return_repr}")
        if args_to_show:
            if ret:
                ret.append(", ")
            ret.append("after execution ")
            ret.append(
                ", ".join(
                    f"{arg}={self.post_execution_args[arg]}" for arg in args_to_show
                )
            )
        # last resort, be explicit about returning none:
        if not ret:
            ret.append("returns None")
        return "".join(ret)


def describe_behavior(
    fn: Callable, args: inspect.BoundArguments
) -> Tuple[object, Optional[str]]:
    with ExceptionFilter() as efilter:
        ret = fn(*args.args, **args.kwargs)
        return (ret, None)
    if efilter.user_exc is not None:
        exc = efilter.user_exc[0]
        debug("user-level exception found", repr(exc), *efilter.user_exc[1])
        return (None, repr(exc))
    if efilter.ignore:
        return (None, "IgnoreAttempt")
    assert False


@dataclasses.dataclass
class BehaviorDiff:
    args: Dict[str, str]
    result1: Result
    result2: Result
    coverage1: CoverageResult
    coverage2: CoverageResult

    def reverse(self) -> "BehaviorDiff":
        return BehaviorDiff(
            self.args, self.result2, self.result1, self.coverage2, self.coverage1
        )


def diff_scorer(
    check_opcodes1: Set[int], check_opcodes2: Set[int]
) -> Callable[[BehaviorDiff], Tuple[float, float]]:
    """
    Create a function to score the usefulness of behavior diffs.

    We aim for a minimal number of examples that gives as much coverage of the
    differing opcodes as possible.
    We break ties on smaller examples. (repr-string-length-wise)
    """
    pass  # for pydocstyle

    def scorer(diff: BehaviorDiff) -> Tuple[float, float]:
        coverage1 = diff.coverage1
        coverage2 = diff.coverage2
        total_opcodes = len(coverage1.all_offsets) + len(coverage2.all_offsets)
        cover1 = len(coverage1.offsets_covered & check_opcodes1)
        cover2 = len(coverage2.offsets_covered & check_opcodes2)
        cover_score = (cover1 + cover2) / total_opcodes
        strlen_score = len(str(diff))
        return (cover_score, strlen_score)

    return scorer


def diff_behavior(
    ctxfn1: FunctionInfo, ctxfn2: FunctionInfo, options: AnalysisOptions
) -> Union[str, List[BehaviorDiff]]:
    fn1, sig1 = ctxfn1.callable()
    fn2, sig2 = ctxfn2.callable()
    debug("Resolved signature:", sig1)
    all_diffs: List[BehaviorDiff] = []
    half1, half2 = options.split_limits(0.5)
    with condition_parser(options.analysis_kind), Patched():
        # We attempt both orderings of functions. This helps by:
        # (1) avoiding code path explosions in one of the functions
        # (2) using both signatures (in case they differ)
        all_diffs.extend(diff_behavior_with_signature(fn1, fn2, sig1, half1))
        all_diffs.extend(
            diff.reverse()
            for diff in diff_behavior_with_signature(fn2, fn1, sig2, half2)
        )
    debug("diff candidates:", all_diffs)
    # greedily pick results:
    result_diffs = []
    opcodeset1 = set(i.offset for i in dis.get_instructions(fn1.__code__))
    opcodeset2 = set(i.offset for i in dis.get_instructions(fn2.__code__))
    while all_diffs:
        scorer = diff_scorer(opcodeset1, opcodeset2)
        selection = max(all_diffs, key=scorer)
        (num_opcodes, _) = scorer(selection)
        debug("Considering input", selection.args, " num opcodes=", num_opcodes)
        if num_opcodes == 0:
            break
        all_diffs.remove(selection)
        result_diffs.append(selection)
        coverage1, coverage2 = selection.coverage1, selection.coverage2
        if coverage1 is not None and coverage2 is not None:
            opcodeset1 -= coverage1.offsets_covered
            opcodeset2 -= coverage2.offsets_covered
    return result_diffs


def diff_behavior_with_signature(
    fn1: Callable, fn2: Callable, sig: inspect.Signature, options: AnalysisOptions
) -> Iterable[BehaviorDiff]:
    search_root = SinglePathNode(True)
    condition_start = time.monotonic()
    for i in range(1, options.max_iterations):
        debug("Iteration ", i)
        itr_start = time.monotonic()
        if itr_start > condition_start + options.per_condition_timeout:
            debug(
                "Stopping due to --per_condition_timeout=",
                options.per_condition_timeout,
            )
            return
        options.incr("num_paths")
        space = StateSpace(
            execution_deadline=itr_start + options.per_path_timeout,
            model_check_timeout=options.per_path_timeout / 2,
            search_root=search_root,
        )
        with StateSpaceContext(space):
            output = None
            try:
                (verification_status, output) = run_iteration(fn1, fn2, sig, space)
            except UnexploredPath:
                verification_status = VerificationStatus.UNKNOWN
            debug("Verification status:", verification_status)
            top_analysis, space_exhausted = space.bubble_status(
                CallAnalysis(verification_status)
            )
            if (
                top_analysis
                and top_analysis.verification_status == VerificationStatus.CONFIRMED
            ):
                debug("Stopping due to code path exhaustion. (yay!)")
                options.incr("exhaustion")
                break
            if output:
                yield output


def run_iteration(
    fn1: Callable, fn2: Callable, sig: inspect.Signature, space: StateSpace
) -> Tuple[Optional[VerificationStatus], Optional[BehaviorDiff]]:
    with NoTracing():
        original_args = gen_args(sig)
    args1 = copy.deepcopy(original_args)
    args2 = copy.deepcopy(original_args)

    coverage_manager = measure_fn_coverage(fn1, fn2)
    with ExceptionFilter() as efilter, coverage_manager as coverage:
        result1 = describe_behavior(fn1, args1)
        result2 = describe_behavior(fn2, args2)
        space.detach_path()
        if result1 == result2 and args1 == args2:
            debug("Functions equivalent")
            return (VerificationStatus.CONFIRMED, None)
        debug("Functions differ")
        realized_args = {k: repr(v) for (k, v) in original_args.arguments.items()}
        post_execution_args1 = {k: repr(v) for k, v in args1.arguments.items()}
        post_execution_args2 = {k: repr(v) for k, v in args2.arguments.items()}
        diff = BehaviorDiff(
            realized_args,
            Result(repr(result1[0]), result1[1], post_execution_args1),
            Result(repr(result2[0]), result2[1], post_execution_args2),
            coverage(fn1),
            coverage(fn2),
        )
        return (VerificationStatus.REFUTED, diff)
    if efilter.user_exc:
        debug(
            "User-level exception found", repr(efilter.user_exc[0]), efilter.user_exc[1]
        )
    return (None, None)
