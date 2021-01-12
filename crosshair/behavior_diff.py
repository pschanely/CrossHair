import contextlib
import copy
import dataclasses
import difflib
import dis
import inspect
import itertools
import sys
import time
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

from crosshair import core_and_libs

from crosshair.condition_parser import resolve_signature
from crosshair.statespace import CallAnalysis
from crosshair.statespace import SinglePathNode
from crosshair.statespace import TrackingStateSpace
from crosshair.statespace import StateSpaceContext
from crosshair.statespace import VerificationStatus
from crosshair.core import gen_args
from crosshair.core import realize
from crosshair.core import Patched
from crosshair.core import AnalysisOptions
from crosshair.core import ExceptionFilter
from crosshair.util import debug
from crosshair.util import IgnoreAttempt
from crosshair.util import UnexploredPath
from crosshair.util import measure_fn_coverage
from crosshair.util import CoverageResult

@dataclasses.dataclass
class Result:
    return_repr: str
    error: Optional[str]
    def __str__(self):
        if self.error is not None:
            return f'raises {self.error}'
        return f'returns {self.return_repr}'

def describe_behavior(
        fn: Callable, args: inspect.BoundArguments) -> Tuple[object, Optional[str]]:
    with ExceptionFilter() as efilter:
        ret = fn(*args.args, **args.kwargs)
        return (ret, None)
    if efilter.user_exc is not None:
        debug('user-level exception found', *efilter.user_exc)
        exc = efilter.user_exc[0]
        return (None, repr(exc))
    if efilter.ignore:
        return (None, 'IgnoreAttempt')
    assert False


@dataclasses.dataclass
class BehaviorDiff:
    args: Dict[str, str]
    result1: Result
    result2: Result
    coverage1: CoverageResult
    coverage2: CoverageResult
    def reverse(self) -> 'BehaviorDiff':
        return BehaviorDiff(self.args,
                            self.result2, self.result1,
                            self.coverage2, self.coverage1)

def diff_opcodes(fn1: Callable, fn2: Callable) -> Tuple[Set[int], Set[int]]:
    ''' Returns the opcode offsets of opcodes that differ between two functions. '''
    instrs1 = list(dis.get_instructions(fn1.__code__))
    instrs2 = list(dis.get_instructions(fn2.__code__))
    offsets1 = [i.offset for i in instrs1]
    offsets2 = [i.offset for i in instrs2]
    opcodes1 = [(i.opcode, i.argval) for i in instrs1]
    opcodes2 = [(i.opcode, i.argval) for i in instrs2]
    ret_offsets1 = set(offsets1)
    ret_offsets2 = set(offsets2)
    matcher = difflib.SequenceMatcher(None, opcodes1, opcodes2)
    for (index1, index2, span_len) in matcher.get_matching_blocks():
        ret_offsets1 -= set(offsets1[index1: index1 + span_len])
        ret_offsets2 -= set(offsets2[index2: index2 + span_len])
    return (ret_offsets1, ret_offsets2)

def diff_scorer(ignore_opcodes1: Set[int], ignore_opcodes2: Set[int]) -> Callable[[BehaviorDiff], Tuple[float, float]]:
    '''
    We aim for a minimal number of examples that gives as much coverage of the differing opcodes
    as possible. We break ties on smaller examples. (repr-string-length-wise)
    '''
    def scorer(diff: BehaviorDiff) -> Tuple[float, float]:
        coverage1 = diff.coverage1
        coverage2 = diff.coverage2
        total_opcodes = len(coverage1.all_offsets) + len(coverage2.all_offsets)
        cover1 = len(coverage1.offsets_covered - ignore_opcodes1)
        cover2 = len(coverage2.offsets_covered - ignore_opcodes2)
        cover_score = (cover1 + cover2) / total_opcodes
        strlen_score = len(str(diff))
        return (cover_score, strlen_score)
    return scorer

def diff_behavior(fn1: Callable, fn2: Callable, options: AnalysisOptions) -> Union[str, List[BehaviorDiff]]:
    sig1, resolution_err = resolve_signature(fn1)
    sig2, resolution_err = resolve_signature(fn2)
    debug('Resolved signature:', sig1)
    if sig1 is None:
        return f'Unable to get signature of {fn1}'
    if sig2 is None:
        return f'Unable to get signature of {fn2}'
    all_diffs: List[BehaviorDiff] = []
    half1, half2 = options.split_limits(0.5)
    with Patched(enabled=lambda: True):
        # We attempt both orderings of functions. This helps by:
        # (1) avoiding code path explosions in one of the functions
        # (2) using both signatures (in case they differ)
        all_diffs.extend(diff_behavior_with_signature(fn1, fn2, sig1, half1))
        all_diffs.extend(diff.reverse() for diff in
                         diff_behavior_with_signature(fn2, fn1, sig2, half2))
    debug('diff candidates:', all_diffs)
    # greedily pick results:
    result_diffs = []
    (ignoreset1, ignoreset2) = diff_opcodes(fn1, fn2)
    while all_diffs:
        scorer = diff_scorer(ignoreset1, ignoreset2)
        selection = max(all_diffs, key=scorer)
        (num_opcodes, _) = scorer(selection)
        if num_opcodes == 0:
            break
        all_diffs.remove(selection)
        result_diffs.append(selection)
        coverage1, coverage2 = selection.coverage1, selection.coverage2
        if coverage1 is not None and coverage2 is not None:
            ignoreset1 |= coverage1.offsets_covered
            ignoreset2 |= coverage2.offsets_covered
    return result_diffs

def diff_behavior_with_signature(
        fn1: Callable, fn2: Callable, sig: inspect.Signature, options: AnalysisOptions) -> Iterable[BehaviorDiff]:
    search_root = SinglePathNode(True)
    condition_start = time.monotonic()
    for i in range(1, options.max_iterations):
        debug('Iteration ', i)
        itr_start = time.monotonic()
        if itr_start > condition_start + options.per_condition_timeout:
            debug('Stopping due to --per_condition_timeout=', options.per_condition_timeout)
            return
        space = TrackingStateSpace(
            execution_deadline=itr_start + options.per_path_timeout,
            model_check_timeout=options.per_path_timeout / 2,
            search_root=search_root)
        with StateSpaceContext(space):
            try:
                (verification_status, output) = run_iteration(fn1, fn2, sig, space)
            except UnexploredPath:
                verification_status = VerificationStatus.UNKNOWN
            debug('Verification status:', verification_status)
            top_analysis, space_exhausted = space.bubble_status(CallAnalysis(verification_status))
            if top_analysis and top_analysis.verification_status == VerificationStatus.CONFIRMED:
                debug('Stopping due to code path exhaustion. (yay!)')
                break
            if output:
                yield output

def run_iteration(
        fn1: Callable,
        fn2: Callable,
        sig: inspect.Signature,
        space: TrackingStateSpace) -> Tuple[
            Optional[VerificationStatus],
            Optional[BehaviorDiff]]:
    original_args = gen_args(sig)
    args1 = copy.deepcopy(original_args)
    args2 = copy.deepcopy(original_args)

    coverage_manager = measure_fn_coverage(fn1, fn2)
    with ExceptionFilter() as efilter, coverage_manager as coverage:
        result1 = describe_behavior(fn1, args1)
        result2 = describe_behavior(fn2, args2)
        space.check_deferred_assumptions()
        if result1 == result2:
            debug('Functions equivalent')
            return (VerificationStatus.CONFIRMED, None)
        debug('Functions differ')
        realized_args = {k: repr(v) for (k, v) in original_args.arguments.items()}
        diff = BehaviorDiff(
            realized_args,
            Result(repr(result1[0]), result1[1]),
            Result(repr(result2[0]), result2[1]),
            coverage(fn1),
            coverage(fn2))
        return (VerificationStatus.REFUTED, diff)
    if efilter.user_exc:
        debug('User-level exception found',  *efilter.user_exc)
    return (None, [])

