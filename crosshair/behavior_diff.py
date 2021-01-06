import contextlib
import copy
import inspect
import itertools
import sys
import time
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from crosshair import core_and_libs

from crosshair.condition_parser import resolve_signature
from crosshair.statespace import CallAnalysis
from crosshair.statespace import SinglePathNode
from crosshair.statespace import TrackingStateSpace
from crosshair.statespace import StateSpaceContext
from crosshair.statespace import VerificationStatus
from crosshair.core import gen_args
from crosshair.core import Patched
from crosshair.core import AnalysisOptions
from crosshair.core import ExceptionFilter
from crosshair.util import debug
from crosshair.util import IgnoreAttempt
from crosshair.util import UnexploredPath
from crosshair.util import measure_fn_coverage

def describe_behavior(
        fn: Callable, args: inspect.BoundArguments) -> Tuple[object, Optional[Exception]]:
    with ExceptionFilter() as efilter:
        ret = fn(*args.args, **args.kwargs)
        return (ret, None)
    if efilter.user_exc is not None:
        debug('user-level exception found', *efilter.user_exc)
        return (None, efilter.user_exc[0])
    if efilter.ignore:
        return (None, IgnoreAttempt())
    assert False

def diff_behavior(fn1: Callable, fn2: Callable, options: AnalysisOptions) -> Iterable[str]:
    sig1, resolution_err = resolve_signature(fn1)
    sig2, resolution_err = resolve_signature(fn2)
    debug('Resolved signature:', sig1)
    if sig1 is None:
        yield f'Unable to get signature of {fn1}'
        return
    if sig2 is None:
        yield f'Unable to get signature of {fn2}'
        return
    coverage_manager = (measure_fn_coverage(fn1) if sys.gettrace() is None else
                        contextlib.nullcontext(lambda:None))
    half1, half2 = options.split_limits(0.5)
    with Patched(enabled=lambda: True), coverage_manager as coverage:
        # We attempt both orderings of functions. This helps by:
        # (1) avoiding code path explosions in one of the functions
        # (2) using both signatures (in case they differ)
        yield from diff_behavior_with_signature(fn1, fn2, sig1, half1)
        yield from diff_behavior_with_signature(fn2, fn1, sig2, half2)
    coverage_results = coverage()
    if coverage_results is not None:
        yield f'(achieved {round(coverage_results.opcode_coverage * 100)}% opcode coverage)'

def diff_behavior_with_signature(
        fn1: Callable, fn2: Callable, sig: inspect.Signature, options: AnalysisOptions) -> Iterable[str]:
    search_root = SinglePathNode(True)
    condition_start = time.monotonic()
    for i in range(1, options.max_iterations):
        debug('Iteration ', i)
        itr_start = time.monotonic()
        if itr_start > condition_start + options.per_condition_timeout:
            yield f'(stopping due to --per_condition_timeout={options.per_condition_timeout})'
            break
        space = TrackingStateSpace(
            execution_deadline=itr_start + options.per_path_timeout,
            model_check_timeout=options.per_path_timeout / 2,
            search_root=search_root)
        with StateSpaceContext(space):
            (verification_status, output) = run_iteration(fn1, fn2, sig, space)
            debug('Verification status:', verification_status)
            top_analysis, space_exhausted = space.bubble_status(CallAnalysis(verification_status))
            if top_analysis and top_analysis.verification_status == VerificationStatus.CONFIRMED:
                break
            yield from output

def run_iteration(
        fn1: Callable,
        fn2: Callable,
        sig: inspect.Signature,
        space: TrackingStateSpace) -> Tuple[
            Optional[VerificationStatus],
            List[str]]:
    original_args = gen_args(sig)
    args1 = copy.deepcopy(original_args)
    args2 = copy.deepcopy(original_args)
    try:
        with ExceptionFilter() as efilter:
            ret1, exc1 = describe_behavior(fn1, args1)
            ret2, exc2 = describe_behavior(fn2, args2)
            err1, err2 = str(exc1), str(exc2)
            if ret1 == ret2 and err1 == err2:
                debug('Functions equivalent')
                return (VerificationStatus.CONFIRMED, [])
            debug('Functions differ')
            inputs = ', '.join(f'{k}={v!r}' for k,v in original_args.arguments.items())
            output = [f'  given: ({inputs})']
            if ret1 != ret2:
                output.append(f'- returns: {ret1}')
                output.append(f'+ returns: {ret2}')
            if err1 != err2:
                output.append(f'- raises: {err1}')
                output.append(f'+ raises: {err2}')
            space.check_deferred_assumptions()
            return (VerificationStatus.REFUTED, output)
        if efilter.user_exc:
            debug('User-level exception found', *efilter.user_exc)
    except UnexploredPath:
        pass
    return (None, [])

def main():
    diff_behavior(AnalysisOptions())

if __name__ == '__main__':
    main()
