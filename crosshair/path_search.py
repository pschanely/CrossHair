import enum
import traceback
from dataclasses import dataclass
from inspect import BoundArguments
from typing import Callable, Optional, Type

from crosshair.copyext import CopyMode, deepcopyext
from crosshair.core import ExceptionFilter, LazyCreationRepr, explore_paths
from crosshair.fnutil import FunctionInfo
from crosshair.libimpl.builtinslib import SymbolicInt
from crosshair.options import AnalysisOptions
from crosshair.statespace import RootNode, StateSpace, context_statespace
from crosshair.tracers import CoverageResult, NoTracing, ResumedTracing
from crosshair.util import (
    CrosshairInternal,
    EvalFriendlyReprContext,
    debug,
    format_boundargs_as_dictionary,
    test_stack,
)


class OptimizationKind(enum.Enum):
    SIMPLIFY = "SIMPLIFY"
    NONE = "NONE"
    MINIMIZE_INT = "MINIMIZE_INT"


@dataclass
class PathSummary:
    args: BoundArguments
    result: object
    exc: Optional[Type[BaseException]]
    post_args: BoundArguments
    coverage: CoverageResult


def realize_args(args: BoundArguments) -> str:
    space = context_statespace()
    reprer = space.extra(LazyCreationRepr)
    args = reprer.deep_realize(args)
    with EvalFriendlyReprContext(reprer.reprs) as ctx:
        args_string = format_boundargs_as_dictionary(args)
    return ctx.cleanup(args_string)


def path_search(
    ctxfn: FunctionInfo,
    options: AnalysisOptions,
    argument_formatter: Optional[Callable[[BoundArguments], str]],
    optimization_kind: OptimizationKind,
    optimize_fn: Optional[Callable],
    on_example: Callable[[str], None],
) -> None:

    if argument_formatter is None:
        checked_format = realize_args
    else:

        def checked_format(args: BoundArguments) -> str:
            assert argument_formatter is not None
            args = deepcopyext(args, CopyMode.REALIZE, {})
            try:
                return argument_formatter(args)
            except Exception as exc:
                raise CrosshairInternal(str(exc)) from exc

    if optimization_kind == OptimizationKind.SIMPLIFY:
        assert optimize_fn is None

        def scorechar(codepoint: int):
            if codepoint >= ord("a"):
                return codepoint - ord("a")
            elif codepoint >= ord("0"):
                return 30 + codepoint - ord("0")
            else:
                return 100 + codepoint

        def shrinkscore(ret, args: BoundArguments):
            with NoTracing():
                reprstr = checked_format(args)
            return len(reprstr) * 1000 + sum(scorechar(ord(ch)) for ch in reprstr)

        optimization_kind = OptimizationKind.MINIMIZE_INT
        optimize_fn = shrinkscore

    fn, sig = ctxfn.callable()
    search_root = RootNode()

    best_input: Optional[str] = None
    best_score: Optional[int] = None
    _optimize_fn = optimize_fn or (lambda _ret, args: fn(*args.args, **args.kwargs))

    def on_path_complete(
        space: StateSpace,
        pre_args: BoundArguments,
        post_args: BoundArguments,
        ret,
        exc: Optional[BaseException],
        exc_stack: Optional[traceback.StackSummary],
    ) -> bool:
        nonlocal best_input, best_score
        with NoTracing():
            if exc is not None:
                debug(
                    "Aborting path, hit exception",
                    type(exc),
                    exc,
                    test_stack(exc_stack),
                )
                return False
            debug("Path succeeded")
            if optimization_kind == OptimizationKind.NONE:
                best_input = checked_format(pre_args)
                debug("Found input:", best_input)
                on_example(best_input)
                return True
        with NoTracing(), ExceptionFilter() as efilter:
            with ResumedTracing():
                score = _optimize_fn(ret, pre_args)
                space.detach_path()
            smt_score = SymbolicInt._coerce_to_smt_sort(score)
            if smt_score is None:
                debug("non integer score returned; ignoring.")
                return False
            if space.smt_fork(smt_score < 0, probability_true=0.0):
                debug("Score was leass than zero; ignoring.")
                return False
            if best_score is not None and space.smt_fork(
                smt_score >= best_score, probability_true=0.0
            ):
                debug("Does not beat the best score of", best_score)
                return False
            known_min = 0
            known_max = best_score
            test = 1000 if known_max is None else (known_max // 2)
            while True:
                debug(known_min, test, known_max)
                if space.smt_fork(smt_score < test, probability_true=1.0):
                    known_max = test - 1
                else:
                    known_min = test
                    if known_max is None:
                        test = known_min * 100
                        continue
                if known_min == known_max:
                    best_score = known_min
                    best_input = checked_format(pre_args)
                    break
                test = (known_min + known_max + 1) // 2
            debug("Minimized score to", best_score)
            debug("For input:", best_input)
            on_example(best_input)
            return best_score == 0
        debug("Skipping path (failure during scoring)", efilter.user_exc)
        return False

    explore_paths(
        lambda ba: fn(*ba.args, **ba.kwargs),
        sig,
        options,
        search_root,
        on_path_complete,
    )
