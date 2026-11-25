import enum
import traceback
from dataclasses import dataclass
from inspect import BoundArguments
from typing import Callable, Optional, Type

from crosshair import smt
from crosshair.copyext import CopyMode, deepcopyext
from crosshair.core import (
    ExceptionFilter,
    LazyCreationRepr,
    deep_realize,
    explore_paths,
)
from crosshair.fnutil import FunctionInfo
from crosshair.libimpl.builtinslib import SymbolicInt
from crosshair.options import AnalysisOptions
from crosshair.statespace import RootNode, StateSpace, context_statespace
from crosshair.tracers import (
    COMPOSITE_TRACER,
    CoverageResult,
    NoTracing,
    ResumedTracing,
)
from crosshair.util import (
    ControlFlowException,
    CrossHairInternal,
    ch_stack,
    debug,
    format_boundargs_as_dictionary,
)


class OptimizationKind(enum.Enum):
    SIMPLIFY = "SIMPLIFY"
    # TODO: simplify mode is quite dumb atm; partly because eval_friendly_format
    # realizes prior to generating the string.
    NONE = "NONE"
    MINIMIZE_INT = "MINIMIZE_INT"


@dataclass
class _Example:
    args: BoundArguments
    formatted_args: str
    score: int


@dataclass
class PathSummary:
    args: BoundArguments
    result: object
    exc: Optional[Type[BaseException]]
    post_args: BoundArguments
    coverage: CoverageResult


def path_search(
    ctxfn: FunctionInfo,
    options: AnalysisOptions,
    argument_formatter: Optional[Callable[[BoundArguments], str]],
    optimization_kind: OptimizationKind,
    optimize_fn: Optional[Callable],
    on_example: Callable[[str], None],
    on_nondeterminism: Optional[Callable[[], None]] = None,
) -> None:

    if argument_formatter is None:
        checked_format = lambda args: (
            context_statespace()
            .extra(LazyCreationRepr)
            .eval_friendly_format(args, format_boundargs_as_dictionary)
        )
    else:

        def checked_format(args: BoundArguments) -> str:
            assert argument_formatter is not None
            args = deepcopyext(args, CopyMode.REALIZE, {})
            try:
                return argument_formatter(args)
            except Exception as exc:
                raise CrossHairInternal(str(exc)) from exc

    score_concretely = optimization_kind == OptimizationKind.MINIMIZE_INT
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
            reprstr = checked_format(args)
            return len(reprstr) * 1000 + sum(scorechar(ord(ch)) for ch in reprstr)

        optimization_kind = OptimizationKind.MINIMIZE_INT
        optimize_fn = shrinkscore

    fn, sig = ctxfn.callable()
    search_root = RootNode()

    best_score: Optional[int] = None
    _optimize_fn = optimize_fn or (lambda _ret, args: fn(*args.args, **args.kwargs))

    def on_path_complete(
        space: StateSpace,
        pre_args: BoundArguments,
        post_args: BoundArguments,
        ret,
        exc: Optional[BaseException],
        exc_stack: Optional[traceback.StackSummary],
    ) -> Optional[_Example]:
        with NoTracing():
            if exc is not None:
                debug(
                    "Aborting path, hit exception",
                    type(exc),
                    exc,
                    ch_stack(exc_stack),
                )
                return None
            debug("Path succeeded")
            if optimization_kind == OptimizationKind.NONE:
                with ResumedTracing():
                    space.detach_path()
                    formatted = checked_format(pre_args)
                return _Example(deep_realize(pre_args), formatted, 0)
        with NoTracing(), ExceptionFilter() as efilter:
            with ResumedTracing():
                score = _optimize_fn(ret, pre_args)
                space.detach_path()
            smt_score = SymbolicInt._coerce_to_smt_sort(score)
            if smt_score is None:
                debug("non integer score returned; ignoring.")
                return None
            if space.smt_fork(smt.lt(smt_score, smt.int_val(0)), probability_true=0.0):
                debug("Score was leass than zero; ignoring.")
                return None
            if best_score is not None and space.smt_fork(
                smt.ge(smt_score, smt.int_val(best_score)), probability_true=0.0
            ):
                debug("Does not beat the best score of", best_score)
                return None
            known_min = 0
            known_max = best_score
            test = 1000 if known_max is None else (known_max // 2)
            while True:
                debug(known_min, test, known_max)
                if space.smt_fork(
                    smt.lt(smt_score, smt.int_val(test)), probability_true=1.0
                ):
                    known_max = test - 1
                else:
                    known_min = test
                    if known_max is None:
                        test = known_min * 100
                        continue
                if known_min == known_max:
                    break
                test = (known_min + known_max + 1) // 2
            debug("Minimized score to", known_min)
            with ResumedTracing():
                formatted = checked_format(pre_args)
            return _Example(deep_realize(pre_args), formatted, known_min)
        debug("Skipping path (failure during scoring)", efilter.user_exc)
        return None

    def concrete_score(example: _Example) -> Optional[int]:
        with NoTracing():
            args = deepcopyext(example.args, CopyMode.REGULAR, {})
        with ExceptionFilter() as efilter:
            ret = fn(*args.args, **args.kwargs)
            if not score_concretely:
                return example.score
            score = _optimize_fn(ret, example.args)
            if isinstance(score, int) and score >= 0:
                return score
            debug("Concrete score is not a non-negative integer:", score)
            return None
        debug("Concrete call failed:", efilter.user_exc)
        return None

    def accept_example(example: _Example) -> bool:
        nonlocal best_score
        try:
            with COMPOSITE_TRACER:
                score = concrete_score(example)
        except ControlFlowException as exc:
            debug("Using symbolic score (unable to replay):", exc)
            score = example.score
        if score is None or (best_score is not None and score >= best_score):
            debug("Rejecting example with concrete score", score)
            return False
        best_score = score
        debug("For input:", example.formatted_args)
        on_example(example.formatted_args)
        return best_score == 0

    explore_paths(
        lambda ba: fn(*ba.args, **ba.kwargs),
        sig,
        options,
        search_root,
        on_path_complete,
        on_nondeterminism=on_nondeterminism,
        consider_candidate=accept_example,
    )
