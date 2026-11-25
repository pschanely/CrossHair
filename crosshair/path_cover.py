import builtins
import enum
import re
import traceback
from dataclasses import dataclass, replace
from inspect import BoundArguments
from typing import Callable, List, Optional, Set, TextIO, Tuple, Type

from crosshair.core import (
    ExceptionFilter,
    LazyCreationRepr,
    Patched,
    deep_realize,
    explore_paths,
    realize,
    replay_concretely,
)
from crosshair.fnutil import FunctionInfo
from crosshair.options import AnalysisOptions
from crosshair.statespace import RootNode, StateSpace, context_statespace
from crosshair.tracers import (
    CoverageResult,
    CoverageTracingModule,
    NoTracing,
    PushedModule,
)
from crosshair.util import (
    EvalFriendlyReprContext,
    ReferencedIdentifier,
    ch_stack,
    debug,
    eval_friendly_repr,
    format_boundargs,
    name_of_type,
)


class CoverageType(enum.Enum):
    OPCODE = "OPCODE"
    PATH = "PATH"


@dataclass
class PathSummary:
    args: BoundArguments
    formatted_args: str
    result: str
    exc: Optional[Type[BaseException]]
    exc_message: Optional[str]
    post_args: BoundArguments
    coverage: CoverageResult
    references: Set[ReferencedIdentifier]


def path_cover(
    ctxfn: FunctionInfo,
    options: AnalysisOptions,
    coverage_type: CoverageType,
    arg_formatter: Callable[[BoundArguments], str] = format_boundargs,
    on_nondeterminism: Optional[Callable[[], None]] = None,
) -> List[PathSummary]:
    fn, sig = ctxfn.callable()
    while getattr(fn, "__wrapped__", None):
        # Usually we don't want to run decorator code. (and we certainly don't want
        # to measure coverage on the decorator rather than the real body) Unwrap:
        fn = fn.__wrapped__  # type: ignore
    search_root = RootNode()

    paths: List[Tuple[PathSummary, str]] = []
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
        exc: Optional[BaseException],
        exc_stack: Optional[traceback.StackSummary],
    ) -> None:
        with ExceptionFilter() as efilter:
            space.detach_path()

            reprer = context_statespace().extra(LazyCreationRepr)
            formatted_pre_args = reprer.eval_friendly_format(pre_args, arg_formatter)

            pre_args = deep_realize(pre_args)
            post_args = deep_realize(post_args)
            with NoTracing():
                plain_result = eval_friendly_repr(deep_realize(ret))
            ret = reprer.eval_friendly_format(ret, lambda x: builtins.repr(x))
            references = reprer.repr_references

            cov = coverage.get_results(fn)
            debug("Realized args:", formatted_pre_args)
            exc_message = None
            if exc is not None:
                debug("user-level exception found", type(exc), exc, ch_stack(exc_stack))
                exc_message = realize(str(exc)) if len(exc.args) > 0 else None
            else:
                debug("Realized return:", ret)
            summary = PathSummary(
                pre_args,
                formatted_pre_args,
                ret,
                None if exc is None else type(exc),
                exc_message,
                post_args,
                cov,
                references,
            )
            paths.append((summary, plain_result))
            return
        debug("Skipping path (failed to realize values)", efilter.user_exc)

    explore_paths(
        run_path,
        sig,
        options,
        search_root,
        on_path_complete,
        on_nondeterminism=on_nondeterminism,
    )

    def replay(path: PathSummary, plain_result: str) -> PathSummary:
        def summarize(args: BoundArguments) -> PathSummary:
            with ExceptionFilter() as efilter:
                ret = run_path(args)
            if efilter.ignore:
                debug("Keeping symbolic path summary (replay was ignored)")
                return path
            replayed = replace(path, post_args=args, coverage=coverage.get_results(fn))
            if efilter.user_exc is not None:
                exc = efilter.user_exc[0]
                exc_message = str(exc) if len(exc.args) > 0 else None
                return replace(
                    replayed, result="None", exc=type(exc), exc_message=exc_message
                )
            with Patched(), EvalFriendlyReprContext() as ctx:
                result = ctx.cleanup(builtins.repr(ret))
            if path.exc is None and result == plain_result:
                return replayed
            debug("Concrete replay differs; it returns", result)
            return replace(
                replayed,
                result=result,
                exc=None,
                exc_message=None,
                references=path.references | ctx.repr_references,
            )

        return replay_concretely(options, path.args, summarize, path)

    paths = [(replay(path, plain_result), plain_result) for path, plain_result in paths]
    summaries = [path for path, _ in paths]

    opcodes_found: Set[int] = set()
    selected: List[PathSummary] = []
    while summaries:
        next_best = max(
            summaries,
            key=lambda p: (
                len(p.coverage.offsets_covered - opcodes_found),  # high coverage
                -len(p.formatted_args),  # with small input size
            ),
        )
        cur_offsets = next_best.coverage.offsets_covered
        if coverage_type == CoverageType.OPCODE:
            debug("Next best path covers these opcode offsets:", cur_offsets)
            if len(cur_offsets - opcodes_found) == 0:
                break
        selected.append(next_best)
        opcodes_found |= cur_offsets
        summaries = [p for p in summaries if p is not next_best]
    return selected


def output_argument_dictionary_paths(
    fn: Callable, paths: List[PathSummary], stdout: TextIO, stderr: TextIO
):
    for path in paths:
        stdout.write(path.formatted_args + "\n")
    stdout.flush()


def output_eval_exression_paths(
    fn: Callable, paths: List[PathSummary], stdout: TextIO, stderr: TextIO
):
    for path in paths:
        stdout.write(fn.__name__ + "(" + path.formatted_args + ")\n")
    stdout.flush()


def import_statements_for_references(references: Set[ReferencedIdentifier]) -> Set[str]:
    imports: Set[str] = set()
    for ref in references:
        if ref.modulename == "builtins":
            continue
        if "." in ref.qualname:
            class_name, _ = ref.qualname.split(".", 1)
            imports.add(f"from {ref.modulename} import {class_name}")
        else:
            imports.add(f"from {ref.modulename} import {ref.qualname}")
    return imports


def output_pytest_paths(
    fn: Callable, paths: List[PathSummary]
) -> Tuple[Set[str], List[str]]:
    fn_name = fn.__qualname__
    lines: List[str] = []
    references = {ReferencedIdentifier(fn.__module__, fn_name)}
    imports: Set[str] = set()

    name_with_underscores = fn_name.replace(".", "_")
    for idx, path in enumerate(paths):
        test_name_suffix = "" if idx == 0 else "_" + str(idx + 1)
        exec_fn = f"{fn_name}({path.formatted_args})"
        lines.append(f"def test_{name_with_underscores}{test_name_suffix}():")
        if path.exc is None:
            lines.append(f"    assert {exec_fn} == {path.result}")
        else:
            imports.add("import pytest")
            if path.exc_message is not None:
                lines.append(
                    f"    with pytest.raises({name_of_type(path.exc)}, match={repr(re.escape(path.exc_message))}):"
                )
            else:
                lines.append(f"    with pytest.raises({name_of_type(path.exc)}):")
            lines.append(f"        {exec_fn}")
        lines.append("")
        references |= path.references
    imports |= import_statements_for_references(references)
    return (imports, lines)
