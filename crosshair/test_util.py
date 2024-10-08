import pathlib
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from crosshair.core import (
    AnalysisMessage,
    Checkable,
    MessageType,
    analyze_function,
    deep_realize,
    run_checkables,
)
from crosshair.options import AnalysisOptionSet
from crosshair.statespace import context_statespace
from crosshair.tracers import NoTracing, ResumedTracing
from crosshair.util import (
    assert_tracing,
    ch_stack,
    debug,
    in_debug,
    is_pure_python,
    name_of_type,
)

ComparableLists = Tuple[List, List]


def simplefs(path: pathlib.Path, files: dict) -> None:
    for name, contents in files.items():
        subpath = path / name
        if isinstance(contents, str):
            with open(subpath, "w") as fh:
                fh.write(contents)
        elif isinstance(contents, dict):
            subpath.mkdir()
            simplefs(subpath, contents)
        else:
            raise Exception("bad input to simplefs")


def check_states(
    fn: Callable,
    expected: MessageType,
    optionset: AnalysisOptionSet = AnalysisOptionSet(),
) -> None:
    if expected == MessageType.POST_FAIL:
        local_opts = AnalysisOptionSet(
            per_condition_timeout=16,
            max_uninteresting_iterations=sys.maxsize,
        )
    elif expected == MessageType.CONFIRMED:
        local_opts = AnalysisOptionSet(
            per_condition_timeout=60,
            per_path_timeout=20,
            max_uninteresting_iterations=sys.maxsize,
        )
    elif expected == MessageType.POST_ERR:
        local_opts = AnalysisOptionSet(max_iterations=20)
    elif expected == MessageType.CANNOT_CONFIRM:
        local_opts = AnalysisOptionSet(
            max_uninteresting_iterations=40,
            per_condition_timeout=3,
        )
    else:
        local_opts = AnalysisOptionSet(
            max_uninteresting_iterations=40,
            per_condition_timeout=5,
        )
    options = local_opts.overlay(optionset)
    found = set([m.state for m in run_checkables(analyze_function(fn, options))])
    assertmsg = f"Got {','.join(map(str, found))} instead of {expected}"
    if not in_debug():
        assertmsg += " (use `pytest -v` to show trace)"
    assert found == {expected}, assertmsg


def check_exec_err(
    fn: Callable, message_prefix="", optionset: AnalysisOptionSet = AnalysisOptionSet()
) -> ComparableLists:
    local_opts = AnalysisOptionSet(max_iterations=20)
    options = local_opts.overlay(optionset)
    messages = run_checkables(analyze_function(fn, options))
    if all(m.message.startswith(message_prefix) for m in messages):
        return ([m.state for m in messages], [MessageType.EXEC_ERR])
    else:
        return (
            [(m.state, m.message) for m in messages],
            [(MessageType.EXEC_ERR, message_prefix)],
        )


def check_messages(checkables: Iterable[Checkable], **kw) -> ComparableLists:
    msgs = run_checkables(checkables)
    if kw.get("state") != MessageType.CONFIRMED:
        # Normally, ignore confirmation messages:
        msgs = [m for m in msgs if m.state != MessageType.CONFIRMED]
    else:
        # When we want CONFIRMED, take the message with the worst status:
        msgs = [max(msgs, key=lambda m: m.state)]
    default_msg = AnalysisMessage(MessageType.CANNOT_CONFIRM, "", "", 0, 0, "")
    msg = msgs[0] if msgs else replace(default_msg)
    fields = (
        "state",
        "message",
        "filename",
        "line",
        "column",
        "traceback",
        "test_fn",
        "condition_src",
    )
    for k in fields:
        if k not in kw:
            default_val = getattr(default_msg, k)
            msg = replace(msg, **{k: default_val})
            kw[k] = default_val
    if msgs:
        msgs[0] = msg
    return (msgs, [AnalysisMessage(**kw)])


def flexible_equal(a, b):
    if type(a) is type(b) and type(a).__eq__ is object.__eq__:
        # If types match and it uses identity-equals, we can't do much. Assume equal.
        return True
    if a != a and b != b:  # handle float('nan')
        return True
    return a == b


@dataclass(eq=False)
class ExecutionResult:
    ret: object  # return value
    exc: Optional[BaseException]  # exception raised, if any
    tb: Optional[str]
    # args after the function terminates:
    post_args: Sequence
    post_kwargs: Mapping[str, object]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExecutionResult):
            return False
        return (
            flexible_equal(self.ret, other.ret)
            and type(self.exc) == type(other.exc)
            and self.post_args == other.post_args
            and self.post_kwargs == other.post_kwargs
        )

    def describe(self, include_postexec=False) -> str:
        ret = ""
        if self.exc:
            exc = self.exc
            exc_type = name_of_type(type(exc))
            tb = self.tb or "(missing traceback)"
            ret = f"exc={exc_type}: {str(exc)} {tb}"
        else:
            ret = f"ret={self.ret!r}"
        if include_postexec:
            a = [repr(a) for a in self.post_args]
            a += [f"{k}={v!r}" for k, v in self.post_kwargs.items()]
            ret += f'  post=({", ".join(a)})'
        return ret


@dataclass
class IterableResult:
    values: tuple
    typ: type


def summarize_execution(
    fn: Callable,
    args: Sequence[object] = (),
    kwargs: Optional[Mapping[str, object]] = None,
    detach_path: bool = True,
) -> ExecutionResult:
    if not kwargs:
        kwargs = {}
    ret: object = None
    exc: Optional[Exception] = None
    tbstr: Optional[str] = None
    try:
        possibly_symbolic_ret = fn(*args, **kwargs)
        if detach_path:
            context_statespace().detach_path()
            detach_path = False
        ret_type = type(possibly_symbolic_ret)
        _ret = deep_realize(possibly_symbolic_ret)
        if hasattr(_ret, "__next__"):
            # Summarize any iterator as the values it produces, plus its type:
            ret = IterableResult(tuple(_ret), ret_type)
        elif callable(_ret) and not is_pure_python(_ret):
            # Summarize C-based callables just based on their type:
            ret = f"C-based callable {type(_ret).__name__}"
        else:
            ret = _ret
        args = deep_realize(args)
        kwargs = deep_realize(kwargs)
    except Exception as e:
        exc = e
        if detach_path:
            context_statespace().detach_path(e)
        exc = deep_realize(exc)
        # NOTE: deep_realize somehow empties the __traceback__ member; re-assign it:
        exc.__traceback__ = e.__traceback__
        tbstr = ch_stack(currently_handling=exc)
        if in_debug():
            debug("hit exception:", type(exc), exc, tbstr)
    return ExecutionResult(ret, exc, tbstr, args, kwargs)


@dataclass
class ResultComparison:
    left: ExecutionResult
    right: ExecutionResult

    def __bool__(self):
        return self.left == self.right and type(self.left) == type(self.right)

    def __repr__(self):
        left, right = self.left, self.right
        include_postexec = left.ret == right.ret and type(left.exc) == type(right.exc)
        return (
            left.describe(include_postexec)
            + "  <--symbolic-vs-concrete-->  "
            + right.describe(include_postexec)
        )


def compare_returns(fn: Callable, *a: object, **kw: object) -> ResultComparison:
    comparison = compare_results(fn, *a, **kw)
    comparison.left.post_args = ()
    comparison.left.post_kwargs = {}
    comparison.right.post_args = ()
    comparison.right.post_kwargs = {}
    return comparison


@assert_tracing(True)
def compare_results(fn: Callable, *a: object, **kw: object) -> ResultComparison:
    original_a = deepcopy(a)
    original_kw = deepcopy(kw)
    symbolic_result = summarize_execution(fn, a, kw)

    concrete_a = deep_realize(original_a)
    concrete_kw = deep_realize(original_kw)

    # Check that realization worked, too:
    with NoTracing():
        labels_and_args = [
            *(
                (f"Argument {idx + 1}", a[idx], arg)
                for idx, arg in enumerate(concrete_a)
            ),
            *((f"Keyword argument '{k}'", kw[k], v) for k, v in concrete_kw.items()),
        ]
        for label, symbolic_arg, concrete_arg in labels_and_args:
            with ResumedTracing():
                symbolic_type = type(symbolic_arg)
                concrete_type = type(concrete_arg)
            true_concrete_type = type(concrete_arg)
            assert (
                true_concrete_type == concrete_type
            ), f"{label} did not realize. It is {true_concrete_type} instead of {concrete_type}."
            assert (
                true_concrete_type == symbolic_type
            ), f"{label} should realize to {symbolic_type}; it is {true_concrete_type} instead."

    with NoTracing():
        concrete_result = summarize_execution(
            fn, concrete_a, concrete_kw, detach_path=False
        )
        debug("concrete_result:", concrete_result)

    ret = ResultComparison(symbolic_result, concrete_result)
    bool(ret)
    return ret
