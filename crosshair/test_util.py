import pathlib
import sys
from collections.abc import Set as AbcSet
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Decimal
from time import process_time
from typing import (
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

from crosshair.core import (
    AnalysisMessage,
    Checkable,
    MessageType,
    Patched,
    analyze_function,
    deep_realize,
    proxy_for_type,
    run_checkables,
    smt_for_unification,
)
from crosshair.options import AnalysisOptionSet
from crosshair.statespace import (
    CallAnalysis,
    IgnoreAttempt,
    RootNode,
    StateSpace,
    StateSpaceContext,
    context_statespace,
)
from crosshair.tracers import COMPOSITE_TRACER, NoTracing, ResumedTracing
from crosshair.util import (
    assert_tracing,
    ch_stack,
    debug,
    in_debug,
    is_iterable,
    name_of_type,
)

ComparableLists = Tuple[List, List]


class _Missing:
    pass


_MISSING = _Missing()


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


class _Unrealizable:
    """
    Sentinel type returned by `safe_deep_realize` when realization fails.

    Treated as equal to anything by `flexible_equal`, so unrealizable values
    do not poison comparisons (e.g. an arg that became a closed I/O stream).
    """

    def __repr__(self) -> str:
        return "<unrealizable>"


UNREALIZABLE = _Unrealizable()


def safe_deep_realize(
    value: object, label: str = "", memo: Optional[Dict] = None
) -> object:
    """
    Best-effort `deep_realize`.

    On any exception, debug-logs the failure and returns `UNREALIZABLE`,
    which `flexible_equal` treats as equal to anything. This is useful for
    diagnostic / comparison contexts (like post-state capture) where we
    don't want a non-realizable concrete object to mask the actual result.

    Pass a shared `memo` to preserve identity across multiple calls when
    realizing several values that may share substructure.
    """
    try:
        return deep_realize(value, memo)
    except Exception as exc:
        debug(
            "Could not realize",
            label or type(value).__name__,
            ":",
            type(exc).__name__,
            exc,
        )
        return UNREALIZABLE


def _is_nan(x: object) -> bool:
    if isinstance(x, float):
        return x != x
    if isinstance(x, Decimal):
        return x.is_nan()
    return False


def flexible_equal(a: object, b: object) -> bool:
    if a is b:
        return True
    if a is UNREALIZABLE or b is UNREALIZABLE:
        return True
    if type(a) is type(b) and type(a).__eq__ is object.__eq__:
        # If types match and it uses identity-equals, we can't do much. Assume equal.
        return True
    if _is_nan(a) and _is_nan(b):
        return True
    if (
        is_iterable(a)
        and not isinstance(a, Collection)
        and is_iterable(b)
        and not isinstance(b, Collection)
    ):  # unsized iterables compare by contents
        a, b = list(a), list(b)  # type: ignore
    if (
        type(a) == type(b)
        and isinstance(a, Collection)
        and not isinstance(a, (str, bytes, AbcSet))
    ):
        # Recursively apply flexible_equal for most containers:
        if len(a) != len(b):  # type: ignore
            return False
        if isinstance(a, Mapping):
            for k, v in a.items():
                if not flexible_equal(v, b.get(k, _MISSING)):  # type: ignore
                    return False
            return True
        else:
            return all(flexible_equal(ai, bi) for ai, bi in zip(a, b))  # type: ignore

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
            and flexible_equal(self.post_args, other.post_args)
            and flexible_equal(self.post_kwargs, other.post_kwargs)
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


# The canonical builtin types -- ``type(x) -> int`` is a stable, identity-eq
# inversion target, so these stay gradable (mirrors ``inputgen.TYPES``).
_GRADABLE_CLASSES = frozenset(
    (int, float, bool, str, bytes, bytearray, list, tuple, dict, set, frozenset)
)


def _is_opaque(v: object, _depth: int = 0) -> bool:
    """True when value-comparing ``v`` (a result value) is meaningless: it is, or
    contains, an identity-eq object -- a callable, hash object, file handle, or an
    arbitrary class instance whose ``__eq__`` is ``object``'s.  ``run_differential``
    SKIPS such outputs (an honest "not checked") rather than reporting a spurious
    C-vs-Python-reimpl divergence, and the support map greys the same ops.

    Classes are split: a *canonical* builtin type (``type(x) -> int``) is a stable,
    identity-eq inversion target, so it stays gradable.  But an arbitrary class --
    e.g. the per-encoding StreamReader that ``codecs.getreader`` returns -- is no
    more checkable than a bare function, so it is opaque like any callable."""
    if v is None:
        return False
    if isinstance(v, type):
        return v not in _GRADABLE_CLASSES
    if callable(v):
        return True
    if isinstance(v, (str, bytes, bytearray, range)):
        return False
    if _depth < 3:
        if isinstance(v, Mapping):
            return any(_is_opaque(x, _depth + 1) for x in v.values())
        if isinstance(v, (list, tuple, set, frozenset)):
            return any(_is_opaque(x, _depth + 1) for x in v)
    return type(v).__eq__ is object.__eq__


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
            ret = IterableResult(tuple(_ret), ret_type)  # type: ignore
        else:
            ret = _ret
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
    # Per-element best-effort realization: an unrealizable arg becomes
    # UNREALIZABLE (compares equal to anything in flexible_equal) without
    # poisoning sibling args. A shared memo preserves identity across
    # arguments that alias the same object.
    memo: Dict = {}
    args = tuple(
        safe_deep_realize(a, label=f"argument {idx + 1}", memo=memo)
        for idx, a in enumerate(args)
    )
    kwargs = {
        k: safe_deep_realize(v, label=f"keyword argument {k!r}", memo=memo)
        for k, v in kwargs.items()
    }
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


# ---------------------------------------------------------------------------
# single-operation differential: run an op with symbolic args pinned to chosen
# concrete values, then concretely, and compare.  A difference in return value,
# exception type, or in-place mutation is a soundness bug.  Shared by
# fuzz_core_test (assert no divergence) and the support measurement (color a cell
# black on any divergence).
# ---------------------------------------------------------------------------
_UNPINNED = object()  # could not pin a symbolic value to a given concrete input
_DIFF_MAX_PIN_ITERS = 80
_DIFF_PIN_TIMEOUT = 10.0

# Operations the differential CANNOT meaningfully check: their forward output
# legitimately depends on unspecified iteration order, isn't value-comparable, or
# the builtin isn't a pure value transform.  A divergence here is NOT a bug -- so
# both the fuzz test (skips them) and the support measurement (won't color them
# black from the differential) consult this list.  Keyed by "<type>.<method>" /
# "<module>.<func>".
DIFFERENTIAL_SKIP: Dict[str, str] = {
    # repr/str of an unordered container -- element order in the string is arbitrary
    "set.__repr__": "set repr element order is unspecified",
    "frozenset.__repr__": "frozenset repr element order is unspecified",
    "dict.__repr__": "dict repr order need not match between symbolic and concrete",
    "set.__str__": "set str element order is unspecified",
    "frozenset.__str__": "frozenset str element order is unspecified",
    "dict.__str__": "dict str order need not match between symbolic and concrete",
    # pop an arbitrary element -- which one is unspecified
    "set.pop": "set.pop removes an arbitrary, unspecified element",
    "dict.popitem": "dict.popitem removes an arbitrary, unspecified item",
    # not value-comparable
    "dict.values": "dict_values has identity equality; not value-comparable",
    # builtins that aren't pure value transforms (identity / code / names / I/O)
    "builtins.id": "identity, not a value",
    "builtins.__import__": "imports a module; not a value transform",
    "builtins.compile": "compiles source; output isn't value-comparable",
    "builtins.exec": "executes code for side effects",
    "builtins.eval": "evaluates a symbolic code string -- not meaningful",
    "builtins.getattr": "attribute access by (symbolic) name",
    "builtins.setattr": "mutates an attribute by name",
    "builtins.delattr": "deletes an attribute by name",
    "builtins.hasattr": "attribute probe by name",
    "builtins.input": "reads input (I/O)",
    "builtins.open": "opens a file (I/O)",
    "builtins.print": "writes output (I/O)",
    "builtins.breakpoint": "enters the debugger",
    "builtins.globals": "introspects the caller's namespace",
    "builtins.locals": "introspects the caller's namespace",
    "builtins.vars": "introspects an object's namespace",
    "builtins.dir": "introspection; order/content not a pure value",
}


@dataclass
class Divergence:
    args: tuple  # the concrete inputs that diverged
    concrete: ExecutionResult
    symbolic: ExecutionResult

    def describe(self) -> str:
        return (
            f"on {self.args!r}:\n"
            f"  concrete: {self.concrete.describe(include_postexec=True)}\n"
            f"  symbolic: {self.symbolic.describe(include_postexec=True)}"
        )


@dataclass
class DiffResult:
    checked: int  # number of inputs actually driven (pinned + run) symbolically
    divergence: Optional[Divergence]  # first symbolic-vs-concrete mismatch, if any


def _proxy_type(v: object) -> object:
    """A (possibly parameterized) type to build a symbolic stand-in for ``v``."""
    t = type(v)
    if t is list:
        return List[_proxy_type(next(iter(v)))] if v else List[int]  # type: ignore
    if t is set:
        return Set[_proxy_type(next(iter(v)))] if v else Set[int]  # type: ignore
    if t is frozenset:
        return FrozenSet[_proxy_type(next(iter(v)))] if v else FrozenSet[int]  # type: ignore
    if t is tuple:
        return Tuple[tuple(_proxy_type(x) for x in v)] if v else Tuple[()]  # type: ignore
    if t is dict:
        if v:
            k, val = next(iter(v.items()))  # type: ignore
            return Dict[_proxy_type(k), _proxy_type(val)]  # type: ignore
        return Dict[int, int]
    return t


def _pin(space: StateSpace, proxies: dict, concrete: dict) -> None:
    """Constrain each symbolic ``proxy`` to equal its concrete value.

    Scalars/strings/lists unify directly via ``smt_for_unification``; anything
    else is pinned by branching (the ``!=`` forks the space, and a non-matching
    branch raises ``IgnoreAttempt`` so the search retries another path)."""
    for name, lit in concrete.items():
        sym = proxies[name]
        with NoTracing():
            eq = smt_for_unification(sym, lit)
        if eq is not None:
            space.add(eq)
            continue
        if isinstance(lit, Collection):
            len_eq = len(lit) == len(sym)  # type: ignore
            if hasattr(len_eq, "var"):  # symbolic length -> add as a solver hint
                space.add(len_eq.var)
        if lit != sym:
            raise IgnoreAttempt(f'symbolic "{name}" != concrete value')
        if repr(lit) != repr(sym):  # dict/set ordering, -0.0 vs 0.0, ...
            raise IgnoreAttempt(f'symbolic "{name}" not repr-equal to concrete value')


def run_symbolic_pinned(
    applier: Callable,
    arg_names: Sequence[str],
    concrete_vals: Sequence[object],
    max_pin_iters: int = _DIFF_MAX_PIN_ITERS,
) -> object:
    """Run ``applier(*symbolic)`` with each symbolic arg pinned to its concrete
    value; return the ExecutionResult, or ``_UNPINNED`` if no path could be
    pinned within the budget."""
    search_root = RootNode()
    with COMPOSITE_TRACER, NoTracing():
        for _itr in range(1, max_pin_iters + 1):
            space = StateSpace(
                process_time() + _DIFF_PIN_TIMEOUT, 3.0, search_root=search_root
            )
            try:
                with Patched(), StateSpaceContext(space):
                    proxies = {
                        n: proxy_for_type(_proxy_type(v), n)
                        for n, v in zip(arg_names, concrete_vals)
                    }
                    with ResumedTracing():
                        _pin(space, proxies, dict(zip(arg_names, concrete_vals)))
                        return summarize_execution(
                            applier, [proxies[n] for n in arg_names]
                        )
            except IgnoreAttempt:
                pass  # this path didn't match the concrete value; try another
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                # CrosshairUnsupported / UnknownSatisfiability / CrossHairInternal
                # etc. mean the engine couldn't model this op here -- "couldn't
                # check", not a divergence (the op's OWN exceptions are Exception
                # subclasses, caught inside summarize_execution and compared).
                return _UNPINNED
            _analysis, exhausted = space.bubble_status(CallAnalysis())
            if exhausted:
                return _UNPINNED
    return _UNPINNED


def run_differential(
    fn: Callable,
    expr: str,
    arg_names: Sequence[str],
    eval_globals: Mapping[str, object],
    k: int = 3,
    seed: int = 0,
    max_pin_iters: int = _DIFF_MAX_PIN_ITERS,
) -> DiffResult:
    """Drive one operation on up to ``k`` valid inputs, comparing a symbolic run
    (args pinned to the input) against a concrete run.  Returns a ``DiffResult``
    with the count of inputs actually driven and the first ``Divergence`` (or
    None if all matched).  ``max_pin_iters`` bounds the per-input pin search; use
    a small value when speed matters more than pinning every container shape (an
    input that can't pin in budget is simply skipped).

    ``expr`` is eval'd over ``arg_names`` (plus ``eval_globals``); see
    ``crosshair.inputgen.op_call``/``func_call`` for building these."""
    from crosshair.inputgen import valid_inputs

    def applier(*vs):
        return eval(expr, dict(eval_globals), dict(zip(arg_names, vs)))

    inputs = [t for t in valid_inputs(fn, k=k, seed=seed) if len(t) == len(arg_names)]
    checked = 0
    for vals in inputs:
        debug("differential:", expr, "with", vals)
        symbolic = run_symbolic_pinned(applier, arg_names, vals, max_pin_iters)
        if symbolic is _UNPINNED:
            continue
        concrete = summarize_execution(applier, deepcopy(list(vals)), detach_path=False)
        if concrete.exc is None and _is_opaque(concrete.ret):
            # Output is (or contains) an identity-eq value -- not value-comparable,
            # so a symbolic-vs-concrete mismatch here is spurious (C vs Python
            # reimpl).  Don't count it; the support map greys these too.
            continue
        checked += 1
        if concrete != symbolic:
            return DiffResult(
                checked,
                Divergence(tuple(vals), concrete, cast(ExecutionResult, symbolic)),
            )
    return DiffResult(checked, None)
