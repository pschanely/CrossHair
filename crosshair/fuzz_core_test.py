import builtins
import copy
import enum
import random
import re
import time
import traceback
from collections.abc import Hashable, Mapping, Sized
from inspect import (
    Parameter,
    Signature,
    getmembers,
    isbuiltin,
    isfunction,
    ismethoddescriptor,
)
from types import ModuleType
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
)

import pytest

import crosshair.core_and_libs  # ensure patches/plugins are loaded
from crosshair.abcstring import AbcString
from crosshair.core import Patched, deep_realize, proxy_for_type, realize
from crosshair.fnutil import resolve_signature
from crosshair.libimpl.builtinslib import origin_of
from crosshair.statespace import (
    CallAnalysis,
    CrossHairInternal,
    IgnoreAttempt,
    RootNode,
    StateSpace,
    StateSpaceContext,
)
from crosshair.stubs_parser import signature_from_stubs
from crosshair.tracers import COMPOSITE_TRACER, NoTracing, ResumedTracing
from crosshair.util import CrosshairUnsupported, debug, type_args_of

FUZZ_SEED = 1348

T = TypeVar("T")


def simple_name(value: object) -> str:
    return re.sub(r"[\W_]+", "_", str(value))


IMMUTABLE_BASE_TYPES = [bool, int, float, str, frozenset]
ALL_BASE_TYPES = IMMUTABLE_BASE_TYPES + [set, dict, list]


def gen_type(r: random.Random, type_root: Type) -> type:
    if type_root is Hashable:
        base = r.choice(IMMUTABLE_BASE_TYPES)
    elif type_root is object:
        base = r.choice(ALL_BASE_TYPES)
    else:
        base = type_root
    if base is dict:
        kt = gen_type(r, Hashable)
        vt = gen_type(r, object)
        return Dict[kt, vt]  # type: ignore
    elif base is list:
        return List[gen_type(r, object)]  # type: ignore
    elif base is set:
        return Set[gen_type(r, Hashable)]  # type: ignore
    elif base is frozenset:
        return FrozenSet[gen_type(r, Hashable)]  # type: ignore
    else:
        return base


# TODO: consider replacing this with typeshed someday!
_SIGNATURE_OVERRIDES = {
    getattr: Signature(
        [
            Parameter("obj", Parameter.POSITIONAL_ONLY, annotation=object),
            Parameter("attr", Parameter.POSITIONAL_ONLY, annotation=str),
            Parameter("default", Parameter.POSITIONAL_ONLY, annotation=object),
        ]
    ),
    dict.items: Signature(),
    dict.keys: Signature(),
    # TODO: fuzz test values() somehow. items() and keys() are sets and
    # therefore comparable -- not values() though:
    # dict.values: Signature(),
    dict.clear: Signature(),
    dict.copy: Signature(),
    dict.pop: Signature(
        [
            Parameter("k", Parameter.POSITIONAL_ONLY, annotation=object),
            Parameter("d", Parameter.POSITIONAL_ONLY, annotation=object),
        ]
    ),
    dict.update: Signature(
        [Parameter("d", Parameter.POSITIONAL_ONLY, annotation=dict)]
    ),
}


def value_for_type(typ: Type, r: random.Random) -> object:
    """
    post: isinstance(_, typ)
    """
    origin = origin_of(typ)
    type_args = type_args_of(typ)
    if typ is bool:
        return r.choice([True, False])
    elif typ is int:
        return r.choice([-1, 0, 1, 2, 10])
    elif typ is float:
        return r.choice([-1.0, 0.0, 1.0, 2.0, 10.0])  # TODO: Inf, NaN
    elif typ is str:
        return r.choice(
            ["", "x", "0", "xyz"]
        )  # , '\0']) # TODO: null does not work properly yet
    elif typ is bytes:
        return r.choice([b"", b"ab", b"abc", b"\x00"])
    elif origin in (list, set, frozenset):
        (item_type,) = type_args
        items = []
        for _ in range(r.choice([0, 0, 0, 1, 1, 2])):
            items.append(value_for_type(item_type, r))
        return origin(items)
    elif origin is dict:
        (key_type, val_type) = type_args
        ret = {}
        for _ in range(r.choice([0, 0, 0, 1, 1, 1, 2])):
            ret[value_for_type(key_type, r)] = value_for_type(val_type, r)  # type: ignore
        return ret
    raise NotImplementedError


def get_signature(method) -> Optional[Signature]:
    override_sig = _SIGNATURE_OVERRIDES.get(method, None)
    if override_sig:
        return override_sig
    sig = resolve_signature(method)
    if isinstance(sig, Signature):
        return sig
    stub_sigs, stub_sigs_valid = signature_from_stubs(method)
    if stub_sigs_valid and len(stub_sigs) == 1:
        debug("using signature from stubs:", stub_sigs[0])
        return stub_sigs[0]
    return None


def get_testable_members(classes: Iterable[type]) -> Iterable[Tuple[type, str]]:
    for cls in classes:
        for method_name, method in getmembers(cls):
            if method_name.startswith("__"):
                continue
            if not (isfunction(method) or ismethoddescriptor(method)):
                # TODO: fuzz test class/staticmethods with symbolic args
                continue
            yield cls, method_name


class TrialStatus(enum.Enum):
    NORMAL = 0
    UNSUPPORTED = 1


class FuzzTester:
    r: random.Random

    def __init__(self, seed):
        self.r = random.Random(seed)

    def symbolic_run(
        self,
        fn: Callable[[StateSpace, Dict[str, object]], object],
        typed_args: Dict[str, type],
    ) -> Tuple[
        object,  # return value
        Optional[Dict[str, object]],  # arguments after execution
        Optional[BaseException],  # exception thrown, if any
        StateSpace,
    ]:
        search_root = RootNode()
        with COMPOSITE_TRACER, NoTracing():
            for itr in range(1, 200):
                debug("iteration", itr)
                space = StateSpace(
                    time.monotonic() + 30.0, 3.0, search_root=search_root
                )
                symbolic_args = {}
                try:
                    with Patched(), StateSpaceContext(space):
                        symbolic_args = {
                            name: proxy_for_type(typ, name)
                            for name, typ in typed_args.items()
                        }
                        with ResumedTracing():
                            ret = fn(space, symbolic_args)
                            ret = (deep_realize(ret), symbolic_args, None, space)
                            space.detach_path()
                        return ret
                except IgnoreAttempt as e:
                    debug("ignore iteration attempt: ", str(e))
                except Exception as e:
                    debug(
                        "exception during symbolic execution:", traceback.format_exc()
                    )
                    return (None, symbolic_args, e, space)
                top_analysis, space_exhausted = space.bubble_status(CallAnalysis())
                if space_exhausted:
                    return (
                        None,
                        symbolic_args,
                        CrossHairInternal(f"exhausted after {itr} iterations"),
                        space,
                    )
        raise CrossHairInternal("Unable to find a successful symbolic execution")

    def runexpr(self, expr, bindings):
        try:
            return (eval(expr, {}, bindings), None)
        except Exception as e:
            debug(f'eval of "{expr}" produced exception {type(e)}: {e}')
            return (None, e)

    def run_function_trials(
        self, fns: Sequence[Tuple[str, Callable]], num_trials: int
    ) -> None:
        for fn_name, fn in fns:
            debug("Checking function", fn_name)
            sig = get_signature(fn)
            if not sig:
                debug("Skipping", fn_name, " - unable to inspect signature")
                continue
            arg_names = [chr(ord("a") + i) for i in range(len(sig.parameters))]
            arg_expr_strings = [
                (a if p.kind != Parameter.KEYWORD_ONLY else f"{p.name}={a}")
                for a, p in zip(arg_names, list(sig.parameters.values()))
            ]
            expr_str = fn_name + "(" + ",".join(arg_expr_strings) + ")"
            arg_type_roots = {name: object for name in arg_names}
            for trial_num in range(num_trials):
                self.run_trial(expr_str, arg_type_roots)

    def run_trial(self, expr_str: str, arg_type_roots: Dict[str, Type]) -> TrialStatus:
        expr = expr_str.format(*arg_type_roots.keys())
        typed_args = {
            name: gen_type(self.r, type_root)
            for name, type_root in arg_type_roots.items()
        }
        literal_args = {
            name: value_for_type(typ, self.r) for name, typ in typed_args.items()
        }

        def symbolic_checker(
            space: StateSpace, symbolic_args: Dict[str, object]
        ) -> object:
            for name in typed_args.keys():
                literal, symbolic = literal_args[name], symbolic_args[name]
                if literal != symbolic:
                    raise IgnoreAttempt(
                        f'symbolic "{name}" not equal to literal "{name}"'
                    )
                if repr(literal) != repr(symbolic):
                    # dict/set ordering, -0.0 vs 0.0, etc
                    raise IgnoreAttempt(
                        f'symbolic "{name}" not repr-equal to literal "{name}"'
                    )
            return eval(expr, symbolic_args.copy())

        debug(f"  =====  {expr} with {literal_args}  =====  ")
        compile(expr, "<string>", "eval")
        postexec_literal_args = copy.deepcopy(literal_args)
        literal_ret, literal_exc = self.runexpr(expr, postexec_literal_args)
        (
            symbolic_ret,
            postexec_symbolic_args,
            symbolic_exc,
            space,
        ) = self.symbolic_run(symbolic_checker, typed_args)
        if isinstance(symbolic_exc, CrosshairUnsupported):
            return TrialStatus.UNSUPPORTED
        with Patched(), StateSpaceContext(space), COMPOSITE_TRACER, NoTracing():
            # compare iterators as the values they produce:
            with ResumedTracing():
                if isinstance(literal_ret, Iterable) and isinstance(
                    symbolic_ret, Iterable
                ):
                    literal_ret = list(literal_ret)
                    symbolic_ret = list(symbolic_ret)
            postexec_symbolic_args = deep_realize(postexec_symbolic_args)
            symbolic_ret = deep_realize(symbolic_ret)
            symbolic_exc = deep_realize(symbolic_exc)
            rets_differ = realize(bool(literal_ret != symbolic_ret))
            postexec_args_differ = realize(
                bool(postexec_literal_args != postexec_symbolic_args)
            )
            if (
                rets_differ
                or postexec_args_differ
                or type(literal_exc) != type(symbolic_exc)
            ):
                debug(f"  *****  BEGIN FAILURE FOR {expr} WITH {literal_args}  *****  ")
                debug(
                    f"  *****  Expected return: {literal_ret} (exc: {type(literal_exc)} {literal_exc})"
                )
                debug(f"  *****    postexec args: {postexec_literal_args}")
                debug(
                    f"  *****  Symbolic return: {symbolic_ret} (exc: {type(symbolic_exc)} {symbolic_exc})"
                )
                debug(f"  *****    postexec args: {postexec_symbolic_args}")
                debug(f"  *****  END FAILURE FOR {expr}  *****  ")
                assert literal_ret == symbolic_ret
                assert False, f"Mismatch while evaluating {expr} with {literal_args}"
            debug(" OK ret= ", literal_ret, "vs", symbolic_ret)
            debug(
                " OK exc= ",
                type(literal_exc),
                literal_exc,
                "vs",
                type(symbolic_exc),
                symbolic_exc,
            )
        return TrialStatus.NORMAL

    def fuzz_function(self, module: ModuleType, method_name: str):
        method = getattr(module, method_name)
        sig = get_signature(method)
        if not sig:
            return
        arg_names = [chr(ord("a") + i) for i in range(len(sig.parameters))]
        arg_expr_strings = [
            (a if p.kind != Parameter.KEYWORD_ONLY else f"{p.name}={a}")
            for a, p in zip(arg_names, list(sig.parameters.values())[1:])
        ]
        expr_str = method_name + "(" + ",".join(arg_expr_strings) + ")"
        arg_type_roots = {name: object for name in arg_names}
        self.run_trial(expr_str, arg_type_roots)

    def fuzz_method(self, cls: type, method_name: str):
        method = getattr(cls, method_name)
        sig = get_signature(method)
        if not sig:
            return
        arg_names = [chr(ord("a") + i - 1) for i in range(1, len(sig.parameters))]
        arg_expr_strings = [
            (a if p.kind != Parameter.KEYWORD_ONLY else f"{p.name}={a}")
            for a, p in zip(arg_names, list(sig.parameters.values())[1:])
        ]
        expr_str = "self." + method_name + "(" + ",".join(arg_expr_strings) + ")"
        arg_type_roots: Dict[str, type] = {name: object for name in arg_names}
        arg_type_roots["self"] = cls
        self.run_trial(expr_str, arg_type_roots)


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize(
    "expr_str",
    [
        "iter({})",
        "reversed({})",
        "len({})",
        # "repr({})",  # false positive with unstable set ordering
        "str({})",
        "+{}",
        "-{}",
        "~{}",
        # TODO: we aren't `dir()`-compatable right now.
    ],
    ids=simple_name,
)
def test_unary_ops(expr_str, seed) -> None:
    tester = FuzzTester(seed)
    arg_type_roots = {"a": object}
    tester.run_trial(expr_str, arg_type_roots)


@pytest.mark.parametrize("seed", set(range(25)) - {17})
@pytest.mark.parametrize(
    "expr_str",
    [
        "{} + {}",
        "{} - {}",
        "{} * {}",
        "{} / {}",
        "{} < {}",
        "{} <= {}",
        "{} >= {}",
        "{} > {}",
        "{} == {}",
        "{}[{}]",
        "{}.__delitem__({})",
        "{} in {}",
        "{} & {}",
        "{} | {}",
        "{} ^ {}",
        "{} and {}",
        "{} or {}",
        "{} // {}",
        "{} ** {}",
        "{} % {}",
    ],
    ids=simple_name,
)
def test_binary_ops(expr_str, seed) -> None:
    tester = FuzzTester(seed)
    arg_type_roots = {"a": object, "b": object}
    tester.run_trial(expr_str, arg_type_roots)


IGNORED_BUILTINS = [
    "breakpoint",
    "copyright",
    "credits",
    "dir",
    "exit",
    "help",
    "id",
    "input",
    "license",
    "locals",
    "object",
    "open",
    "property",
    "quit",
    # TODO: debug and un-ignore the following:
    "isinstance",
    "issubclass",
    "float",
]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "module,method_name",
    [
        (builtins, name)
        for (name, _) in getmembers(builtins, isbuiltin)
        if not name.startswith("_")
    ],
    ids=simple_name,
)
def test_builtin_functions(seed, module, method_name) -> None:
    if method_name not in IGNORED_BUILTINS:
        FuzzTester(seed).fuzz_function(module, method_name)
    # fns = [
    #     (name, fn)
    #     for name, fn in getmembers(builtins)
    #     if (hasattr(fn, "__call__") and not name.startswith("_") and name not in ignore)
    # ]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "cls,method_name",
    # we don't inspect str directly, because many signature() fails on several members:
    get_testable_members([AbcString]),
    ids=simple_name,
)
def test_str_methods(seed, cls, method_name) -> None:
    FuzzTester(seed).fuzz_method(str, method_name)
    # # we don't inspect str directly, because many signature() fails on several members:
    # # TODO test maketrans()


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "cls,method_name",
    get_testable_members([list, dict]),
    ids=simple_name,
)
def test_container_methods(seed, cls, method_name) -> None:
    FuzzTester(seed).fuzz_method(cls, method_name)


# TODO: deal with iteration order (and then increase repeat count)
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "cls,method_name",
    get_testable_members([set, frozenset]),
    ids=simple_name,
)
def test_set_methods(seed, cls, method_name) -> None:
    FuzzTester(seed).fuzz_method(cls, method_name)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize(
    "cls,method_name",
    get_testable_members([int, float]),
    ids=simple_name,
)
def test_numeric_methods(seed, cls, method_name) -> None:
    FuzzTester(seed).fuzz_method(cls, method_name)
    # TODO test int properties: real, imag, numerator, denominator
    # TODO test int.conjugate()
    # TODO test float properties: real, imag
    # TODO test float.conjugate()
