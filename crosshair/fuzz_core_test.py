import builtins
import copy
import enum
import random
import time
import traceback
from collections.abc import Hashable, Iterable, Mapping, Sized
from inspect import Parameter, Signature, getmembers, isfunction, ismethoddescriptor
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
)

import crosshair.core_and_libs  # ensure patches/plugins are loaded
from crosshair.abcstring import AbcString
from crosshair.core import Patched, deep_realize, proxy_for_type, type_args_of
from crosshair.fnutil import resolve_signature
from crosshair.libimpl.builtinslib import origin_of
from crosshair.statespace import (
    CallAnalysis,
    CrosshairInternal,
    IgnoreAttempt,
    RootNode,
    StateSpace,
    StateSpaceContext,
)
from crosshair.tracers import COMPOSITE_TRACER
from crosshair.util import CrosshairUnsupported, debug, set_debug

FUZZ_SEED = 1348

T = TypeVar("T")


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


class TrialStatus(enum.Enum):
    NORMAL = 0
    UNSUPPORTED = 1


class FuzzTester:
    r: random.Random

    def __init__(self, *a):
        self.r = random.Random(FUZZ_SEED)

    def gen_unary_op(self) -> Tuple[str, Type]:
        return self.r.choice(
            [
                ("iter({})", object),
                ("reversed({})", object),
                ("len({})", object),
                # ("repr({})", object),  # false positive with unstable set ordering
                ("str({})", object),
                ("+{}", object),
                ("-{}", object),
                ("~{}", object),
                # TODO: we aren't `dir()`-compatable right now.
            ]
        )

    def gen_binary_op(self) -> Tuple[str, Type, Type]:
        """
        post: _[0].format('a', 'b')
        """
        return self.r.choice(
            [
                ("{} + {}", object, object),
                ("{} - {}", object, object),
                ("{} * {}", object, object),
                ("{} / {}", object, object),
                ("{} < {}", object, object),
                ("{} <= {}", object, object),
                ("{} >= {}", object, object),
                ("{} > {}", object, object),
                ("{} == {}", object, object),
                ("{}[{}]", object, object),
                ("{}.__delitem__({})", object, object),
                ("{} in {}", object, object),
                ("{} & {}", object, object),
                ("{} | {}", object, object),
                ("{} ^ {}", object, object),
                ("{} and {}", object, object),
                ("{} or {}", object, object),
                ("{} // {}", object, object),
                ("{} ** {}", object, object),
                ("{} % {}", object, object),
            ]
        )

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
        with COMPOSITE_TRACER, Patched():
            for itr in range(1, 200):
                debug("iteration", itr)
                space = StateSpace(
                    time.monotonic() + 30.0, 3.0, search_root=search_root
                )
                symbolic_args = {}
                try:
                    with StateSpaceContext(space):
                        symbolic_args = {
                            name: proxy_for_type(typ, name)
                            for name, typ in typed_args.items()
                        }
                        ret = fn(space, symbolic_args)
                        ret = (deep_realize(ret), symbolic_args, None, space)
                        space.detach_path()
                        return ret
                except IgnoreAttempt as e:
                    debug("ignore iteration attempt: ", str(e))
                except BaseException as e:
                    debug(traceback.format_exc())
                    return (None, symbolic_args, e, space)
                top_analysis, space_exhausted = space.bubble_status(CallAnalysis())
                if space_exhausted:
                    return (
                        None,
                        symbolic_args,
                        CrosshairInternal(f"exhausted after {itr} iterations"),
                        space,
                    )
        return (
            None,
            None,
            CrosshairInternal("Unable to find a successful symbolic execution"),
            space,
        )

    def runexpr(self, expr, bindings):
        try:
            return (eval(expr, {}, bindings), None)
        except Exception as e:
            debug(f'eval of "{expr}" produced exception "{e}"')
            return (None, e)

    def get_signature(self, method) -> Optional[Signature]:
        override_sig = _SIGNATURE_OVERRIDES.get(method, None)
        if override_sig:
            return override_sig
        sig = resolve_signature(method)
        if isinstance(sig, Signature):
            return sig
        return None

    def run_function_trials(
        self, fns: Sequence[Tuple[str, Callable]], num_trials: int
    ) -> None:
        for fn_name, fn in fns:
            debug("Checking function", fn_name)
            sig = self.get_signature(fn)
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
                status = self.run_trial(
                    expr_str, arg_type_roots, f"{fn_name} #{trial_num}"
                )

    def run_class_method_trials(
        self,
        cls: Type,
        min_trials: int,
        members: Optional[List[Tuple[str, Callable]]] = None,
    ) -> None:
        debug("Checking class", cls)
        if members is None:
            members = list(getmembers(cls))
        for method_name, method in members:
            if method_name.startswith("__"):
                debug(
                    "Skipping",
                    method_name,
                    " - it is likely covered by unary/binary op tests",
                )
                continue
            if not (isfunction(method) or ismethoddescriptor(method)):
                # TODO: fuzz test class/staticmethods with symbolic args
                debug(
                    "Skipping",
                    method_name,
                    " - we do not expect class/static methods to be called on SMT types",
                )
                continue
            sig = self.get_signature(method)
            if not sig:
                debug("Skipping", method_name, " - unable to inspect signature")
                continue
            debug("Checking method", method_name)
            num_trials = min_trials  # TODO: something like this?:  min_trials + round(len(sig.parameters) ** 1.5)
            arg_names = [chr(ord("a") + i - 1) for i in range(1, len(sig.parameters))]
            arg_expr_strings = [
                (a if p.kind != Parameter.KEYWORD_ONLY else f"{p.name}={a}")
                for a, p in zip(arg_names, list(sig.parameters.values())[1:])
            ]
            expr_str = "self." + method_name + "(" + ",".join(arg_expr_strings) + ")"
            arg_type_roots = {name: object for name in arg_names}
            arg_type_roots["self"] = cls
            num_unsupported = 0
            for trial_num in range(num_trials):
                status = self.run_trial(
                    expr_str, arg_type_roots, f"{method_name} #{trial_num}"
                )
                if status is TrialStatus.UNSUPPORTED:
                    num_unsupported += 1
            assert (
                num_unsupported != num_trials
            ), f'{num_unsupported} unsupported cases out of {num_trials} testing the method "{method_name}"'

    def run_trial(
        self, expr_str: str, arg_type_roots: Dict[str, Type], trial_desc: str
    ) -> TrialStatus:
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
                if isinstance(literal, (set, frozenset, dict)):
                    assert isinstance(symbolic, Sized)
                    # We need not only equality, but equal ordering, because some operations
                    # like pop() are order-dependent:
                    if len(literal) != len(symbolic):
                        raise IgnoreAttempt(
                            f'symbolic "{name}" not equal to literal "{name}"'
                        )
                    if isinstance(literal, Mapping):
                        assert isinstance(symbolic, Mapping)
                        literal, symbolic = list(literal.items()), list(
                            symbolic.items()
                        )
                    else:
                        assert isinstance(symbolic, Iterable)
                        literal, symbolic = list(literal), list(symbolic)
                if literal != symbolic:
                    raise IgnoreAttempt(
                        f'symbolic "{name}" not equal to literal "{name}"'
                    )
            return eval(expr, symbolic_args.copy())

        try:
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
            with Patched(), StateSpaceContext(space), COMPOSITE_TRACER:
                # compare iterators as the values they produce:
                if isinstance(literal_ret, Iterable) and isinstance(
                    symbolic_ret, Iterable
                ):
                    literal_ret = list(literal_ret)
                    symbolic_ret = list(symbolic_ret)
                postexec_symbolic_args = deep_realize(postexec_symbolic_args)
                symbolic_ret = deep_realize(symbolic_ret)
                symbolic_exc = deep_realize(symbolic_exc)
                rets_differ = bool(literal_ret != symbolic_ret)
                postexec_args_differ = bool(
                    postexec_literal_args != postexec_symbolic_args
                )
                if (
                    rets_differ
                    or postexec_args_differ
                    or type(literal_exc) != type(symbolic_exc)
                ):
                    debug(
                        f"  *****  BEGIN FAILURE FOR {expr} WITH {literal_args}  *****  "
                    )
                    debug(f"  *****  Expected: {literal_ret} / {literal_exc}")
                    debug(f"  *****    {postexec_literal_args}")
                    debug(f"  *****  Symbolic result: {symbolic_ret} / {symbolic_exc}")
                    debug(f"  *****    {postexec_symbolic_args}")
                    debug(f"  *****  END FAILURE FOR {expr}  *****  ")
                    assert (literal_ret, literal_exc) == (symbolic_ret, symbolic_exc)
                debug(" OK ret= ", literal_ret, symbolic_ret)
                debug(" OK exc= ", literal_exc, symbolic_exc)
        except AssertionError as e:
            raise AssertionError(
                f"Trial {trial_desc}: evaluating {expr} with {literal_args}: {e}"
            )
        return TrialStatus.NORMAL


def test_unary_ops() -> None:
    NUM_TRIALS = 100  # raise this as we make fixes
    tester = FuzzTester()
    for i in range(NUM_TRIALS):
        expr_str, type_root = tester.gen_unary_op()
        arg_type_roots = {"a": type_root}
        tester.run_trial(expr_str, arg_type_roots, str(i))


def test_binary_ops() -> None:
    NUM_TRIALS = 300  # raise this as we make fixes
    tester = FuzzTester()
    for i in range(NUM_TRIALS):
        expr_str, type_root1, type_root2 = tester.gen_binary_op()
        arg_type_roots = {"a": type_root1, "b": type_root2}
        tester.run_trial(expr_str, arg_type_roots, str(i))


def test_builtin_functions() -> None:
    ignore = [
        "copyright",
        "credits",
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
    fns = [
        (name, fn)
        for name, fn in getmembers(builtins)
        if (hasattr(fn, "__call__") and not name.startswith("_") and name not in ignore)
    ]
    FuzzTester().run_function_trials(fns, 1)


def test_str_methods() -> None:
    # we don't inspect str directly, because many signature() fails on several members:
    str_members = list(getmembers(AbcString))
    FuzzTester().run_class_method_trials(str, 4, str_members)
    # TODO test maketrans()


def test_list_methods() -> None:
    FuzzTester().run_class_method_trials(list, 5)


def test_dict_methods() -> None:
    FuzzTester().run_class_method_trials(dict, 4)


def test_int_methods() -> None:
    FuzzTester().run_class_method_trials(int, 10)
    # TODO test properties: real, imag, numerator, denominator
    # TODO test conjugate()


def test_float_methods() -> None:
    FuzzTester().run_class_method_trials(float, 10)
    # TODO test properties: real, imag
    # TODO test conjugate()


def test_bytes_methods() -> None:
    FuzzTester().run_class_method_trials(bytes, 2)
