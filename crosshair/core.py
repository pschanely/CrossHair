# *** Not prioritized for v0 ***
# TODO: increase test coverage: TypeVar('T', int, str) vs bounded type vars
# TODO: do not claim "unable to meet preconditions" when we have path timeouts
# TODO: consider raises conditions (guaranteed to raise, guaranteed to not raise?)
# TODO: precondition strengthening ban (Subclass constraint rule)
# TODO: double-check counterexamples
# TODO: contracts for builtins
# TODO: standard library contracts
# TODO: mutating symbolic Callables?
# TODO: contracts on the contracts of function and object inputs/outputs?
# TODO: conditions on Callable arguments/return values

from dataclasses import dataclass, replace
from typing import *
import ast
import builtins
import collections
import copy
import enum
import inspect
from inspect import BoundArguments
from inspect import Signature
import io
import itertools
import functools
import linecache
import os.path
import sys
import time
import traceback
import types
import typing

import forbiddenfruit  # type: ignore
import typing_inspect  # type: ignore
import z3  # type: ignore

from crosshair import dynamic_typing

from crosshair.condition_parser import fn_globals
from crosshair.condition_parser import AssertsParser
from crosshair.condition_parser import CompositeConditionParser
from crosshair.condition_parser import ConditionParser
from crosshair.condition_parser import Pep316Parser
from crosshair.condition_parser import IcontractParser
from crosshair.condition_parser import resolve_signature
from crosshair.condition_parser import Conditions
from crosshair.condition_parser import ConditionExpr
from crosshair.enforce import EnforcedConditions
from crosshair.enforce import PostconditionFailed
from crosshair.statespace import context_statespace
from crosshair.statespace import optional_context_statespace
from crosshair.statespace import prefer_true
from crosshair.statespace import AnalysisMessage
from crosshair.statespace import CallAnalysis
from crosshair.statespace import HeapRef
from crosshair.statespace import MessageType
from crosshair.statespace import SinglePathNode
from crosshair.statespace import StateSpace
from crosshair.statespace import StateSpaceContext
from crosshair.statespace import TrackingStateSpace
from crosshair.statespace import VerificationStatus
from crosshair.fnutil import walk_qualname
from crosshair.fnutil import FunctionInfo
from crosshair.type_repo import get_subclass_map
from crosshair.util import debug
from crosshair.util import eval_friendly_repr
from crosshair.util import extract_module_from_file
from crosshair.util import frame_summary_for_fn
from crosshair.util import is_pure_python
from crosshair.util import name_of_type
from crosshair.util import samefile
from crosshair.util import AttributeHolder
from crosshair.util import CrosshairInternal
from crosshair.util import CrosshairUnsupported
from crosshair.util import DynamicScopeVar
from crosshair.util import IdentityWrapper
from crosshair.util import IgnoreAttempt
from crosshair.util import UnexploredPath


_MISSING = object()

_PATCH_REGISTRATIONS: Dict[
    IdentityWrapper, Dict[str, Callable]
] = collections.defaultdict(dict)

# TODO Unify common logic here with EnforcedConditions?
class Patched:
    def __init__(self, enabled: Callable[[], bool], patches=_PATCH_REGISTRATIONS):
        self._patches = patches
        self._enabled = enabled
        self._originals: Dict[
            IdentityWrapper, Dict[str, object]
        ] = collections.defaultdict(dict)

    def set(self, target: object, key: str, value: object):
        if is_pure_python(target):
            target.__dict__[key] = value
        else:
            forbiddenfruit.curse(target, key, value)

    def patch(self, target: object, key: str, patched_fn: Callable):
        enabled = self._enabled
        orig_fn = target.__dict__.get(key, None)
        if orig_fn is None:
            self.set(target, key, patched_fn)
        else:

            def call_if_enabled(*a, **kw):
                if enabled():
                    return patched_fn(*a, **kw)
                else:
                    return orig_fn(*a, **kw)

            functools.update_wrapper(call_if_enabled, orig_fn)
            self.set(target, key, call_if_enabled)

    def __enter__(self) -> None:
        for target_wrapper, members in self._patches.items():
            container_originals = self._originals[target_wrapper]
            container = target_wrapper.get()
            for key, val in members.items():
                container_originals[key] = getattr(container, key, _MISSING)
                self.patch(container, key, val)

    def __exit__(self, exc_type, exc_value, tb) -> None:
        for target_wrapper, members in self._patches.items():
            container = target_wrapper.get()
            originals = self._originals[target_wrapper]
            for key, orig_val in originals.items():
                if orig_val is _MISSING:
                    del container.__dict__[key]
                else:
                    self.set(container, key, orig_val)
        return False


class ExceptionFilter:
    analysis: CallAnalysis
    ignore: bool = False
    ignore_with_confirmation: bool = False
    user_exc: Optional[Tuple[Exception, traceback.StackSummary]] = None
    expected_exceptions: Tuple[Type[BaseException], ...]

    def __init__(
        self, expected_exceptions: FrozenSet[Type[BaseException]] = frozenset()
    ):
        self.expected_exceptions = (NotImplementedError,) + tuple(expected_exceptions)

    def has_user_exception(self) -> bool:
        return self.user_exc is not None

    def __enter__(self) -> "ExceptionFilter":
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if isinstance(exc_value, (PostconditionFailed, IgnoreAttempt)):
            if isinstance(exc_value, PostconditionFailed):
                # Postcondition : although this indicates a problem, it's with a
                # subroutine; not this function.
                # Usualy we want to ignore this because it will be surfaced more locally
                # in the subroutine.
                debug(f"Ignoring based on internal failed post condition: {exc_value}")
            self.ignore = True
            self.analysis = CallAnalysis()
            return True
        if isinstance(exc_value, self.expected_exceptions):
            debug(f"Hit expected exception: {type(exc_value).__name__}: {exc_value}")
            self.ignore = True
            self.analysis = CallAnalysis(VerificationStatus.CONFIRMED)
            return True
        if isinstance(exc_value, TypeError):
            exc_str = str(exc_value)
            if (
                "SmtStr" in exc_str
                or "SmtInt" in exc_str
                or "SmtFloat" in exc_str
                or "__hash__ method should return an integer" in exc_str
                or "expected string or bytes-like object" in exc_str
            ):
                # Ideally we'd attempt literal strings after encountering this.
                # See https://github.com/pschanely/CrossHair/issues/8
                debug("Proxy intolerace at: ", traceback.format_exc())
                raise CrosshairUnsupported("Detected proxy intolerance: " + exc_str)
        if isinstance(exc_value, (UnexploredPath, CrosshairInternal, z3.Z3Exception)):
            return False  # internal issue: re-raise
        if isinstance(
            exc_value, BaseException
        ):  # TODO: should this be "Exception" instead?
            # Most other issues are assumed to be user-level exceptions:
            self.user_exc = (exc_value, traceback.extract_tb(sys.exc_info()[2]))
            self.analysis = CallAnalysis(VerificationStatus.REFUTED)
            return True  # suppress user-level exception
        return False  # re-raise resource and system issues


class CrossHairValue:
    def __ch_realize__(self):
        raise NotImplementedError(
            f"__ch_realize__ not implemented on {name_of_type(type(self))}"
        )


def normalize_pytype(typ: Type) -> Type:
    if typing_inspect.is_typevar(typ):
        # we treat type vars in the most general way possible (the bound, or as 'object')
        bound = typing_inspect.get_bound(typ)
        if bound is not None:
            return normalize_pytype(bound)
        constraints = typing_inspect.get_constraints(typ)
        if constraints:
            raise CrosshairUnsupported
            # TODO: not easy; interpreting as a Union allows the type to be
            # instantiated differently in different places. So, this doesn't work:
            # return Union.__getitem__(tuple(map(normalize_pytype, constraints)))
        return object
    if typ is Any:
        # The distinction between any and object is for type checking, crosshair treats them the same
        return object
    if typ is Type:
        return type
    return typ


def origin_of(typ: Type) -> Type:
    if hasattr(typ, "__origin__"):
        return typ.__origin__
    return typ


def type_arg_of(typ: Type, index: int) -> Type:
    args = type_args_of(typ)
    return args[index] if index < len(args) else object


def type_args_of(typ: Type) -> Tuple[Type, ...]:
    if getattr(typ, "__args__", None):
        return typing_inspect.get_args(typ, evaluate=True)
    else:
        return ()


def python_type(o: object) -> Type:
    if hasattr(o, "__ch_pytype__"):
        return o.__ch_pytype__()  # type: ignore
    else:
        return type(o)


def realize(value: object):
    if isinstance(value, CrossHairValue):
        with context_statespace().framework():
            return value.__ch_realize__()
    else:
        return value


def with_realized_args(fn: Callable) -> Callable:
    def realizer(*a, **kw):
        a = map(realize, a)
        kw = {k: realize(v) for (k, v) in kw.items()}
        return fn(*a, **kw)

    functools.update_wrapper(realizer, fn)
    return realizer


_IMMUTABLE_TYPES = (int, float, complex, bool, tuple, frozenset, type(None))


class SmtProxyMarker(CrossHairValue):
    def __ch_pytype__(self):
        bases = type(self).__bases__
        assert len(bases) == 2 and bases[0] is SmtProxyMarker
        return bases[1]


_SMT_PROXY_TYPES: Dict[type, Optional[type]] = {}


def get_smt_proxy_type(cls: type) -> Optional[type]:
    if issubclass(cls, SmtProxyMarker):
        return cls
    global _SMT_PROXY_TYPES
    cls_name = name_of_type(cls)
    if cls not in _SMT_PROXY_TYPES:

        def symbolic_init(self):
            try:
                object.__setattr__(self, "__class__", cls)
            except TypeError:
                # NOTE: this is likely "object layout differs from ..."
                # and `cls` is likely implemented in C or has __slots__.
                # We can just continue with our fingers crossed, though:
                pass

        proxy_name = cls_name + "_proxy"
        proxy_super = (SmtProxyMarker, cls)
        proxy_body = {"__init__": symbolic_init}
        proxy_cls: Optional[type] = None
        try:
            proxy_cls = type(proxy_name, proxy_super, proxy_body)
        except Exception as e:
            debug(traceback.format_exc())
        _SMT_PROXY_TYPES[cls] = proxy_cls
    return _SMT_PROXY_TYPES[cls]


def proxy_class_as_masquerade(cls: type, varname: str) -> object:
    constructor = get_smt_proxy_type(cls)
    if constructor is None:
        raise CrosshairUnsupported(
            f"Unable to create a type that masquerades as {name_of_type(cls)}"
        )
    try:
        proxy = constructor()
    except TypeError as e:
        # likely the type has a __new__ that expects arguments
        raise CrosshairUnsupported(f"Unable to proxy {name_of_type(cls)}: {e}")
    for name, typ in get_type_hints(cls).items():
        origin = getattr(typ, "__origin__", None)
        if origin is Callable:
            continue
        if sys.version_info >= (3, 8) and origin_of(typ) is Final:
            value = getattr(cls, name)
        else:
            value = proxy_for_type(
                typ, varname + "." + name + context_statespace().uniq()
            )
        object.__setattr__(proxy, name, value)
    return proxy


def choose_type(space: StateSpace, from_type: Type) -> Type:
    subtypes = get_subclass_map()[from_type]
    # Note that this is written strangely to leverage the default
    # preference for false when forking:
    if not subtypes or not space.smt_fork(desc="choose_" + name_of_type(from_type)):
        return from_type
    for subtype in subtypes[:-1]:
        if not space.smt_fork(desc="choose_" + name_of_type(subtype)):
            return choose_type(space, subtype)
    return choose_type(space, subtypes[-1])


def get_constructor_params(cls: Type) -> Optional[Iterable[inspect.Parameter]]:
    if hasattr(cls, "__signature__"):
        sig = resolve_signature(cls)
        if isinstance(sig, inspect.Signature):
            return list(sig.parameters.values())
    new_fn = cls.__new__
    init_fn = cls.__init__
    if (
        new_fn is not object.__new__
        and
        # Some superclasses like Generic[T] define __new__ with typless (*a,**kw)
        # args. Skip if we don't have types on __new__.
        # TODO: merge the type signatures of __init__ and __new__, pulling the
        # most specific types from each.
        len(get_type_hints(new_fn)) > 0
    ):
        sig = resolve_signature(new_fn)
    elif init_fn is not object.__init__:
        sig = resolve_signature(init_fn)
    else:
        return ()
    if isinstance(sig, inspect.Signature):
        return list(sig.parameters.values())[1:]
    return None


def proxy_class_as_concrete(typ: Type, varname: str) -> object:
    """Try aggressively to create an instance of a class with symbolic members."""
    data_members = get_type_hints(typ)

    # Special handling for some magical types:
    if issubclass(typ, tuple):
        args = {
            k: proxy_for_type(t, varname + "." + k) for (k, t) in data_members.items()
        }
        return typ(**args)  # type: ignore
    elif sys.version_info >= (3, 8) and type(typ) is typing._TypedDictMeta:  # type: ignore
        optional_keys = getattr(typ, "__optional_keys__", ())
        keys = (
            k
            for k in data_members.keys()
            if k not in optional_keys or context_statespace().smt_fork()
        )
        return {k: proxy_for_type(data_members[k], varname + "." + k) for k in keys}
    constructor_params = get_constructor_params(typ)
    if constructor_params is None:
        debug(f"unable to create concrete instance of {typ} due to bad constructor")
        return _MISSING
    EMPTY = inspect.Parameter.empty
    args = {}
    for param in constructor_params:
        name = param.name
        smtname = varname + "." + name
        annotation = param.annotation
        if annotation is not EMPTY:
            args[name] = proxy_for_type(annotation, smtname)
        else:
            if param.default is EMPTY:
                debug(
                    "unable to create concrete instance of",
                    typ,
                    "due to lack of type annotation on",
                    name,
                )
                return _MISSING
            else:
                # TODO: consider whether we should fall back to a proxy
                # instead of letting this slide. Or try both paths?
                pass
    try:
        with context_statespace().unframework():  # TODO: more testing for unframework
            obj = typ(**args)
    except BaseException as e:
        debug(
            f"unable to create concrete instance of {name_of_type(typ)} with init: {name_of_type(type(e))}: {e}"
        )
        return _MISSING

    # Additionally, for any typed members, ensure that they are also
    # symbolic. (classes sometimes have valid states that are not directly
    # constructable)
    for (key, typ) in data_members.items():
        if sys.version_info >= (3, 8) and origin_of(typ) is Final:
            continue
        if isinstance(getattr(obj, key, None), CrossHairValue):
            continue
        symbolic_value = proxy_for_type(typ, varname + "." + key)
        try:
            setattr(obj, key, symbolic_value)
        except Exception as e:
            debug("Unable to assign symbolic value to concrete class:", e)
            # TODO: consider whether we should fall back to a proxy
            # instead of letting this slide. Or try both paths?
    return obj


def proxy_for_class(typ: Type, varname: str, meet_class_invariants: bool) -> object:
    # if the class has data members, we attempt to create a concrete instance with
    # symbolic members; otherwise, we'll create an object proxy that emulates it.
    obj = proxy_class_as_concrete(typ, varname)
    if obj is _MISSING:
        debug("Proxy as a masquerade of", name_of_type(typ))
        obj = proxy_class_as_masquerade(typ, varname)
    else:
        debug("Proxy as a concrete instance of", name_of_type(typ))
    class_conditions = _CALLTREE_PARSER.get().get_class_conditions(typ)
    # symbolic custom classes may assume their invariants:
    if meet_class_invariants and class_conditions is not None:
        for inv_condition in class_conditions.inv:
            if inv_condition.evaluate is None:
                continue
            isok = False
            with ExceptionFilter() as efilter, context_statespace().unframework():
                isok = realize(inv_condition.evaluate({"self": obj}))
            if efilter.user_exc:
                raise IgnoreAttempt(
                    f'Class proxy could not meet invariant "{inv_condition.expr_source}" on '
                    f"{varname} (proxy of {typ}) because it raised: {repr(efilter.user_exc[0])}"
                )
            else:
                if efilter.ignore or not isok:
                    raise IgnoreAttempt(
                        "Class proxy did not meet invariant ", inv_condition.expr_source
                    )
    return obj


def register_patch(
    entity: object, patch_value: Callable, attr_name: Optional[str] = None
):
    if attr_name in _PATCH_REGISTRATIONS[IdentityWrapper(entity)]:
        raise CrosshairInternal(f"Doubly registered patch: {object} . {attr_name}")
    if attr_name is None:
        attr_name = getattr(patch_value, "__name__", None)
        assert attr_name is not None
    _PATCH_REGISTRATIONS[IdentityWrapper(entity)][attr_name] = patch_value


def builtin_patches():
    return _PATCH_REGISTRATIONS[IdentityWrapper(builtins)]


_SIMPLE_PROXIES: MutableMapping[object, Callable] = {}


def register_type(typ: Type, creator: Union[Type, Callable]) -> None:
    assert typ is origin_of(
        typ
    ), f'Only origin types may be registered, not "{typ}": try "{origin_of(typ)}" instead.'
    if typ in _SIMPLE_PROXIES:
        raise CrosshairInternal(f'Duplicate type "{typ}" registered')
    _SIMPLE_PROXIES[typ] = creator


def proxy_for_type(
    typ: Type, varname: str, meet_class_invariants=True, allow_subtypes=False
) -> object:
    space = context_statespace()
    with space.framework():
        typ = normalize_pytype(typ)
        origin = origin_of(typ)
        type_args = type_args_of(typ)
        # special cases
        if isinstance(typ, type) and issubclass(typ, enum.Enum):
            enum_values = list(typ)  # type:ignore
            if not enum_values:
                raise IgnoreAttempt("No values for enum")
            for enum_value in enum_values[:-1]:
                if space.smt_fork(desc="choose_enum_" + str(enum_value)):
                    return enum_value
            return enum_values[-1]
        # It's easy to forget to import crosshair.core_and_libs; check:
        assert _SIMPLE_PROXIES, "No proxy type registrations exist"
        proxy_factory = _SIMPLE_PROXIES.get(origin)
        if proxy_factory:
            # TODO: make this a class with __call__
            def recursive_proxy_factory(t: Type):
                return proxy_for_type(
                    t, varname + space.uniq(), allow_subtypes=allow_subtypes
                )

            recursive_proxy_factory.space = space  # type: ignore
            recursive_proxy_factory.pytype = typ  # type: ignore
            recursive_proxy_factory.varname = varname  # type: ignore
            return proxy_factory(recursive_proxy_factory, *type_args)
        if allow_subtypes and typ is not object:
            typ = choose_type(space, typ)
        return proxy_for_class(typ, varname, meet_class_invariants)


def gen_args(sig: inspect.Signature) -> inspect.BoundArguments:
    args = sig.bind_partial()
    space = context_statespace()
    for param in sig.parameters.values():
        smt_name = param.name + space.uniq()
        proxy_maker = lambda typ, **kw: proxy_for_type(
            typ, smt_name, allow_subtypes=True, **kw
        )
        has_annotation = param.annotation != inspect.Parameter.empty
        value: object
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            if has_annotation:
                varargs_type = List[param.annotation]  # type: ignore
                value = proxy_maker(varargs_type)
            else:
                value = proxy_maker(List[Any])
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            if has_annotation:
                varargs_type = Dict[str, param.annotation]  # type: ignore
                value = cast(dict, proxy_maker(varargs_type))
                # Using ** on a dict requires concrete string keys. Force
                # instiantiation of keys here:
                value = {k.__str__(): v for (k, v) in value.items()}
            else:
                value = proxy_maker(Dict[str, Any])
        else:
            is_self = param.name == "self"
            # Object parameters should meet thier invariants iff they are not the
            # class under test ("self").
            meet_class_invariants = not is_self
            allow_subtypes = not is_self
            if has_annotation:
                value = proxy_for_type(
                    param.annotation, smt_name, meet_class_invariants, allow_subtypes
                )
            else:
                value = proxy_for_type(
                    cast(type, Any), smt_name, meet_class_invariants, allow_subtypes
                )
        debug("created proxy for", param.name, "as type:", name_of_type(type(value)))
        args.arguments[param.name] = value
    return args


_UNABLE_TO_REPR = "<unable to repr>"


def message_sort_key(m: AnalysisMessage) -> tuple:
    return (m.state, _UNABLE_TO_REPR not in m.message, -len(m.message))


class MessageCollector:
    def __init__(self):
        self.by_pos = {}

    def extend(self, messages: Iterable[AnalysisMessage]) -> None:
        for message in messages:
            self.append(message)

    def append(self, message: AnalysisMessage) -> None:
        key = (message.filename, message.line, message.column)
        if key in self.by_pos:
            self.by_pos[key] = max(self.by_pos[key], message, key=message_sort_key)
        else:
            self.by_pos[key] = message

    def get(self) -> List[AnalysisMessage]:
        return [m for (k, m) in sorted(self.by_pos.items())]


class AnalysisKind(enum.Enum):
    PEP316 = "PEP316"
    icontract = "icontract"
    asserts = "asserts"
    # hypothesis = "hypothesis"
    def __str__(self):
        return self.value


@dataclass
class AnalysisOptions:
    per_condition_timeout: float = 1.5
    per_path_timeout: float = 0.75
    max_iterations: int = (
        sys.maxsize
    )  # TODO: use during check and expose on command line
    report_all: bool = False
    analysis_kind: Sequence[AnalysisKind] = (AnalysisKind.PEP316,)

    # Lazily-initialized values
    _condition_parser: Optional[ConditionParser] = None

    # Transient members (not user-configurable):
    deadline: float = float("NaN")
    stats: Optional[collections.Counter] = None

    # Helpers
    def condition_parser(self) -> ConditionParser:
        if self._condition_parser is None:
            debug("Using parsers: ", self.analysis_kind)
            _PARSER_MAP = {
                AnalysisKind.PEP316: Pep316Parser,
                AnalysisKind.icontract: IcontractParser,
                AnalysisKind.asserts: AssertsParser,
            }
            self._condition_parser = CompositeConditionParser()
            self._condition_parser.parsers.extend(
                _PARSER_MAP[k](self._condition_parser) for k in self.analysis_kind
            )
        assert self._condition_parser is not None
        return self._condition_parser

    def split_limits(
        self, priority: float
    ) -> Tuple["AnalysisOptions", "AnalysisOptions"]:
        """
        Divide resource allotments into two.

        Namely, the resource allotments (timeouts, iteration caps) are split
        into allotments for two stages of analysis.

        pre: 0.0 <= priority <= 1.0
        post: _[0].max_iterations + _[1].max_iterations == self.max_iterations
        """
        options1 = replace(
            self,
            per_condition_timeout=self.per_condition_timeout * priority,
            per_path_timeout=self.per_path_timeout * priority,
            max_iterations=round(self.max_iterations * priority),
        )
        inv_priority = 1.0 - priority
        options2 = replace(
            self,
            per_condition_timeout=self.per_condition_timeout * inv_priority,
            per_path_timeout=self.per_path_timeout * inv_priority,
            max_iterations=self.max_iterations - options1.max_iterations,
        )
        return (options1, options2)

    def incr(self, key: str):
        if self.stats is not None:
            self.stats[key] += 1


DEFAULT_OPTIONS = AnalysisOptions()


class Checkable:
    def analyze(self) -> Iterable[AnalysisMessage]:
        raise NotImplementedError


@dataclass
class ConditionCheckable(Checkable):
    ctxfn: FunctionInfo
    options: AnalysisOptions
    conditions: Conditions

    def analyze(self) -> Iterable[AnalysisMessage]:
        options = self.options
        conditions = self.conditions
        debug('Analyzing postcondition: "', conditions.post[0].expr_source, '"')
        debug(
            "assuming preconditions: ",
            ",".join([p.expr_source for p in conditions.pre]),
        )
        options.deadline = time.monotonic() + options.per_condition_timeout

        with scoped_parser(options.condition_parser()):
            analysis = analyze_calltree(options, conditions)

        (condition,) = conditions.post
        addl_ctx = (
            " " + condition.addl_context if condition.addl_context else ""
        ) + "."
        if analysis.verification_status is VerificationStatus.UNKNOWN:
            message = "Not confirmed" + addl_ctx
            analysis.messages = [
                AnalysisMessage(
                    MessageType.CANNOT_CONFIRM,
                    message,
                    condition.filename,
                    condition.line,
                    0,
                    "",
                )
            ]
        elif analysis.verification_status is VerificationStatus.CONFIRMED:
            message = "Confirmed over all paths" + addl_ctx
            analysis.messages = [
                AnalysisMessage(
                    MessageType.CONFIRMED,
                    message,
                    condition.filename,
                    condition.line,
                    0,
                    "",
                )
            ]

        return analysis.messages


class ClampedCheckable(Checkable):
    """
    Clamp messages for a class method to appear on the class itself.

    So, even if the method is defined on a superclass, or defined dynamically (via
    decorator etc), we report it on the class definition instead.
    """

    def __init__(self, checkable: Checkable, cls: type):
        self.checkable = checkable
        self.cls_file = inspect.getsourcefile(cls)
        (_lines, self.cls_start_line) = inspect.getsourcelines(cls)

    def analyze(self) -> Iterable[AnalysisMessage]:
        cls_file = self.cls_file
        ret = []
        for message in self.checkable.analyze():
            if not samefile(message.filename, cls_file):
                ret.append(
                    replace(message, filename=cls_file, line=self.cls_start_line)
                )
            else:
                ret.append(message)
        return ret


@dataclass
class SyntaxErrorCheckable(Checkable):
    messages: List[AnalysisMessage]

    def analyze(self) -> Iterable[AnalysisMessage]:
        return self.messages


def run_checkables(checkables: Iterable[Checkable]) -> Iterable[AnalysisMessage]:
    collector = MessageCollector()
    for checkable in checkables:
        collector.extend(checkable.analyze())
    return collector.get()


def analyzable_members(
    module: types.ModuleType,
) -> Iterator[Union[type, FunctionInfo]]:
    module_name = module.__name__
    for name, member in inspect.getmembers(module):
        if not (
            inspect.isclass(member)
            or inspect.isfunction(member)
            or inspect.ismethod(member)
        ):
            continue
        if member.__module__ != module_name:
            continue
        if inspect.isclass(member):
            yield member
        else:
            yield FunctionInfo.from_module(module, name)


def analyze_any(
    entity: Union[types.ModuleType, type, FunctionInfo], options: AnalysisOptions
) -> Iterable[Checkable]:
    if inspect.isclass(entity):
        yield from analyze_class(cast(Type, entity), options)
    elif isinstance(entity, FunctionInfo):
        yield from analyze_function(entity, options)
    elif inspect.ismodule(entity):
        yield from analyze_module(cast(types.ModuleType, entity), options)
    else:
        raise CrosshairInternal("Entity type not analyzable: " + str(type(entity)))


def analyze_module(
    module: types.ModuleType, options: AnalysisOptions
) -> Iterable[Checkable]:
    for member in analyzable_members(module):
        yield from analyze_any(member, options)


def analyze_class(
    cls: type, options: AnalysisOptions = DEFAULT_OPTIONS
) -> Iterable[Checkable]:
    debug("Analyzing class ", cls.__name__)
    class_conditions = options.condition_parser().get_class_conditions(cls)
    for method, conditions in class_conditions.methods.items():
        if conditions.has_any():
            # Note the use of getattr_static to check superclass contracts on functions
            # that the subclass doesn't define.
            ctxfn = FunctionInfo(cls, method, inspect.getattr_static(cls, method))
            for checkable in analyze_function(ctxfn, options=options):
                yield ClampedCheckable(checkable, cls)


def analyze_function(
    ctxfn: Union[FunctionInfo, types.FunctionType],
    options: AnalysisOptions = DEFAULT_OPTIONS,
) -> List[Checkable]:

    if not isinstance(ctxfn, FunctionInfo):
        ctxfn = FunctionInfo.from_fn(ctxfn)
    debug("Analyzing ", ctxfn.name)
    parser = options.condition_parser()
    if not isinstance(ctxfn.context, type):
        conditions = parser.get_fn_conditions(ctxfn)
    else:
        class_conditions = parser.get_class_conditions(ctxfn.context)
        conditions = class_conditions.methods[ctxfn.name]

    if conditions is None:
        debug("Skipping", ctxfn.name, " because it has no conditions")
        return []
    syntax_messages = list(conditions.syntax_messages())
    if syntax_messages:
        messages = [
            AnalysisMessage(
                MessageType.SYNTAX_ERR,
                syntax_message.message,
                syntax_message.filename,
                syntax_message.line_num,
                0,
                "",
            )
            for syntax_message in syntax_messages
        ]
        return [SyntaxErrorCheckable(messages)]
    else:
        return [
            ConditionCheckable(
                ctxfn, options, replace(conditions, post=[post_condition])
            )
            for post_condition in conditions.post
        ]


def scoped_parser(parser: ConditionParser) -> ContextManager:
    return _CALLTREE_PARSER.open(parser)


class ShortCircuitingContext:
    engaged = False

    def __enter__(self):
        assert not self.engaged
        self.engaged = True

    def __exit__(self, exc_type, exc_value, tb):
        assert self.engaged
        self.engaged = False
        return False

    def make_interceptor(self, original: Callable) -> Callable:
        # TODO: calling from_fn is wrong here
        subconditions = _CALLTREE_PARSER.get().get_fn_conditions(
            FunctionInfo.from_fn(original)
        )
        original_name = original.__name__
        if subconditions is None:
            return original
        sig = subconditions.sig

        def wrapper(*a: object, **kw: Dict[str, object]) -> object:
            space = optional_context_statespace()
            if (not self.engaged) or (not space) or space.running_framework_code:
                debug("short circuit: not eligable", original_name)
                return original(*a, **kw)

            with space.framework():
                bound = sig.bind(*a, **kw)
                assert subconditions is not None
                return_type = consider_shortcircuit(original, sig, bound, subconditions)
                if return_type is None:
                    callinto_probability = 1.0
                else:
                    short_stats, callinto_stats = space.stats_lookahead()
                    if callinto_stats.unknown_pct < short_stats.unknown_pct:
                        callinto_probability = 1.0
                    else:
                        callinto_probability = 0.7

                debug("short circuit: call-into probability", callinto_probability)
                do_short_circuit = space.fork_parallel(
                    callinto_probability, desc=f"shortcircuit {original_name}"
                )

                # Statespace can pick either even with 0.0 or 1.0 probability:
                do_short_circuit &= return_type is not None
            if do_short_circuit:
                assert return_type is not None
                try:
                    self.engaged = False
                    debug(
                        "short circuit: Short circuiting over a call to ", original_name
                    )
                    return shortcircuit(original, sig, bound, return_type)
                finally:
                    self.engaged = True
            else:
                debug("short circuit: Not short circuiting", original_name)
                return original(*a, **kw)

        functools.update_wrapper(wrapper, original)
        return wrapper


# Condition parsers may be needed at various places in the stack.
# We configure them through the use of a magic threadlocal value:
_CALLTREE_PARSER = DynamicScopeVar(ConditionParser, "calltree parser")


@dataclass
class CallTreeAnalysis:
    messages: Sequence[AnalysisMessage]
    verification_status: VerificationStatus
    num_confirmed_paths: int = 0


def analyze_calltree(
    options: AnalysisOptions, conditions: Conditions
) -> CallTreeAnalysis:
    fn = conditions.fn
    debug("Begin analyze calltree ", fn.__name__)

    all_messages = MessageCollector()
    search_root = SinglePathNode(True)
    space_exhausted = False
    failing_precondition: Optional[ConditionExpr] = (
        conditions.pre[0] if conditions.pre else None
    )
    failing_precondition_reason: str = ""
    num_confirmed_paths = 0

    _ = get_subclass_map()  # ensure loaded
    short_circuit = ShortCircuitingContext()
    top_analysis: Optional[CallAnalysis] = None
    enforced_conditions = EnforcedConditions(
        options.condition_parser(),
        fn_globals(fn),
        builtin_patches(),
        interceptor=short_circuit.make_interceptor,
    )

    def in_symbolic_mode():
        space = optional_context_statespace()
        return space and not space.running_framework_code

    patched = Patched(in_symbolic_mode)
    with enforced_conditions, patched, enforced_conditions.disabled_enforcement():
        for i in range(1, options.max_iterations + 1):
            start = time.monotonic()
            if start > options.deadline:
                debug("Exceeded condition timeout, stopping")
                break
            options.incr("num_paths")
            debug("Iteration ", i)
            space = TrackingStateSpace(
                execution_deadline=start + options.per_path_timeout,
                model_check_timeout=options.per_path_timeout / 2,
                search_root=search_root,
            )
            try:
                # The real work happens here!:
                with StateSpaceContext(space):
                    call_analysis = attempt_call(
                        conditions, fn, short_circuit, enforced_conditions
                    )
                if failing_precondition is not None:
                    cur_precondition = call_analysis.failing_precondition
                    if cur_precondition is None:
                        if call_analysis.verification_status is not None:
                            # We escaped the all the pre conditions on this try:
                            failing_precondition = None
                    elif (
                        cur_precondition.line == failing_precondition.line
                        and call_analysis.failing_precondition_reason
                    ):
                        failing_precondition_reason = (
                            call_analysis.failing_precondition_reason
                        )
                    elif cur_precondition.line > failing_precondition.line:
                        failing_precondition = cur_precondition
                        failing_precondition_reason = (
                            call_analysis.failing_precondition_reason
                        )

            except UnexploredPath:
                call_analysis = CallAnalysis(VerificationStatus.UNKNOWN)
            except IgnoreAttempt:
                call_analysis = CallAnalysis()
            status = call_analysis.verification_status
            if status == VerificationStatus.CONFIRMED:
                num_confirmed_paths += 1
            top_analysis, space_exhausted = space.bubble_status(call_analysis)
            debug("Path tree stats", search_root.stats())
            overall_status = top_analysis.verification_status if top_analysis else None
            debug(
                "Iter complete. Worst status found so far:",
                overall_status.name if overall_status else "None",
            )
            if space_exhausted or top_analysis == VerificationStatus.REFUTED:
                break
    top_analysis = search_root.child.get_result()
    if top_analysis.messages:
        all_messages.extend(
            replace(
                m, test_fn=fn.__qualname__, condition_src=conditions.post[0].expr_source
            )
            for m in top_analysis.messages
        )
    if top_analysis.verification_status is None:
        top_analysis.verification_status = VerificationStatus.UNKNOWN
    if failing_precondition:
        assert num_confirmed_paths == 0
        addl_ctx = (
            " " + failing_precondition.addl_context
            if failing_precondition.addl_context
            else ""
        )
        message = f"Unable to meet precondition{addl_ctx}"
        if failing_precondition_reason:
            message += f" (possibly because {failing_precondition_reason}?)"
        all_messages.extend(
            [
                AnalysisMessage(
                    MessageType.PRE_UNSAT,
                    message + ".",
                    failing_precondition.filename,
                    failing_precondition.line,
                    0,
                    "",
                )
            ]
        )
        top_analysis = CallAnalysis(VerificationStatus.REFUTED)

    assert top_analysis.verification_status is not None
    debug(
        ("Exhausted" if space_exhausted else "Aborted"),
        "calltree search with",
        top_analysis.verification_status.name,
        "and",
        len(all_messages.get()),
        "messages.",
        "Number of iterations: ",
        i,
    )
    return CallTreeAnalysis(
        messages=all_messages.get(),
        verification_status=top_analysis.verification_status,
        num_confirmed_paths=num_confirmed_paths,
    )


def get_input_description(
    fn_name: str,
    bound_args: inspect.BoundArguments,
    return_val: object = _MISSING,
    addl_context: str = "",
) -> str:
    with eval_friendly_repr():
        call_desc = ""
        if return_val is not _MISSING:
            try:
                repr_str = repr(return_val)
            except Exception as e:
                if isinstance(e, (IgnoreAttempt, UnexploredPath)):
                    raise
                debug(
                    f"Exception attempting to repr function output: ",
                    traceback.format_exc(),
                )
                repr_str = _UNABLE_TO_REPR
            if repr_str != "None":
                call_desc = call_desc + " (which returns " + repr_str + ")"
        messages: List[str] = []
        for argname, argval in list(bound_args.arguments.items()):
            try:
                repr_str = repr(argval)
            except Exception as e:
                if isinstance(e, (IgnoreAttempt, UnexploredPath)):
                    raise
                debug(
                    f'Exception attempting to repr input "{argname}": ',
                    traceback.format_exc(),
                )
                repr_str = _UNABLE_TO_REPR
            messages.append(argname + " = " + repr_str)
        call_desc = fn_name + "(" + ", ".join(messages) + ")" + call_desc

        if addl_context:
            return addl_context + " when calling " + call_desc
        elif messages:
            return "when calling " + call_desc
        else:
            return "for any input"


class UnEqual:
    pass


_UNEQUAL = UnEqual()


def deep_eq(old_val: object, new_val: object, visiting: Set[Tuple[int, int]]) -> bool:
    # TODO: test just about all of this
    if old_val is new_val:
        return True
    if type(old_val) != type(new_val):
        return False
    visit_key = (id(old_val), id(new_val))
    if visit_key in visiting:
        return True
    visiting.add(visit_key)
    try:
        if isinstance(old_val, CrossHairValue):
            return old_val == new_val
        elif hasattr(old_val, "__dict__") and hasattr(new_val, "__dict__"):
            return deep_eq(old_val.__dict__, new_val.__dict__, visiting)
        elif isinstance(old_val, dict):
            assert isinstance(new_val, dict)
            for key in set(itertools.chain(old_val.keys(), *new_val.keys())):
                if (key in old_val) ^ (key in new_val):
                    return False
                if not deep_eq(
                    old_val.get(key, _UNEQUAL), new_val.get(key, _UNEQUAL), visiting
                ):
                    return False
            return True
        elif isinstance(old_val, Iterable):
            assert isinstance(new_val, Sized)
            if isinstance(old_val, Sized):
                if len(old_val) != len(new_val):
                    return False
            assert isinstance(new_val, Iterable)
            return all(
                deep_eq(o, n, visiting)
                for (o, n) in itertools.zip_longest(
                    old_val, new_val, fillvalue=_UNEQUAL
                )
            )
        elif type(old_val) is object:
            # deepclone'd object instances are close enough to equal for our purposes
            return True
        else:
            # hopefully this is just ints, bools, etc
            return old_val == new_val
    finally:
        visiting.remove(visit_key)


class MessageGenerator:
    def __init__(self, fn: Callable):
        self.filename = ""
        if hasattr(fn, "__code__"):
            code_obj = fn.__code__
            self.filename = code_obj.co_filename
            self.start_lineno = code_obj.co_firstlineno
            try:
                (lines, _) = inspect.getsourcelines(fn)
            except (OSError, TypeError):
                lines = []
            self.end_lineno = self.start_lineno + len(lines)

    def make(
        self,
        message_type: MessageType,
        detail: str,
        suggested_filename: Optional[str],
        suggested_lineno: int,
        tb: str,
    ) -> AnalysisMessage:
        if (
            suggested_filename is not None
            and (os.path.abspath(suggested_filename) == os.path.abspath(self.filename))
            and (self.start_lineno <= suggested_lineno <= self.end_lineno)
        ):
            return AnalysisMessage(
                message_type, detail, suggested_filename, suggested_lineno, 0, tb
            )
        else:
            try:
                exprline = linecache.getlines(suggested_filename)[
                    suggested_lineno - 1
                ].strip()
            except IndexError:
                exprline = "<unknown>"
            detail = f'"{exprline}" yields {detail}'
            return AnalysisMessage(
                message_type, detail, self.filename, self.start_lineno, 0, tb
            )


def attempt_call(
    conditions: Conditions,
    fn: Callable,
    short_circuit: ShortCircuitingContext,
    enforced_conditions: EnforcedConditions,
) -> CallAnalysis:
    assert fn is conditions.fn  # TODO: eliminate the explicit `fn` parameter?
    space = context_statespace()
    bound_args = gen_args(conditions.sig)

    msg_gen = MessageGenerator(conditions.src_fn)
    with space.framework():
        # TODO: looks wrong(-ish) to guard this with space.framework().
        # Copy on custom objects may require patched builtins. (datetime.timedelta is one such case)
        original_args = copy.deepcopy(bound_args)
    space.checkpoint()

    lcls: Mapping[str, object] = bound_args.arguments
    # In preconditions, __old__ exists but is just bound to the same args.
    # This lets people write class invariants using `__old__` to, for example,
    # demonstrate immutability.
    lcls = {"__old__": AttributeHolder(lcls), **lcls}
    expected_exceptions = conditions.raises
    for precondition in conditions.pre:
        with ExceptionFilter(expected_exceptions) as efilter:
            with enforced_conditions.enabled_enforcement(), short_circuit:
                precondition_ok = prefer_true(precondition.evaluate(lcls))
            if not precondition_ok:
                debug("Failed to meet precondition", precondition.expr_source)
                return CallAnalysis(failing_precondition=precondition)
        if efilter.ignore:
            debug("Ignored exception in precondition.", efilter.analysis)
            return efilter.analysis
        elif efilter.user_exc is not None:
            (user_exc, tb) = efilter.user_exc
            debug(
                "Exception attempting to meet precondition",
                precondition.expr_source,
                ":",
                user_exc,
                tb.format(),
            )
            return CallAnalysis(
                failing_precondition=precondition,
                failing_precondition_reason=f'it raised "{repr(user_exc)} at {tb.format()[-1]}"',
            )

    with ExceptionFilter(expected_exceptions) as efilter:
        with enforced_conditions.enabled_enforcement(), short_circuit:
            assert not space.running_framework_code
            __return__ = fn(*bound_args.args, **bound_args.kwargs)
        lcls = {
            **bound_args.arguments,
            "__return__": __return__,
            "_": __return__,
            "__old__": AttributeHolder(original_args.arguments),
            fn.__name__: fn,
        }

    if efilter.ignore:
        debug("Ignored exception in function.", efilter.analysis)
        return efilter.analysis
    elif efilter.user_exc is not None:
        space.check_deferred_assumptions()
        (e, tb) = efilter.user_exc
        detail = name_of_type(type(e)) + ": " + str(e)
        frame_filename, frame_lineno = frame_summary_for_fn(conditions.src_fn, tb)
        tb_desc = tb.format()
        detail += " " + get_input_description(fn.__name__, original_args, _MISSING)
        debug("exception while evaluating function body:", detail, tb_desc)
        return CallAnalysis(
            VerificationStatus.REFUTED,
            [
                msg_gen.make(
                    MessageType.EXEC_ERR,
                    detail,
                    frame_filename,
                    frame_lineno,
                    "".join(tb_desc),
                )
            ],
        )

    for argname, argval in bound_args.arguments.items():
        if (
            conditions.mutable_args is not None
            and argname not in conditions.mutable_args
        ):
            old_val, new_val = original_args.arguments[argname], argval
            if not deep_eq(old_val, new_val, set()):
                space.check_deferred_assumptions()
                detail = 'Argument "{}" is not marked as mutable, but changed from {} to {}'.format(
                    argname, old_val, new_val
                )
                debug("Mutablity problem:", detail)
                return CallAnalysis(
                    VerificationStatus.REFUTED,
                    [msg_gen.make(MessageType.POST_ERR, detail, None, 0, "")],
                )

    (post_condition,) = conditions.post
    with ExceptionFilter(expected_exceptions) as efilter:
        # TODO: re-enable post-condition short circuiting. This will require refactoring how
        # enforced conditions and short curcuiting interact, so that post-conditions are
        # selectively run when, and only when, performing a short circuit.
        # with enforced_conditions.enabled_enforcement(), short_circuit:
        isok = bool(post_condition.evaluate(lcls))
    if efilter.ignore:
        debug("Ignored exception in postcondition.", efilter.analysis)
        return efilter.analysis
    elif efilter.user_exc is not None:
        space.check_deferred_assumptions()
        (e, tb) = efilter.user_exc
        detail = (
            repr(e)
            + " "
            + get_input_description(
                fn.__name__, original_args, __return__, post_condition.addl_context
            )
        )
        debug("exception while calling postcondition:", detail)
        failures = [
            msg_gen.make(
                MessageType.POST_ERR,
                detail,
                post_condition.filename,
                post_condition.line,
                "".join(tb.format()),
            )
        ]
        return CallAnalysis(VerificationStatus.REFUTED, failures)
    if isok:
        debug("Postcondition confirmed.")
        return CallAnalysis(VerificationStatus.CONFIRMED)
    else:
        space.check_deferred_assumptions()
        detail = "false " + get_input_description(
            fn.__name__, original_args, __return__, post_condition.addl_context
        )
        debug(detail)
        failures = [
            msg_gen.make(
                MessageType.POST_FAIL,
                detail,
                post_condition.filename,
                post_condition.line,
                "",
            )
        ]
        return CallAnalysis(VerificationStatus.REFUTED, failures)


# Objects of these types are known to always be *deeply* immutable:
_ATOMIC_IMMUTABLE_TYPES = (
    type(None),
    int,
    str,
    float,
    complex,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.LambdaType,
    types.MethodType,
    types.BuiltinMethodType,
)


def is_deeply_immutable(o: object) -> bool:
    orig_hash = builtins.hash

    def _mutability_testing_hash(o: object) -> int:
        if isinstance(o, _ATOMIC_IMMUTABLE_TYPES):
            return 0
        return orig_hash(o)

    with Patched(
        lambda: True, {IdentityWrapper(builtins): {"hash": _mutability_testing_hash}}
    ):
        try:
            hash(o)
            return True
        except TypeError:
            return False


def consider_shortcircuit(
    fn: Callable, sig: Signature, bound: BoundArguments, subconditions: Conditions
) -> Optional[type]:
    """
    Consider the feasibility of short-circuiting (skipping) a function with the given arguments.

    :return: The type of a symbolic value that could be returned by ``fn``.
    :return: None if a short-circuiting should not be attempted.
    """
    return_type = sig.return_annotation

    mutable_args = subconditions.mutable_args
    if mutable_args is None or len(mutable_args) > 0:
        # we don't deal with mutation inside the skipped function yet.
        debug("aborting shortcircuit: function has matuable args")
        return None

    # Deduce type vars if necessary
    if len(typing_inspect.get_parameters(return_type)) > 0 or typing_inspect.is_typevar(
        return_type
    ):

        typevar_bindings: typing.ChainMap[object, type] = collections.ChainMap()
        bound.apply_defaults()
        for param in sig.parameters.values():
            argval = bound.arguments[param.name]
            # We don't need all args to be symbolic, but we don't currently
            # short circuit in that case as a heuristic.
            if not isinstance(argval, CrossHairValue):
                debug("aborting shortcircuit: {param.name} is not symbolic")
                return None
            value_type = python_type(argval)
            if not dynamic_typing.unify(value_type, param.annotation, typevar_bindings):
                debug("aborting shortcircuit: {param.name} fails unification")
                return None
        return_type = dynamic_typing.realize(sig.return_annotation, typevar_bindings)
    return return_type


def shortcircuit(
    fn: Callable, sig: Signature, bound: BoundArguments, return_type: Type
) -> object:
    space = context_statespace()
    debug("short circuit: Deduced return type was ", return_type)

    # Deep copy the arguments for reconciliation later.
    # (we know that this function won't mutate them, but not that others won't)
    argscopy = {}
    for name, val in bound.arguments.items():
        if is_deeply_immutable(val):
            argscopy[name] = val
        else:
            argscopy[name] = copy.deepcopy(val)
    bound_copy = BoundArguments(sig, argscopy)  # type: ignore

    retval = None
    if return_type is not type(None):
        # note that the enforcement wrapper ensures postconditions for us, so
        # we can just return a free variable here.
        retval = proxy_for_type(return_type, "proxyreturn" + space.uniq())

    def reconciled() -> bool:
        return retval == fn(*bound_copy.args, **bound_copy.kwargs)

    space.defer_assumption("Reconcile short circuit", reconciled)

    return retval
