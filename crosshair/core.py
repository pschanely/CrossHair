# TODO: drop to PDB option
# TODO: detect problems with backslashes in docstrings

# *** Not prioritized for v0 ***
# TODO: increase test coverage: TypeVar('T', int, str) vs bounded type vars
# TODO: consider raises conditions (guaranteed to raise, guaranteed to not raise?)
# TODO: precondition strengthening ban (Subclass constraint rule)
# TODO: mutating symbolic Callables?
# TODO: contracts on the contracts of function and object inputs/outputs?

import enum
import functools
import inspect
import itertools
import linecache
import os.path
import sys
import time
import traceback
import types
import typing
from collections import ChainMap, defaultdict, deque
from contextlib import ExitStack
from dataclasses import dataclass, replace
from inspect import BoundArguments, Signature
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
    overload,
)

import typing_inspect  # type: ignore
import z3  # type: ignore

from crosshair import dynamic_typing
from crosshair.codeconfig import collect_options
from crosshair.condition_parser import (
    ConditionExpr,
    ConditionExprType,
    Conditions,
    condition_parser,
    get_current_parser,
)
from crosshair.copyext import CopyMode, deepcopyext
from crosshair.enforce import (
    EnforcedConditions,
    NoEnforce,
    PostconditionFailed,
    PreconditionFailed,
    WithEnforcement,
)
from crosshair.fnutil import FunctionInfo, resolve_signature
from crosshair.options import DEFAULT_OPTIONS, AnalysisOptions, AnalysisOptionSet
from crosshair.register_contract import get_contract
from crosshair.statespace import (
    AnalysisMessage,
    CallAnalysis,
    MessageType,
    NotDeterministic,
    RootNode,
    SimpleStateSpace,
    StateSpace,
    StateSpaceContext,
    VerificationStatus,
    context_statespace,
    optional_context_statespace,
    prefer_true,
)
from crosshair.tracers import (
    COMPOSITE_TRACER,
    CompositeTracer,
    NoTracing,
    PatchingModule,
    ResumedTracing,
    TracingModule,
    TracingOnly,
    is_tracing,
)
from crosshair.type_repo import get_subclass_map
from crosshair.util import (
    ATOMIC_IMMUTABLE_TYPES,
    UNABLE_TO_REPR_TEXT,
    AttributeHolder,
    CrosshairInternal,
    CrosshairUnsupported,
    IdKeyedDict,
    IgnoreAttempt,
    UnexploredPath,
    debug,
    eval_friendly_repr,
    format_boundargs,
    frame_summary_for_fn,
    name_of_type,
    samefile,
    smtlib_typename,
    sourcelines,
    test_stack,
    warn,
)

_MISSING = object()


_OPCODE_PATCHES: List[TracingModule] = []

_PATCH_REGISTRATIONS: Dict[Callable, Callable] = {}
_PATCH_FN_TYPE_REGISTRATIONS: Dict[type, Callable] = {}


class Patched(TracingModule):
    def __enter__(self):
        COMPOSITE_TRACER.push_module(
            PatchingModule(_PATCH_REGISTRATIONS, _PATCH_FN_TYPE_REGISTRATIONS)
        )
        push_count = 1
        if len(_OPCODE_PATCHES) == 0:
            raise CrosshairInternal("Opcode patches haven't been loaded yet.")
        for module in _OPCODE_PATCHES:
            COMPOSITE_TRACER.push_module(module)
            push_count += 1
        self.push_count = push_count
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for _ in range(self.push_count):
            COMPOSITE_TRACER.pop_config()
        return False


class _StandaloneStatespace(ExitStack):
    def __enter__(self):
        # We explicitly don't set up contexts to enforce conditions - that's because
        # conditions involve a choice, and standalone_statespace is for testing that
        # does not require making any choices.
        super().__enter__()
        space = SimpleStateSpace()
        self.enter_context(condition_parser(DEFAULT_OPTIONS.analysis_kind))
        self.enter_context(Patched())
        self.enter_context(StateSpaceContext(space))
        self.enter_context(COMPOSITE_TRACER)
        COMPOSITE_TRACER.trace_caller()
        return space


standalone_statespace = _StandaloneStatespace()


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
        if not is_tracing():
            raise CrosshairInternal("must be tracing during exception filter")
        return self

    def __exit__(self, exc_type, exc_value, tb) -> bool:
        with NoTracing():
            if isinstance(exc_value, (PostconditionFailed, IgnoreAttempt)):
                if isinstance(exc_value, PostconditionFailed):
                    # Postcondition : although this indicates a problem, it's with a
                    # subroutine; not this function.
                    # Usualy we want to ignore this because it will be surfaced more locally
                    # in the subroutine.
                    debug(
                        f"Ignoring based on internal failed post condition: {exc_value}"
                    )
                self.ignore = True
                self.analysis = CallAnalysis()
                return True
            if isinstance(exc_value, self.expected_exceptions):
                exc_type_name = type(exc_value).__name__
                debug(f"Hit expected exception: {exc_type_name}: {exc_value}")
                self.ignore = True
                self.analysis = CallAnalysis(VerificationStatus.CONFIRMED)
                return True
            if isinstance(exc_value, TypeError):
                exc_str = str(exc_value)
                if (
                    "SymbolicStr" in exc_str
                    or "SymbolicInt" in exc_str
                    or "SymbolicFloat" in exc_str
                    or "__hash__ method should return an integer" in exc_str
                    or "expected string or bytes-like object" in exc_str
                ):
                    # Ideally we'd attempt literal strings after encountering this.
                    # See https://github.com/pschanely/CrossHair/issues/8
                    debug("Proxy intolerace at: ", traceback.format_exc())
                    raise CrosshairUnsupported("Detected proxy intolerance: " + exc_str)
            if isinstance(
                exc_value, (UnexploredPath, CrosshairInternal, z3.Z3Exception)
            ):
                return False  # internal issue: re-raise
            if isinstance(exc_value, BaseException):
                # Most other issues are assumed to be user-facing exceptions:
                self.user_exc = (exc_value, traceback.extract_tb(sys.exc_info()[2]))
                self.analysis = CallAnalysis(VerificationStatus.REFUTED)
                return True  # suppress user-level exception
            return False  # re-raise resource and system issues


_T = TypeVar("_T")


def realize(value: Any) -> Any:
    with NoTracing():
        if hasattr(type(value), "__ch_realize__"):
            return value.__ch_realize__()  # type: ignore
        else:
            return value


def deep_realize(value: _T) -> _T:
    with NoTracing():
        return deepcopyext(value, CopyMode.REALIZE, {})


class CrossHairValue:
    pass


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


# TODO: remove this nonsense on the next typing_inspect release:
ExtraUnionType = getattr(types, "UnionType") if sys.version_info >= (3, 10) else None


def origin_of(typ: Type) -> Type:
    if hasattr(typ, "__origin__"):
        return typ.__origin__
    elif ExtraUnionType and isinstance(typ, ExtraUnionType):
        return cast(Type, Union)
    else:
        return typ


def type_arg_of(typ: Type, index: int) -> Type:
    args = type_args_of(typ)
    return args[index] if index < len(args) else object


def type_args_of(typ: Type) -> Tuple[Type, ...]:
    if getattr(typ, "__args__", None):
        if ExtraUnionType and isinstance(typ, ExtraUnionType):
            return typ.__args__
        return typing_inspect.get_args(typ, evaluate=True)
    else:
        return ()


def python_type(o: object) -> Type:
    if is_tracing():
        raise CrosshairInternal("should not be tracing while getting pytype")
    if hasattr(type(o), "__ch_pytype__"):
        obj_type = o.__ch_pytype__()  # type: ignore
        if hasattr(obj_type, "__origin__"):
            obj_type = obj_type.__origin__
        return obj_type
    else:
        return type(o)


def with_realized_args(fn: Callable) -> Callable:
    def realizer(*a, **kw):
        with NoTracing():
            a = map(realize, a)
            kw = {k: realize(v) for (k, v) in kw.items()}
            return fn(*a, **kw)

    functools.update_wrapper(realizer, fn)
    return realizer


def with_symbolic_self(symbolic_cls: Type, fn: Callable):
    def call_with_symbolic_self(self, *args, **kwargs):
        with NoTracing():
            if isinstance(self, symbolic_cls):
                # Handles (unlikely!) cases like str.isspace(<symbolic string>)
                target_fn = getattr(symbolic_cls, fn.__name__)
            elif any(isinstance(a, CrossHairValue) for a in args) or (
                kwargs and any(isinstance(a, CrossHairValue) for a in kwargs.values())
            ):
                self = symbolic_cls._smt_promote_literal(self)
                target_fn = getattr(symbolic_cls, fn.__name__)
            else:
                args = map(realize, args)
                kwargs = {k: realize(v) for (k, v) in kwargs.items()}
                target_fn = fn
        return target_fn(self, *args, **kwargs)

    functools.update_wrapper(call_with_symbolic_self, fn)
    return call_with_symbolic_self


def with_uniform_probabilities(
    collection: Collection[_T],
) -> List[Tuple[_T, float]]:
    count = len(collection)
    return [(item, 1.0 / (count - idx)) for (idx, item) in enumerate(collection)]


def iter_types(from_type: Type) -> List[Tuple[Type, float]]:
    types = []
    queue = deque([from_type])
    subclassmap = get_subclass_map()
    while queue:
        cur = queue.popleft()
        queue.extend(subclassmap[cur])
        types.append(cur)
    ret = with_uniform_probabilities(types)
    # Bias a little extra for the first (base) type;
    # e.g. pick `int` more readily than the subclasses of int:
    first_probability = ret[0][1]
    ret[0] = (from_type, (first_probability + 3.0) / 4.0)
    return ret


def choose_type(space: StateSpace, from_type: Type) -> Type:
    for typ, probability_true in iter_types(from_type):
        # true_probability=1.0 does not guarantee selection
        # (in particular, when the true path is exhausted)
        if probability_true == 1.0:
            return typ
        if space.smt_fork(
            desc="pick_" + smtlib_typename(typ), probability_true=probability_true
        ):
            return typ
    raise CrosshairInternal


def get_constructor_signature(cls: Type) -> Optional[inspect.Signature]:
    # pydantic sets __signature__ on the class, so we look for that as well as on
    # __init__ (see https://github.com/samuelcolvin/pydantic/pull/1034)
    if hasattr(cls, "__signature__"):
        sig = resolve_signature(cls)
        if isinstance(sig, inspect.Signature):
            return sig
    new_fn = cls.__new__
    sig = resolve_signature(new_fn)
    # TODO: merge the type signatures of __init__ and __new__, pulling the
    # most specific types from each.
    # Fall back to __init__ if we don't have types:
    if isinstance(sig, str) or all(
        p.annotation is Signature.empty for p in sig.parameters.values()
    ):
        init_fn = cls.__init__
        if init_fn is not object.__init__:
            sig = resolve_signature(init_fn)
        else:
            return inspect.Signature([])
    if isinstance(sig, inspect.Signature):
        # strip first argument
        newparams = list(sig.parameters.values())[1:]
        return sig.replace(parameters=newparams)
    return None


def proxy_for_class(typ: Type, varname: str) -> object:
    data_members = get_type_hints(typ)

    if sys.version_info >= (3, 8) and type(typ) is typing._TypedDictMeta:  # type: ignore
        # Handling for TypedDict
        optional_keys = getattr(typ, "__optional_keys__", ())
        keys = (
            k
            for k in data_members.keys()
            if k not in optional_keys or context_statespace().smt_fork()
        )
        return {k: proxy_for_type(data_members[k], varname + "." + k) for k in keys}

    constructor_sig = get_constructor_signature(typ)
    if constructor_sig is None:
        raise CrosshairUnsupported(
            f"unable to create concrete instance of {typ} due to bad constructor"
        )
    args = gen_args(constructor_sig)
    typename = name_of_type(typ)
    try:
        with ResumedTracing():
            obj = WithEnforcement(typ)(*args.args, **args.kwargs)
    except (PreconditionFailed, PostconditionFailed):
        # preconditions can be invalidated when the __init__ method has preconditions.
        # postconditions can be invalidated when the class has invariants.
        raise IgnoreAttempt
    except BaseException as e:
        debug("Root-cause type construction traceback:", test_stack(e.__traceback__))
        raise CrosshairUnsupported(
            f"error constructing {typename} instance: {name_of_type(type(e))}: {e}",
        ) from e

    debug("Proxy as a concrete instance of", typename)
    repr_overrides = context_statespace().extra(LazyCreationRepr).reprs

    def regenerate_construction_string(_):
        realized_args = deep_realize(args)
        with NoTracing():
            return f"{typename}({format_boundargs(realized_args)})"

    repr_overrides[obj] = regenerate_construction_string

    debug("repr register lazy", hex(id(obj)), typename)
    return obj


def register_patch(entity: Callable, patch_value: Callable):
    if entity in _PATCH_REGISTRATIONS:
        raise CrosshairInternal(f"Doubly registered patch: {entity}")
    _PATCH_REGISTRATIONS[entity] = patch_value


def register_fn_type_patch(typ: type, patch_value: Callable[[Callable], Callable]):
    if typ in _PATCH_FN_TYPE_REGISTRATIONS:
        raise CrosshairInternal(f"Doubly registered fn type patch: {typ}")
    _PATCH_FN_TYPE_REGISTRATIONS[typ] = patch_value


def register_opcode_patch(module: TracingModule) -> None:
    _OPCODE_PATCHES.append(module)


class SymbolicFactory:
    """
    A callable object that creates symbolic values.

    .. automethod:: __call__
    """

    def __init__(self, space: StateSpace, pytype: object, varname: str):
        self.space = space
        self.pytype: Any = pytype
        self.varname = varname

    @overload
    def __call__(
        self, typ: Callable[..., _T], suffix: str = "", allow_subtypes: bool = True
    ) -> _T:
        ...

    @overload
    def __call__(self, typ: Any, suffix: str = "", allow_subtypes: bool = True) -> Any:
        ...

    def __call__(self, typ, suffix: str = "", allow_subtypes: bool = True):
        """
        Create a new symbolic value.

        :param typ: The corresponding Python type for the returned symbolic.
        :type typ: type
        :param suffix: A descriptive suffix used to name variable(s) in the solver.
        :type suffix: str
        :param allow_subtypes: Whether it's ok to return a subtype of given type.
        :type allow_subtypes: bool
        :returns: A new symbolic value.
        """
        return proxy_for_type(
            typ,
            self.varname + suffix + self.space.uniq(),
            allow_subtypes=allow_subtypes,
        )


_SIMPLE_PROXIES: MutableMapping[object, Callable] = {}

SymbolicCreationCallback = Union[
    # Sadly Callable[] doesn't support variable arguments. Just enumerate:
    Callable[[SymbolicFactory], object],
    Callable[[SymbolicFactory, Type], object],
    Callable[[SymbolicFactory, Type, Type], object],
    Callable[[SymbolicFactory, Type, Type, Type], object],
    Callable[[SymbolicFactory, Type, Type, Type, Type], object],
]


def register_type(typ: Type, creator: SymbolicCreationCallback) -> None:
    """
    Register a custom creation function to create symbolic values for a type.

    :param typ: The Python type (or typing annotation) to handle.
    :param creator: A function that takes a :class:`SymbolicFactory` instance and
      returns a symbolic value. When creating a parameterized type (e.g. List[int]),
      type parameters will be given to `creator` as additional arguments following the
      factory.
    """
    assert typ is origin_of(
        typ
    ), f'Only origin types may be registered, not "{typ}": try "{origin_of(typ)}" instead.'
    if typ in _SIMPLE_PROXIES:
        raise CrosshairInternal(f'Duplicate type "{typ}" registered')
    _SIMPLE_PROXIES[typ] = creator


@dataclass
class LazyCreationRepr:
    def __init__(self, *a):
        self.reprs = IdKeyedDict()


@overload
def proxy_for_type(
    typ: Callable[..., _T],
    varname: str,
    allow_subtypes: bool = True,
) -> _T:
    ...


@overload
def proxy_for_type(
    typ: Any,
    varname: str,
    allow_subtypes: bool = True,
) -> Any:
    ...


def proxy_for_type(
    typ: Any,
    varname: str,
    allow_subtypes: bool = False,
) -> Any:
    space = context_statespace()
    with NoTracing():
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
            recursive_proxy_factory = SymbolicFactory(space, typ, varname)
            return proxy_factory(recursive_proxy_factory, *type_args)
        if allow_subtypes and typ is not object:
            typ = choose_type(space, typ)
        return proxy_for_class(typ, varname)


def gen_args(sig: inspect.Signature) -> inspect.BoundArguments:
    if is_tracing():
        raise CrosshairInternal
    args = sig.bind_partial()
    space = context_statespace()
    for param in sig.parameters.values():
        smt_name = param.name + space.uniq()

        def proxy_maker(typ):
            return proxy_for_type(typ, smt_name, allow_subtypes=True)

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
            # Object parameters can be any valid subtype iff they are not the
            # class under test ("self").
            allow_subtypes = not is_self
            if has_annotation:
                value = proxy_for_type(param.annotation, smt_name, allow_subtypes)
            else:
                value = proxy_for_type(cast(type, Any), smt_name, allow_subtypes)
        debug(
            "created proxy for",
            param.name,
            "as type:",
            name_of_type(type(value)),
            hex(id(value)),
        )
        args.arguments[param.name] = value
    return args


def message_sort_key(m: AnalysisMessage) -> tuple:
    return (m.state, UNABLE_TO_REPR_TEXT not in m.message, -len(m.message))


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

        with condition_parser(options.analysis_kind):
            analysis = analyze_calltree(options, conditions)

        (condition,) = conditions.post
        if analysis.verification_status is VerificationStatus.UNKNOWN:
            message = "Not confirmed."
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
            message = "Confirmed over all paths."
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
        filename, start_line, _ = sourcelines(cls)
        self.cls_file = filename
        self.cls_start_line = start_line

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


def run_checkables(checkables: Iterable[Checkable]) -> List[AnalysisMessage]:
    collector = MessageCollector()
    for checkable in checkables:
        collector.extend(checkable.analyze())
    return collector.get()


def analyze_any(
    entity: Union[types.ModuleType, type, FunctionInfo], options: AnalysisOptionSet
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
    module: types.ModuleType, options: AnalysisOptionSet
) -> Iterable[Checkable]:
    """Analyze the classes and functions defined in a module."""
    module_name = module.__name__
    for name, member in inspect.getmembers(module):
        if not (
            inspect.isclass(member)
            or inspect.isfunction(member)
            or inspect.ismethod(member)
        ):
            continue
        if member.__module__ != module_name:
            # Modules often have contents that are imported from elsewhere
            continue
        if inspect.isclass(member):
            yield from analyze_class(member, options)
        else:
            yield from analyze_function(FunctionInfo.from_module(module, name), options)


def analyze_class(
    cls: type, options: AnalysisOptionSet = AnalysisOptionSet()
) -> Iterable[Checkable]:
    debug("Analyzing class ", cls.__name__)
    analysis_kinds = DEFAULT_OPTIONS.overlay(options).analysis_kind
    with condition_parser(analysis_kinds) as parser:
        class_conditions = parser.get_class_conditions(cls)
        for method_name, conditions in class_conditions.methods.items():
            if method_name == "__init__":
                # Don't check invariants on __init__.
                # (too often this just requires turning the invariant into a very
                # similar precondition)
                filtered_post = [
                    c
                    for c in conditions.post
                    if c.condition_type != ConditionExprType.INVARIANT
                ]
                conditions = replace(conditions, post=filtered_post)
            if conditions.has_any():
                # Note the use of getattr_static to check superclass contracts on
                # functions that the subclass doesn't define.
                ctxfn = FunctionInfo(
                    cls, method_name, inspect.getattr_static(cls, method_name)
                )
                for checkable in analyze_function(ctxfn, options=options):
                    yield ClampedCheckable(checkable, cls)


def analyze_function(
    ctxfn: Union[FunctionInfo, types.FunctionType, Callable],
    options: AnalysisOptionSet = AnalysisOptionSet(),
) -> List[Checkable]:

    if not isinstance(ctxfn, FunctionInfo):
        ctxfn = FunctionInfo.from_fn(ctxfn)
    debug("Analyzing ", ctxfn.name)
    pair = ctxfn.get_callable()
    fn_options = collect_options(pair[0]) if pair else AnalysisOptionSet()
    full_options = DEFAULT_OPTIONS.overlay(fn_options).overlay(options)
    if not full_options.enabled:
        debug("Skipping", ctxfn.name, " because CrossHair is not enabled")
        return []

    with condition_parser(full_options.analysis_kind) as parser:
        if not isinstance(ctxfn.context, type):
            conditions = parser.get_fn_conditions(ctxfn)
        else:
            class_conditions = parser.get_class_conditions(ctxfn.context)
            conditions = class_conditions.methods.get(ctxfn.name)

    if conditions is None:
        debug("Skipping", ctxfn.name, " because it has no conditions")
        return []
    syntax_messages = list(conditions.syntax_messages())
    if syntax_messages:
        debug("Syntax error(s): ", *(m.message for m in syntax_messages))
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
    return [
        ConditionCheckable(
            ctxfn, full_options, replace(conditions, post=[post_condition])
        )
        for post_condition in conditions.post
        if post_condition.evaluate is not None
    ]


def fn_returning(values: list) -> Callable:
    itr = iter(values)

    def patched_call(*a, **kw):
        try:
            return next(itr)
        except StopIteration:
            raise NotDeterministic

    return patched_call


def patch_to_return(return_values: Dict[Callable, list]) -> typing.ContextManager:
    patches = {fn: fn_returning(values) for (fn, values) in return_values.items()}
    tracer = CompositeTracer()
    tracer.push_module(PatchingModule(patches))
    return tracer


class FunctionInterps:
    _interpretations: Dict[Callable, List[object]]

    def __init__(self, *a):
        self._interpretations = defaultdict(list)

    def append_return(self, callable: Callable, retval: object) -> None:
        self._interpretations[callable].append(retval)

    def patch_string(self) -> Optional[str]:
        if self._interpretations:
            patches = ",".join(
                f"{eval_friendly_repr(fn)}: {eval_friendly_repr(deep_realize(vals))}"
                for fn, vals in self._interpretations.items()
            )
            return f"crosshair.patch_to_return({{{patches}}})"
        return None


class ShortCircuitingContext:
    def __init__(self):
        self.engaged = False

        # Note: this cache is not really for performance; it preserves
        # function identity so that contract enforcement can correctly detect
        # re-entrant contracts.
        self.interceptor_cache = {}

    def __enter__(self):
        assert not self.engaged
        self.engaged = True

    def __exit__(self, exc_type, exc_value, tb):
        assert self.engaged
        self.engaged = False
        return False

    def make_interceptor(self, original: Callable) -> Callable:
        interceptor = self.interceptor_cache.get(original)
        if interceptor:
            return interceptor

        # TODO: calling from_fn is wrong here
        subconditions = get_current_parser().get_fn_conditions(
            FunctionInfo.from_fn(original)
        )
        original_name = original.__name__
        if subconditions is None:
            self.interceptor_cache[original] = original
            return original
        sig = subconditions.sig

        def _crosshair_wrapper(*a: object, **kw: Dict[str, object]) -> object:
            space = optional_context_statespace()
            if (not self.engaged) or (not space):
                debug("Not short-circuiting", original_name, "(not engaged)")
                return original(*a, **kw)

            with NoTracing():
                assert subconditions is not None
                # Skip function body if it has the option `specs_complete`.
                short_circuit = collect_options(original).specs_complete
                # Also skip if the function was manually registered to be skipped.
                contract = get_contract(original)
                if contract and contract.skip_body:
                    short_circuit = True
                # TODO: In the future, sig should be a list of sigs and the parser
                # would directly return contract.sigs, so no need to fetch it here.
                sigs = [sig]
                if contract and contract.sigs:
                    sigs = contract.sigs
                best_sig = sigs[0]
                # The function is overloaded, find the best signature.
                if len(sigs) > 1:
                    new_sig = find_best_sig(sigs, *a, *kw)
                    if new_sig:
                        best_sig = new_sig
                    else:
                        # If no signature is valid, we cannot shortcircuit.
                        short_circuit = False
                        warn(
                            "No signature match with the given parameters for function",
                            original_name,
                        )
                bound = best_sig.bind(*a, **kw)
                return_type = consider_shortcircuit(
                    original,
                    best_sig,
                    bound,
                    subconditions,
                    allow_interpretation=not short_circuit,
                )
                if short_circuit:
                    assert return_type is not None
                    retval = proxy_for_type(return_type, "proxyreturn" + space.uniq())
                    space.extra(FunctionInterps).append_return(original, retval)
                    debug("short circuit: specs complete; skipping (as uninterpreted)")
                    return retval
            if return_type is not None:
                try:
                    self.engaged = False
                    debug(
                        "short circuit: Short circuiting over a call to ", original_name
                    )
                    return shortcircuit(original, best_sig, bound, return_type)
                finally:
                    self.engaged = True
            else:
                debug("short circuit: Not short circuiting", original_name)
                return original(*a, **kw)

        functools.update_wrapper(_crosshair_wrapper, original)
        self.interceptor_cache[original] = _crosshair_wrapper
        return _crosshair_wrapper


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
    search_root = RootNode()
    space_exhausted = False
    failing_precondition: Optional[ConditionExpr] = (
        conditions.pre[0] if conditions.pre else None
    )
    failing_precondition_reason: str = ""
    num_confirmed_paths = 0

    short_circuit = ShortCircuitingContext()
    top_analysis: Optional[CallAnalysis] = None
    enforced_conditions = EnforcedConditions(
        interceptor=short_circuit.make_interceptor,
    )
    patched = Patched()
    # TODO clean up how encofrced conditions works here?
    with enforced_conditions, patched:
        for i in range(1, options.max_iterations + 1):
            start = time.monotonic()
            if start > options.deadline:
                debug("Exceeded condition timeout, stopping")
                break
            options.incr("num_paths")
            debug("Iteration ", i)
            per_path_timeout = options.get_per_path_timeout()
            space = StateSpace(
                execution_deadline=start + per_path_timeout,
                model_check_timeout=per_path_timeout / 2,
                search_root=search_root,
            )
            try:
                with StateSpaceContext(space), COMPOSITE_TRACER:
                    # The real work happens here!:
                    call_analysis = attempt_call(
                        conditions, short_circuit, enforced_conditions
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
            if space_exhausted or overall_status == VerificationStatus.REFUTED:
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
        message = f"Unable to meet precondition"
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
        i - 1,
    )
    return CallTreeAnalysis(
        messages=all_messages.get(),
        verification_status=top_analysis.verification_status,
        num_confirmed_paths=num_confirmed_paths,
    )


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
        with NoTracing():
            is_ch_value = isinstance(old_val, CrossHairValue)
        if is_ch_value:
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
            # Plain object instances are close enough to equal for our purposes
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
            _, _, lines = sourcelines(fn)
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
            exprline = "<unknown>"
            if suggested_filename is not None:
                lines = linecache.getlines(suggested_filename)
                try:
                    exprline = lines[suggested_lineno - 1].strip()
                except IndexError:
                    pass
            detail = f'"{exprline}" yields {detail}'
            return AnalysisMessage(
                message_type, detail, self.filename, self.start_lineno, 0, tb
            )


def make_counterexample_message(
    conditions: Conditions, args: BoundArguments, return_val: object = None
) -> str:
    repr_overrides = context_statespace().extra(LazyCreationRepr).reprs

    with NoTracing():
        arg_memo: dict = {}
        args = deepcopyext(args, CopyMode.REALIZE, arg_memo)
        for orig_id, new_obj in arg_memo.items():
            old_repr = repr_overrides.inner.get(orig_id, None)
            if old_repr:
                repr_overrides.inner[id(new_obj)] = old_repr

    return_val = deep_realize(return_val)

    with NoTracing():
        invocation, retstring = conditions.format_counterexample(
            args, return_val, repr_overrides
        )

        patch_expr = context_statespace().extra(FunctionInterps).patch_string()
        if patch_expr:
            invocation += f" with {patch_expr}"
        if retstring == "None":
            return f"when calling {invocation}"
        else:
            return f"when calling {invocation} (which returns {retstring})"


def attempt_call(
    conditions: Conditions,
    short_circuit: ShortCircuitingContext,
    enforced_conditions: EnforcedConditions,
) -> CallAnalysis:
    # TODO: can we invert the top level so that it is no-tracing?
    # Seems like that might yield some more speed-ups.
    with NoTracing():
        fn = conditions.fn
        space = context_statespace()
        msg_gen = MessageGenerator(conditions.src_fn)
    with enforced_conditions.enabled_enforcement(), NoTracing():
        original_args = gen_args(conditions.sig)
        space.checkpoint()
        bound_args = deepcopyext(original_args, CopyMode.BEST_EFFORT, {})

    lcls: Mapping[str, object] = bound_args.arguments
    # In preconditions, __old__ exists but is just bound to the same args.
    # This lets people write class invariants using `__old__` to, for example,
    # demonstrate immutability.
    with NoTracing():
        lcls = {"__old__": AttributeHolder(lcls), **lcls}
        expected_exceptions = conditions.raises
    for precondition in conditions.pre:
        if not precondition.evaluate:
            continue
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
            with NoTracing():
                formatted_tb = tb.format()
            debug(
                "Exception attempting to meet precondition",
                precondition.expr_source,
                ":",
                user_exc,
                formatted_tb,
            )
            return CallAnalysis(
                failing_precondition=precondition,
                failing_precondition_reason=f'it raised "{repr(user_exc)} at {formatted_tb[-1]}"',
            )

    with ExceptionFilter(expected_exceptions) as efilter:
        with NoTracing():
            unenforced_fn = NoEnforce(fn)
            bargs, bkwargs = bound_args.args, bound_args.kwargs
        with enforced_conditions.enabled_enforcement(), short_circuit:
            debug("Starting function body")
            __return__ = unenforced_fn(*bargs, **bkwargs)
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
        (e, tb) = efilter.user_exc
        detail = name_of_type(type(e)) + ": " + str(e)
        with NoTracing():
            tb_desc = tb.format()
            frame_filename, frame_lineno = frame_summary_for_fn(conditions.src_fn, tb)
        if not isinstance(e, NotDeterministic):
            space.detach_path()
            detail += " " + make_counterexample_message(conditions, original_args)
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
            # TODO: Do we really need custom equality here? Would love to drop that
            # `deep_eq` function.
            if not deep_eq(old_val, new_val, set()):
                space.detach_path()
                detail = 'Argument "{}" is not marked as mutable, but changed from {} to {}'.format(
                    argname, old_val, new_val
                )
                debug("Mutablity problem:", detail)
                return CallAnalysis(
                    VerificationStatus.REFUTED,
                    [msg_gen.make(MessageType.POST_ERR, detail, None, 0, "")],
                )

    (post_condition,) = conditions.post
    assert post_condition.evaluate is not None
    with ExceptionFilter(expected_exceptions) as efilter:
        # TODO: re-enable post-condition short circuiting. This will require refactoring how
        # enforced conditions and short curcuiting interact, so that post-conditions are
        # selectively run when, and only when, performing a short circuit.
        # with enforced_conditions.enabled_enforcement(), short_circuit:
        debug("Starting postcondition")
        isok = bool(post_condition.evaluate(lcls))
    if efilter.ignore:
        debug("Ignored exception in postcondition.", efilter.analysis)
        return efilter.analysis
    elif efilter.user_exc is not None:
        (e, tb) = efilter.user_exc
        detail = name_of_type(type(e)) + ": " + str(e)
        if not isinstance(e, NotDeterministic):
            space.detach_path()
            detail += " " + make_counterexample_message(
                conditions, original_args, __return__
            )
        debug("exception while calling postcondition:", detail)
        debug("exception traceback:", test_stack(tb))
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
        space.detach_path()
        detail = "false " + make_counterexample_message(
            conditions, original_args, __return__
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


def _mutability_testing_hash(o: object) -> int:
    if isinstance(o, ATOMIC_IMMUTABLE_TYPES):
        return 0
    if hasattr(o, "__ch_is_deeply_immutable__"):
        if o.__ch_is_deeply_immutable__():  # type: ignore
            return 0
        else:
            raise TypeError
    typ = type(o)
    if not hasattr(typ, "__hash__"):  # TODO: test for __hash__ = None (list has this)
        raise TypeError
    # We err on the side of mutability if this object is using the default hash:
    if typ.__hash__ is object.__hash__:
        raise TypeError
    return typ.__hash__(o)


def is_deeply_immutable(o: object) -> bool:
    with TracingOnly(PatchingModule({hash: _mutability_testing_hash})):
        # debug('entered patching context', COMPOSITE_TRACER.modules)
        try:
            hash(o)
            return True
        except TypeError:
            return False


def find_best_sig(
    sigs: List[Signature],
    *args: object,
    **kwargs: Dict[str, object],
) -> Optional[Signature]:
    """Return the first signature which complies with the args."""
    for sig in sigs:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        bindings: typing.ChainMap[object, type] = ChainMap()
        is_valid = True
        for param in sig.parameters.values():
            argval = bound.arguments[param.name]
            value_type = python_type(argval)
            if not dynamic_typing.unify(value_type, param.annotation, bindings):
                is_valid = False
                break
        if is_valid:
            return sig
    return None


def consider_shortcircuit(
    fn: Callable,
    sig: Signature,
    bound: BoundArguments,
    subconditions: Conditions,
    allow_interpretation: bool,
) -> Optional[type]:
    """
    Consider the feasibility of short-circuiting (skipping) a function with the given arguments.

    :return: The type of a symbolic value that could be returned by ``fn``.
    :return: None if a short-circuiting should not be attempted.
    """
    return_type = sig.return_annotation
    if return_type == Signature.empty:
        return_type = object
    elif return_type is None:
        return_type = type(None)

    mutable_args = subconditions.mutable_args
    if allow_interpretation:
        if mutable_args is None or len(mutable_args) > 0:
            # we don't deal with mutation inside the skipped function yet.
            debug("aborting shortcircuit: function has matuable args")
            return None

    # Deduce type vars if necessary
    if len(typing_inspect.get_parameters(return_type)) > 0 or typing_inspect.is_typevar(
        return_type
    ):

        typevar_bindings: typing.ChainMap[object, type] = ChainMap()
        bound.apply_defaults()
        for param in sig.parameters.values():
            argval = bound.arguments[param.name]
            # We don't need all args to be symbolic, but we don't currently
            # short circuit in that case as a heuristic.
            if allow_interpretation and not isinstance(argval, CrossHairValue):
                debug("aborting shortcircuit:", param.name, "is not symbolic")
                return None
            value_type = python_type(argval)
            if not dynamic_typing.unify(value_type, param.annotation, typevar_bindings):
                if allow_interpretation:
                    debug("aborting shortcircuit", param.name, "fails unification")
                    return None
                else:
                    raise CrosshairUnsupported
        return_type = dynamic_typing.realize(sig.return_annotation, typevar_bindings)

    if not allow_interpretation:
        return return_type

    space = context_statespace()
    short_stats, callinto_stats = space.stats_lookahead()
    if callinto_stats.unknown_pct < short_stats.unknown_pct:
        callinto_probability = 1.0
    else:
        callinto_probability = 0.7

    debug("short circuit: call-into probability", callinto_probability)
    do_short_circuit = space.fork_parallel(
        callinto_probability, desc=f"shortcircuit {fn.__name__}"
    )
    return return_type if do_short_circuit else None


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
            with NoTracing():
                argscopy[name] = deepcopyext(val, CopyMode.BEST_EFFORT, {})
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
