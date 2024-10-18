import builtins
import collections
import collections.abc
import contextlib
import functools
import importlib.util
import math
import os
import pathlib
import re
import sys
import threading
import time
import traceback
import types
from dataclasses import dataclass
from enum import Enum
from inspect import (
    BoundArguments,
    Parameter,
    getmodulename,
    getsourcefile,
    getsourcelines,
    isfunction,
)
from types import BuiltinFunctionType, FunctionType, MethodDescriptorType, TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import typing_inspect  # type: ignore

from crosshair.auditwall import opened_auditwall
from crosshair.tracers import COMPOSITE_TRACER, NoTracing, ResumedTracing, is_tracing

_DEBUG_STREAM: Optional[TextIO] = None


def is_iterable(o: object) -> bool:
    try:
        iter(o)  # type: ignore
        return True
    except TypeError:
        return False


def is_hashable(o: object) -> bool:
    return getattr(type(o), "__hash__", None) is not None


def is_pure_python(obj: object) -> bool:
    if isinstance(obj, type):
        return True if "__dict__" in dir(obj) else hasattr(obj, "__slots__")
    elif callable(obj):
        return isfunction(obj)  # isfunction selects "user-defined" functions only
    else:
        return True


def memo(f):
    """Decorate a function taking a single argument with a memoization decorator."""
    saved = {}

    @functools.wraps(f)
    def memo_wrapper(a):
        if a not in saved:
            saved[a] = f(a)
        return saved[a]

    return memo_wrapper


# Valid smtlib identifier chars:  ~ ! @ $ % ^ & * _ - + = < > . ? /
# See the section on "symbols" here:
# https://smtlib.cs.uiowa.edu/papers/smt-lib-reference-v2.6-r2017-07-18.pdf
_SMTLIB_TRANSLATION = str.maketrans("[],", "<>.", " ")


def smtlib_typename(typ: Type) -> str:
    return name_of_type(typ).translate(_SMTLIB_TRANSLATION)


def name_of_type(typ: Type) -> str:
    return typ.__name__ if hasattr(typ, "__name__") else str(typ).split(".")[-1]


def samefile(f1: Optional[str], f2: Optional[str]) -> bool:
    try:
        return f1 is not None and f2 is not None and os.path.samefile(f1, f2)
    except FileNotFoundError:
        return False


def true_type(obj: object) -> Type:
    with NoTracing():
        return type(obj)


CROSSHAIR_EXTRA_ASSERTS = os.environ.get("CROSSHAIR_EXTRA_ASSERTS", "0") == "1"

if CROSSHAIR_EXTRA_ASSERTS:

    def assert_tracing(should_be_tracing):
        def decorator(fn):
            fn_name = fn.__qualname__

            @functools.wraps(fn)
            def check_tracing(*a, **kw):
                if is_tracing() != should_be_tracing:
                    with NoTracing():
                        if should_be_tracing:
                            raise CrossHairInternal(
                                f"should be tracing when calling {fn_name}, but isn't"
                            )
                        else:
                            raise CrossHairInternal(
                                f"should not be tracing when calling {fn_name}, but is"
                            )
                return fn(*a, **kw)

            return check_tracing

        return decorator

else:

    def assert_tracing(should_be_tracing):
        def decorator(fn):
            return fn

        return decorator


class IdKeyedDict(collections.abc.MutableMapping):
    def __init__(self):
        # Confusingly, we hold both the key object and value object in
        # our inner dict. Holding the key object ensures that we don't
        # GC the key object, which could lead to reusing the same id()
        # for a different object.
        self.inner: Dict[int, Tuple[object, object]] = {}

    def __getitem__(self, k):
        return self.inner.__getitem__(id(k))[1]

    def __setitem__(self, k, v):
        return self.inner.__setitem__(id(k), (k, v))

    def __delitem__(self, k):
        return self.inner.__delitem__(id(k))

    def __iter__(self):
        return map(id, self.inner.__iter__())

    def __len__(self):
        return len(self.inner)


_SOURCE_CACHE: MutableMapping[object, Tuple[str, int, Tuple[str, ...]]] = IdKeyedDict()


def sourcelines(thing: object) -> Tuple[str, int, Tuple[str, ...]]:
    # If it's a bound method, pull the function out:
    while hasattr(thing, "__func__"):
        thing = thing.__func__  # type: ignore
    # Unwrap decorators as necessary:
    while hasattr(thing, "__wrapped__"):
        thing = thing.__wrapped__  # type: ignore
    filename, start_line, lines = "<unknown file>", 0, ()
    ret = _SOURCE_CACHE.get(thing, None)
    if ret is None:
        try:
            filename = getsourcefile(thing)  # type: ignore
            (lines, start_line) = getsourcelines(thing)  # type: ignore
        except (OSError, TypeError):
            pass
        ret = (filename, start_line, tuple(lines))
        _SOURCE_CACHE[thing] = ret
    return ret


def frame_summary_for_fn(
    fn: Callable, frames: traceback.StackSummary
) -> Tuple[str, int]:
    fn_name = fn.__name__
    fn_file = cast(str, getsourcefile(fn))
    for frame in reversed(frames):
        if frame.name == fn_name and samefile(frame.filename, fn_file):
            return (frame.filename, frame.lineno or 1)
    return sourcelines(fn)[:2]


def set_debug(new_debug: bool, output: TextIO = sys.stderr):
    global _DEBUG_STREAM
    if new_debug:
        _DEBUG_STREAM = output
    else:
        _DEBUG_STREAM = None


def in_debug() -> bool:
    return bool(_DEBUG_STREAM)


def debug(*a):
    """
    Print debugging information in CrossHair's nested log output.

    Arguments are serialized with ``str()`` and printed when running in CrossHair's
    verbose mode.

    Avoid passing symbolic values, as taking the string of a
    symbolic will change the path exploration that CrossHair normally takes, leading to
    different outcomes in verbose and non-verbose mode.
    """
    if not _DEBUG_STREAM:
        return
    with NoTracing():
        stack = traceback.extract_stack()
        frame = stack[-2]
        indent = len(stack) - 3
        print(
            "{:06.3f}|{}|{}() {}".format(
                time.monotonic(), " " * indent, frame.name, " ".join(map(str, a))
            ),
            file=_DEBUG_STREAM,
        )


def warn(*a):
    """
    Display a warning to the user.

    It currently does not do more than printing `WARNING:`, followed by the arguments
    serialized with `str` to the `stderr` stream.
    """
    debug("WARNING:", *a)


TracebackLike = Union[None, TracebackType, Sequence[traceback.FrameSummary]]


def ch_stack(
    tb: TracebackLike = None,
    last_n_frames: int = sys.maxsize,
    currently_handling: Optional[BaseException] = None,
) -> str:
    with NoTracing():
        if currently_handling:
            if tb is not None:
                raise CrossHairInternal
            lower_frames = traceback.extract_tb(currently_handling.__traceback__)
            higher_frames = traceback.extract_stack()[:-2]
            frames: Sequence[traceback.FrameSummary] = higher_frames + lower_frames
        elif tb is None:
            frames = traceback.extract_stack()[:-1]
        elif isinstance(tb, TracebackType):
            frames = traceback.extract_tb(tb)
        else:
            frames = tb
        if last_n_frames == sys.maxsize:
            # TODO: let's move ch_stack into core.py
            from crosshair.statespace import optional_context_statespace

            space = optional_context_statespace()
            if space is not None:
                if space._stack_depth_of_context_entry is None:
                    raise CrossHairInternal
                # TODO: I don't think this _stack_depth_of_context_entry is working properly.
                # Debug the next time we notice it:
                last_n_frames = 1 + len(frames) - space._stack_depth_of_context_entry
        output: List[str] = []
        for frame in frames[-last_n_frames:]:
            filename = os.path.split(frame.filename)[1]
            output.append(f"({frame.name} {filename}:{frame.lineno})")
        return " ".join(output)


class ErrorDuringImport(Exception):
    pass


@contextlib.contextmanager
def add_to_pypath(*paths: Union[str, pathlib.Path]) -> Generator:
    old_path = sys.path[:]
    for path in paths:
        sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = old_path


class _TypingAccessDetector:
    accessed = False

    def __bool__(self):
        self.accessed = True
        return False


def import_module(module_name):
    # Some packages like to write tmp files on import,
    # e.g. https://github.com/pschanely/CrossHair/issues/172
    with opened_auditwall():
        orig_modules = set(sys.modules.values())
        result_module = importlib.import_module(module_name)

        return result_module


def load_file(filename: str) -> types.ModuleType:
    """
    Load a module from a file.

    :raises ErrorDuringImport: if the file cannot be imported
    """
    try:
        root_path, module_name = extract_module_from_file(filename)
        with add_to_pypath(root_path):
            return import_module(module_name)
    except Exception as e:
        raise ErrorDuringImport from e


@contextlib.contextmanager
def imported_alternative(name: str, suppress: Tuple[str, ...] = ()):
    """Load an alternative version of a module with some modules suppressed."""
    modules = sys.modules
    orig_module = importlib.import_module(name)  # Ensure the regular version is loaded
    modules.update({k: None for k in suppress})  # type: ignore
    alternative = importlib.reload(orig_module)
    try:
        yield
    finally:
        for k in suppress:
            del modules[k]
        importlib.reload(alternative)


def format_boundargs_as_dictionary(bound_args: BoundArguments) -> str:
    body = ", ".join(f'"{k}": {repr(v)}' for k, v in bound_args.arguments.items())
    return "{" + body + "}"


def format_boundargs(bound_args: BoundArguments) -> str:
    arg_strings: List[str] = []
    for (name, param) in bound_args.signature.parameters.items():
        param_kind = param.kind
        vals = bound_args.arguments.get(name, param.default)
        if param_kind == Parameter.VAR_POSITIONAL:
            arg_strings.extend(map(repr, vals))
        elif param_kind == Parameter.VAR_KEYWORD:
            arg_strings.extend(f"{k}={repr(v)}" for k, v in vals.items())
        else:
            if param_kind == Parameter.POSITIONAL_ONLY:
                use_keyword = False
            elif param_kind == Parameter.KEYWORD_ONLY:
                use_keyword = True
            else:
                use_keyword = param.default is not Parameter.empty
            if use_keyword:
                arg_strings.append(f"{name}={repr(vals)}")
            else:
                arg_strings.append(repr(vals))
    return ", ".join(arg_strings)


UNABLE_TO_REPR_TEXT = "<unable to repr>"


def eval_friendly_repr(obj: object) -> str:
    assert not is_tracing()
    with ResumedTracing(), EvalFriendlyReprContext() as ctx:
        try:
            # TODO: probably only the repr should have tracing enabled
            return ctx.cleanup(repr(obj))
        except Exception as e:
            if isinstance(e, (IgnoreAttempt, UnexploredPath)):
                raise
            debug("Repr failed, ", type(e), ":", str(e))
            debug("Repr failed at:", ch_stack(currently_handling=e))
            return UNABLE_TO_REPR_TEXT


@dataclass(frozen=True)
class ReferencedIdentifier:
    modulename: str
    qualname: str

    def __str__(self):
        if self.modulename in ("builtins", ""):
            return self.qualname
        else:
            return f"{self.modulename}.{self.qualname}"


def callable_identifier(cls: Callable) -> ReferencedIdentifier:
    return ReferencedIdentifier(cls.__module__, cls.__qualname__)


def method_identifier(fn: Callable) -> ReferencedIdentifier:
    if getattr(fn, "__objclass__", None):
        clsref = callable_identifier(fn.__objclass__)  # type: ignore
        return ReferencedIdentifier(
            clsref.modulename, f"{clsref.qualname}.{fn.__name__}"
        )
    return callable_identifier(fn)


# Objects of these types are known to always be *deeply* immutable:
ATOMIC_IMMUTABLE_TYPES = (
    type(None),
    bool,
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


class EvalFriendlyReprContext:
    """
    Monkey-patch repr() to make some cases more ammenible to eval().

    In particular:
    * object instances repr as "object()" rather than "<object object at ...>"
    * non-finite floats like inf repr as 'float("inf")' rather than just 'inf'
    * functions repr as their fully qualified names
    * enums repr like "Color.RED" instead of "<Color.RED: 0>"
    * uses the walrus (:=) operator to faithfully represent aliased values

    Use the cleanup method to strip unnecessary assignments from the output.

    >>> with EvalFriendlyReprContext() as ctx:
    ...   ctx.cleanup(repr(object()))
    'object()'
    >>> with EvalFriendlyReprContext() as ctx:
    ...   ctx.cleanup(repr(float("nan")))
    'float("nan")'

    The same context can be re-used to perform aliasing across multiple calls to repr:

    >>> lst = []
    >>> ctx = EvalFriendlyReprContext()
    >>> with ctx:
    ...   part1 = repr(lst)
    >>> with ctx:
    ...   part2 = repr(lst)
    >>> ctx.cleanup(part1 + " and also " + part2)
    'v1:=[] and also v1'
    """

    def __init__(self, instance_overrides: Optional[IdKeyedDict] = None):
        self.instance_overrides = (
            IdKeyedDict() if instance_overrides is None else instance_overrides
        )
        self.repr_references: Set[ReferencedIdentifier] = set()

    def __enter__(self):
        if not is_tracing():
            raise CrossHairInternal
        OVERRIDES: Dict[type, Callable[[Any], Union[str, ReferencedIdentifier]]] = {
            object: lambda o: "object()",
            list: lambda o: f"[{', '.join(map(repr, o))}]",  # (de-optimize C-level repr)
            memoryview: lambda o: f"memoryview({repr(o.obj)})",
            FunctionType: callable_identifier,
            BuiltinFunctionType: callable_identifier,
            MethodDescriptorType: method_identifier,
        }
        instance_overrides = self.instance_overrides

        @functools.wraps(builtins.repr)
        def _eval_friendly_repr(obj) -> str:
            oid = id(obj)
            typ = type(obj)
            if obj in instance_overrides:
                repr_fn: Callable[
                    [Any], Union[str, ReferencedIdentifier]
                ] = instance_overrides[obj]
            elif typ == float:
                if math.isfinite(obj):
                    repr_fn = repr
                else:
                    repr_fn = lambda o: f'float("{o}")'
            elif typ in OVERRIDES:
                repr_fn = OVERRIDES[typ]
            elif isinstance(obj, Enum) and obj in typ:
                repr_fn = lambda _: ReferencedIdentifier(
                    typ.__module__, f"{typ.__qualname__}.{obj.name}"
                )
            elif isinstance(obj, type):
                repr_fn = callable_identifier
            else:
                repr_fn = repr
            str_or_ref = repr_fn(obj)
            if isinstance(str_or_ref, ReferencedIdentifier):
                self.repr_references.add(str_or_ref)
                return str_or_ref.qualname
            value_str = str_or_ref
            if isinstance(obj, (ATOMIC_IMMUTABLE_TYPES, Enum)):
                return value_str
            name = f"_ch_efr_{oid}_"
            instance_overrides[obj] = lambda _: name
            return value_str if value_str == name else f"{name}:={value_str}"

        self.patches = {repr: _eval_friendly_repr}
        COMPOSITE_TRACER.patching_module.add(self.patches)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        COMPOSITE_TRACER.patching_module.pop(self.patches)

    def cleanup(self, output: str) -> str:
        counts = collections.Counter(re.compile(r"\b_ch_efr_\d+_\b").findall(output))
        assignment_remaps = {}
        nextvarnum = 1
        for (varname, count) in counts.items():
            if count > 1:
                assignment_remaps[varname + ":="] = f"v{nextvarnum}:="
                assignment_remaps[varname] = f"v{nextvarnum}"
                nextvarnum += 1
        return re.compile(r"\b(_ch_efr_\d+_)\b(\:\=)?").sub(
            lambda match: assignment_remaps.get(match.group(), ""), output
        )


def extract_module_from_file(filename: str) -> Tuple[str, str]:
    module_name = getmodulename(filename)
    dirs = []
    if module_name and module_name != "__init__":
        dirs.append(module_name)
    path = os.path.split(os.path.realpath(filename))[0]
    while os.path.exists(os.path.join(path, "__init__.py")):
        path, cur = os.path.split(path)
        dirs.append(cur)
    dirs.reverse()
    module = ".".join(dirs)
    return path, module


def renamed_function(fn: FunctionType, new_name: str):
    """Produced a completely renamed function"""
    return FunctionType(
        fn.__code__.replace(co_name=new_name, co_filename=new_name + ".py"),
        fn.__globals__,
        new_name,
        fn.__defaults__,
        fn.__closure__,
    )


_T = TypeVar("_T")


class DynamicScopeVar(Generic[_T]):
    """
    Manage a hidden value that can get passed through the callstack.

    This has similar downsides to threadlocals/globals; it should be
    used sparingly.

    >>> _VAR = DynamicScopeVar(int)
    >>> with _VAR.open(42):
    ...   _VAR.get()
    42
    """

    def __init__(self, typ: Type[_T], name_for_debugging: str = ""):
        self._local = threading.local()
        self._name = name_for_debugging

    @contextlib.contextmanager
    def open(self, value: _T, reentrant: bool = True):
        _local = self._local
        old_value = getattr(_local, "value", None)
        if not reentrant:
            assert old_value is None, f"Already in a {self._name} context"
        _local.value = value
        try:
            yield value
        finally:
            assert getattr(_local, "value", None) is value
            _local.value = old_value

    def get(self, default: Optional[_T] = None) -> _T:
        ret = getattr(self._local, "value", None)
        if ret is not None:
            return ret
        if default is not None:
            return default
        assert False, f"Not in a {self._name} context"

    def get_if_in_scope(self) -> Optional[_T]:
        return getattr(self._local, "value", None)


class AttributeHolder:
    def __init__(self, attrs: Mapping[str, object]):
        for (k, v) in attrs.items():
            self.__dict__[k] = v


class CrossHairValue:
    """Base class for values that are pretending to be other values."""

    pass


class ControlFlowException(BaseException):
    # CrossHair sometimes uses exceptions to abort a path mid-execution.
    # We extend such exceptions from BaseException instead of Exception,
    # because expect that user code will usually only handle Exception.
    pass


class CrossHairInternal(ControlFlowException):
    def __init__(self, *a):
        ControlFlowException.__init__(self, *a)
        if in_debug():
            debug("CrossHairInternal:", str(self))
            debug("CrossHairInternal stack trace:")
            for entry in traceback.format_stack()[:-1]:
                for line in entry.splitlines():
                    debug("", line)


class UnexploredPath(ControlFlowException):
    pass


class UnknownSatisfiability(UnexploredPath):
    def __init__(self, *a):
        UnexploredPath.__init__(self, *a)
        debug("UnknownSatisfiability", str(self))


class NotDeterministic(Exception):
    pass


class PathTimeout(UnexploredPath):
    pass


class CrosshairUnsupported(UnexploredPath):
    def __init__(self, *a):
        debug("CrosshairUnsupported: ", str(self))
        debug(" Stack trace:\n" + "".join(traceback.format_stack()))


class IgnoreAttempt(ControlFlowException):
    def __init__(self, *a):
        if in_debug():
            debug(f"IgnoreAttempt", *a)
            debug("IgnoreAttempt stack:", ch_stack())


ExtraUnionType = getattr(types, "UnionType") if sys.version_info >= (3, 10) else None


def origin_of(typ: Type) -> Type:
    if hasattr(typ, "__origin__"):
        return typ.__origin__
    elif ExtraUnionType and isinstance(typ, ExtraUnionType):
        return cast(Type, Union)
    else:
        return typ


def type_args_of(typ: Type) -> Tuple[Type, ...]:
    if getattr(typ, "__args__", None):
        if ExtraUnionType and isinstance(typ, ExtraUnionType):
            return typ.__args__
        return typing_inspect.get_args(typ, evaluate=True)
    else:
        return ()


def type_arg_of(typ: Type, index: int) -> Type:
    args = type_args_of(typ)
    return args[index] if index < len(args) else object


def mem_usage_kb():
    try:
        import resource
    except ImportError:
        return 0  # do not bother monitoring memory on windows
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / 1024  # (bytes on osx)
    else:
        return usage  # (kb)
