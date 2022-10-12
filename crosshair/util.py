import builtins
import collections
import collections.abc
import contextlib
import dataclasses
import dis
import functools
import importlib.util
import inspect
import math
import os
import pathlib
import re
import sys
import threading
import time
import traceback
import types
from types import BuiltinFunctionType, FunctionType, MethodDescriptorType, TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from crosshair.auditwall import opened_auditwall

_DEBUG = False


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
        return inspect.isfunction(
            obj
        )  # isfunction selects "user-defined" functions only
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
            filename = inspect.getsourcefile(thing)  # type: ignore
            (lines, start_line) = inspect.getsourcelines(thing)  # type: ignore
        except (OSError, TypeError):
            pass
        ret = (filename, start_line, tuple(lines))
        _SOURCE_CACHE[thing] = ret
    return ret


def frame_summary_for_fn(
    fn: Callable, frames: traceback.StackSummary
) -> Tuple[str, int]:
    fn_name = fn.__name__
    fn_file = cast(str, inspect.getsourcefile(fn))
    for frame in reversed(frames):
        if frame.name == fn_name and samefile(frame.filename, fn_file):
            return (frame.filename, frame.lineno or 1)
    return sourcelines(fn)[:2]


def set_debug(debug: bool):
    global _DEBUG
    _DEBUG = debug


def in_debug() -> bool:
    global _DEBUG
    return _DEBUG


from crosshair.tracers import NoTracing


def debug(*a):
    """
    Print debugging information in CrossHair's nested log output.

    Arguments are serialized with ``str()`` and printed when running in CrossHair's
    verbose mode.

    Avoid passing symbolic values, as taking the string of a
    symbolic will change the path exploration that CrossHair normally takes, leading to
    different outcomes in verbose and non-verbose mode.
    """
    if not _DEBUG:
        return
    with NoTracing():
        stack = traceback.extract_stack()
        frame = stack[-2]
        indent = len(stack) - 3
        print(
            "{:06.3f}|{}|{}() {}".format(
                time.monotonic(), " " * indent, frame.name, " ".join(map(str, a))
            ),
            file=sys.stderr,
        )


def warn(*a):
    """
    Display a warning to the user.

    It currently does not do more than printing `WARNING:`, followed by the arguments
    serialized with `str` to the `stderr` stream.
    """
    print("WARNING:", " ".join(map(str, a)), file=sys.stderr)


TracebackLike = Union[None, TracebackType, Iterable[traceback.FrameSummary]]


def test_stack(tb: TracebackLike = None) -> str:
    return tiny_stack(tb, ignore=re.compile("^$"))


def tiny_stack(tb: TracebackLike = None, **kw) -> str:
    with NoTracing():
        if tb is None:
            frames: Iterable[traceback.FrameSummary] = traceback.extract_stack()[:-1]
        elif isinstance(tb, TracebackType):
            frames = traceback.extract_tb(tb)
        else:
            frames = tb
        return _tiny_stack_frames(frames, **kw)


def _tiny_stack_frames(
    frames: Iterable[traceback.FrameSummary],
    ignore=re.compile(r".*\b(crosshair|z3|typing_inspect|unittest)\b"),
) -> str:
    output: List[str] = []
    ignore_ct = 0
    for frame in frames:
        if ignore.match(frame.filename) and not frame.filename.endswith("_test.py"):
            ignore_ct += 1
        else:
            if ignore_ct > 0:
                if output:
                    output.append(f"(...x{ignore_ct})")
                ignore_ct = 0
            filename = os.path.split(frame.filename)[1]
            output.append(f"({frame.name} {filename}:{frame.lineno})")
    if ignore_ct > 0:
        output.append(f"(...x{ignore_ct})")
    return " ".join(output)


@dataclasses.dataclass
class CoverageResult:
    offsets_covered: Set[int]
    all_offsets: Set[int]
    opcode_coverage: float


@contextlib.contextmanager
def measure_fn_coverage(*fns: Callable):
    codeobjects = set(fn.__code__ for fn in fns)
    opcode_offsets = {
        code: set(i.offset for i in dis.get_instructions(code)) for code in codeobjects
    }
    offsets_seen: Dict[types.CodeType, Set[int]] = collections.defaultdict(set)

    previous_trace = sys.gettrace()

    # TODO: per-line stats would be nice too
    def trace(frame, event, arg):
        code = frame.f_code
        if code in codeobjects:
            frame.f_trace_opcodes = True
            if event == "opcode":
                assert frame.f_lasti in opcode_offsets[code]
                offsets_seen[code].add(frame.f_lasti)
            if previous_trace:
                previous_trace(frame, event, arg)
                # Discard the prior tracer's return value.
                # (because we want to be the top-level tracer)
            return trace
        else:
            if previous_trace:
                return previous_trace(frame, event, arg)
            else:
                return None

    sys.settrace(trace)

    def result_getter(fn: Optional[Callable] = None):
        if fn is None:
            assert len(fns) == 1
            fn = fns[0]
        possible = opcode_offsets[fn.__code__]
        seen = offsets_seen[fn.__code__]
        return CoverageResult(
            offsets_covered=seen,
            all_offsets=possible,
            opcode_coverage=len(seen) / len(possible),
        )

    try:
        yield result_getter
    finally:
        assert sys.gettrace() is trace
        sys.settrace(previous_trace)


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


def import_alternative(name: str, suppress: Tuple[str, ...] = ()):
    """Load an alternative version of a module with some modules suppressed."""
    modules = sys.modules
    orig_module = importlib.import_module(name)  # Ensure the regular version is loaded
    prev = modules.copy()
    modules.update({k: None for k in suppress})  # type: ignore
    try:
        return importlib.reload(orig_module)
    finally:
        # sys.modules = prev
        pass


def format_boundargs(bound_args: inspect.BoundArguments) -> str:
    arg_strings = []
    for (name, param) in bound_args.signature.parameters.items():
        strval = repr(bound_args.arguments[name])
        use_keyword = param.default is not inspect.Parameter.empty
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            use_keyword = False
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            use_keyword = True
        if use_keyword:
            arg_strings.append(f"{name} = {strval}")
        else:
            arg_strings.append(strval)
    return ", ".join(arg_strings)


UNABLE_TO_REPR_TEXT = "<unable to repr>"


def eval_friendly_repr(obj: object) -> str:
    with EvalFriendlyReprContext() as ctx:
        try:
            return ctx.cleanup(repr(obj))
        except Exception as e:
            if isinstance(e, (IgnoreAttempt, UnexploredPath)):
                raise
            debug("Repr failed, ", type(e), ":", str(e))
            debug("Repr failed at:", test_stack(e.__traceback__))
            return UNABLE_TO_REPR_TEXT


def qualified_class_name(cls: type):
    module = cls.__module__
    if module in ("builtins", None):
        return cls.__qualname__
    elif module:
        return f"{cls.__module__}.{cls.__qualname__}"
    else:
        return cls.__qualname__


def qualified_function_name(fn: FunctionType):
    if getattr(fn, "__objclass__", None):
        return f"{qualified_class_name(fn.__objclass__)}.{fn.__name__}"  # type: ignore
    module = fn.__module__
    if module in ("builtins", None):
        return fn.__qualname__
    elif module:
        return f"{fn.__module__}.{fn.__qualname__}"
    else:
        return fn.__qualname__


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

    def __enter__(self):
        self._orig_repr = _orig_repr = builtins.repr
        OVERRIDES: Dict[type, Callable[[Any], str]] = {
            object: lambda o: "object()",
            float: lambda o: _orig_repr(o) if math.isfinite(o) else f'float("{o}")',
            list: lambda o: f"[{', '.join(map(repr, o))}]",  # (de-optimize C-level repr)
            memoryview: lambda o: f"memoryview({repr(o.obj)})",
            type: qualified_class_name,
            FunctionType: qualified_function_name,
            BuiltinFunctionType: qualified_function_name,
            MethodDescriptorType: qualified_function_name,
        }
        instance_overrides = self.instance_overrides

        @functools.wraps(builtins.repr)
        def _eval_friendly_repr(obj):
            oid = id(obj)
            typ = type(obj)
            if obj in instance_overrides:
                repr_fn = instance_overrides[obj]
            elif typ in OVERRIDES:
                repr_fn = OVERRIDES[typ]
            else:
                repr_fn = self._orig_repr
            value_str = repr_fn(obj)
            if isinstance(obj, ATOMIC_IMMUTABLE_TYPES):
                return value_str
            name = f"_ch_efr_{oid}_"
            instance_overrides[obj] = lambda _: name
            return value_str if value_str == name else f"{name}:={value_str}"

        builtins.repr = _eval_friendly_repr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.repr = self._orig_repr

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
    module_name = inspect.getmodulename(filename)
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


class CrosshairInternal(Exception):
    def __init__(self, *a):
        Exception.__init__(self, *a)
        debug("CrosshairInternal", str(self))
        debug(" Stack trace:\n" + "".join(traceback.format_stack()))


class UnexploredPath(Exception):
    pass


class UnknownSatisfiability(UnexploredPath):
    def __init__(self, *a):
        Exception.__init__(self, *a)
        debug("UnknownSatisfiability", str(self))


class PathTimeout(UnexploredPath):
    pass


class CrosshairUnsupported(UnexploredPath):
    def __init__(self, *a):
        debug("CrosshairUnsupported: ", str(self))
        debug(" Stack trace:\n" + "".join(traceback.format_stack()))


class IgnoreAttempt(Exception):
    def __init__(self, *a):
        debug("IgnoreAttempt", str(self))
