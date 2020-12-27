import builtins
import contextlib
import importlib
import importlib.util
import inspect
import functools
import math
import os
import sys
import threading
import traceback
import typing
from typing import *


_DEBUG = False

def is_iterable(o: object) -> bool:
    try:
        iter(o) # type: ignore
        return True
    except TypeError:
        return False

def is_hashable(o: object) -> bool:
    return getattr(o, '__hash__', None) is not None

def name_of_type(typ: Type) -> str:
    return typ.__name__ if hasattr(typ, '__name__') else str(typ).split('.')[-1]

def samefile(f1: Optional[str], f2: Optional[str]) -> bool:
    try:
        return f1 is not None and f2 is not None and os.path.samefile(f1, f2)
    except FileNotFoundError:
        return False

def source_position(thing: object) -> Tuple[str, int]:
    ''' Best-effort source filename and line number. '''
    filename, start_line = ('<unknown file>', 0)
    try:
        filename = inspect.getsourcefile(thing)
        (_, start_line) = inspect.getsourcelines(thing)
    except OSError:
        pass
    except TypeError:  # Note getsourcefile raises TypeError for builtins
        pass
    return filename, start_line

def frame_summary_for_fn(frames: traceback.StackSummary, fn: Callable) -> Tuple[str, int]:
    fn_name = fn.__name__
    fn_file = cast(str, inspect.getsourcefile(fn))
    for frame in reversed(frames):
        if (frame.name == fn_name and
            samefile(frame.filename, fn_file)):
            return (frame.filename, frame.lineno)
    try:
        (_, fn_start_line) = inspect.getsourcelines(fn)
        return fn_file, fn_start_line
    except OSError:
        debug(f'Unable to get source information for function {fn_name} in file "{fn_file}"')
        return (fn_file, 0)

def set_debug(debug: bool):
    global _DEBUG
    _DEBUG = debug


def debug(*a):
    if not _DEBUG:
        return
    stack = traceback.extract_stack()
    frame = stack[-2]
    indent = len(stack) - 3
    print('|{}|{}() {}'.format(
        ' ' * indent, frame.name, ' '.join(map(str, a))), file=sys.stderr)


class NotFound(ValueError):
    pass

class ErrorDuringImport(Exception):
    pass


@contextlib.contextmanager
def add_to_pypath(path: str):
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path = old_path

def walk_qualname(obj: object, name: str) -> object:
    '''
    >>> walk_qualname(list, 'append') == list.append
    True
    >>> class Foo:
    ...   class Bar:
    ...     def doit():
    ...       pass
    >>> walk_qualname(Foo, 'Bar.doit') == Foo.Bar.doit
    True
    '''
    for part in name.split('.'):
        if part == '<locals>':
            raise ValueError(
                'object defined inline are non-addressable(' + name + ')')
        if not hasattr(obj, part):
            raise NotFound('Name "' + part + '" not found')
        obj = getattr(obj, part)
    return obj

@contextlib.contextmanager
def typing_access_detector():
    class Detector:
        accessed = False
        def __bool__(self):
            self.accessed = True
            return False
    typing.TYPE_CHECKING = Detector()
    try:
        yield typing.TYPE_CHECKING
    finally:
        typing.TYPE_CHECKING = False

def import_module(module_name):
    with typing_access_detector() as detector:
        module = importlib.import_module(module_name)

    # It's common to avoid circular imports with TYPE_CHECKING guards.
    # We need those imports however, so we work around this by re-importing
    # modules that use such guards, with the TYPE_CHECKING flag turned on.
    # (see https://github.com/pschanely/CrossHair/issues/32)
    if detector.accessed:
        typing.TYPE_CHECKING = True
        try:
            importlib.reload(module)
        finally:
            typing.TYPE_CHECKING = False
    return module

def load_file(filename: str) -> object:
    ''' Can be a filename or module name '''
    try:
        root_path, module_name = extract_module_from_file(filename)
        with add_to_pypath(root_path):
            return import_module(module_name)
    except Exception as e:
        raise ErrorDuringImport(e, traceback.extract_tb(sys.exc_info()[2])[-1])


@contextlib.contextmanager
def eval_friendly_repr():
    '''
    Context manager that monkey patches repr() to make some cases more ammenible
    to eval(). In particular:
    * object instances repr as "object()" rather than "<object object at ...>"
    * non-finite floats like inf repr as 'float("inf")' rather than just 'inf'

    >>> with eval_friendly_repr():
    ...   repr(object())
    'object()'
    >>> with eval_friendly_repr():
    ...   repr(float("nan"))
    'float("nan")'
    >>> # returns to original behavior afterwards:
    >>> repr(float("nan"))
    'nan'
    >>> repr(object())[:20]
    '<object object at 0x'
    '''
    _orig = builtins.repr
    OVERRIDES = {
        object: lambda o: 'object()',
        float: lambda o: _orig(o) if math.isfinite(o) else f'float("{o}")'
    }
    @functools.wraps(_orig)
    def _eval_friendly_repr(obj):
        typ = type(obj)
        if typ in OVERRIDES:
            return OVERRIDES[typ](obj)
        return _orig(obj)
    builtins.repr = _eval_friendly_repr
    try:
        yield
    finally:
        assert builtins.repr is _eval_friendly_repr
        builtins.repr = _orig


def load_by_qualname(name: str) -> object:
    '''
    >>> type(load_by_qualname('os'))
    <class 'module'>
    >>> type(load_by_qualname('os.path'))
    <class 'module'>
    >>> type(load_by_qualname('os.path.join'))
    <class 'function'>
    >>> type(load_by_qualname('pathlib.Path'))
    <class 'type'>
    >>> type(load_by_qualname('pathlib.Path.is_dir'))
    <class 'function'>
    '''
    parts = name.split('.')
    # try progressively shorter prefixes until we can load a module:
    for i in reversed(range(1, len(parts) + 1)):
        cur_module_name = '.'.join(parts[:i])
        try:
            try:
                spec_exists = importlib.util.find_spec(cur_module_name) is not None
                if not spec_exists:
                    raise ModuleNotFoundError(f"No module named '{cur_module_name}'")
            except ModuleNotFoundError:
                if i == 1:
                    raise
                else:
                    continue
            module = import_module(cur_module_name)
        except Exception as e:
            raise ErrorDuringImport(e, traceback.extract_tb(sys.exc_info()[2])[-1])
        remaining = '.'.join(parts[i:])
        if remaining:
            return walk_qualname(module, remaining)
        else:
            return module
    assert False


def extract_module_from_file(filename: str) -> Tuple[str, str]:
    module_name = inspect.getmodulename(filename)
    dirs = []
    if module_name and module_name != '__init__':
        dirs.append(module_name)
    path = os.path.split(os.path.realpath(filename))[0]
    while os.path.exists(os.path.join(path, '__init__.py')):
        path, cur = os.path.split(path)
        dirs.append(cur)
    dirs.reverse()
    module = '.'.join(dirs)
    return path, module


def memo(f):
    """ Memoization decorator for a function taking a single argument """
    saved = {}
    @functools.wraps(f)
    def memo_wrapper(a):
        if not a in saved:
            saved[a] = f(a)
        return saved[a]
    return memo_wrapper


_T = TypeVar('_T')

class DynamicScopeVar(Generic[_T]):
    '''
    Manages a hidden value that can get passed through the callstack.

    >>> _VAR = DynamicScopeVar(int)
    >>> with _VAR.open(42):
    ...   _VAR.get()
    42

    This has similar downsides to threadlocals/globals; it should be
    used sparingly.
    '''
    def __init__(self,
                 typ: Type[_T],
                 name_for_debugging: str = ''):
        self._local = threading.local()
        self._name = name_for_debugging

    @contextlib.contextmanager
    def open(self, value: _T, reentrant: bool = True):
        _local = self._local
        old_value = getattr(_local, 'value', None)
        if not reentrant:
            assert old_value is None, f'Already in a {self._name} context'
        self._local.value = value
        yield
        assert getattr(_local, 'value', None) is value
        _local.value = old_value

    def get(self) -> _T:
        ret = getattr(self._local, 'value', None)
        assert ret is not None, f'Not in a {self._name} context'
        return ret

    def get_if_in_scope(self) -> Optional[_T]:
        return getattr(self._local, 'value', None)


class IdentityWrapper(Generic[_T]):
    def __init__(self, o: _T):
        self.o = o

    def __hash__(self):
        return id(self.o)

    def __eq__(self, o):
        return hash(self) == hash(o)

    def get(self):
        return self.o


class AttributeHolder:
    def __init__(self, attrs: Mapping[str, object]):
        for (k, v) in attrs.items():
            self.__dict__[k] = v


class CrosshairInternal(Exception):
    def __init__(self, *a):
        Exception.__init__(self, *a)
        debug('CrosshairInternal', str(self))


class UnexploredPath(Exception):
    pass


class UnknownSatisfiability(UnexploredPath):
    def __init__(self, *a):
        Exception.__init__(self, *a)
        debug('UnknownSatisfiability', str(self))


class PathTimeout(UnexploredPath):
    pass


class CrosshairUnsupported(UnexploredPath):
    def __init__(self, *a):
        debug('CrosshairUnsupported: ', str(self))
        debug(' Stack trace:\n' + ''.join(traceback.format_stack()))


class IgnoreAttempt(Exception):
    def __init__(self, *a):
        debug('IgnoreAttempt', str(self))
