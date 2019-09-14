import collections
import importlib
import inspect
import functools
import gc
import os
import sys
import traceback
from typing import *


_DEBUG = False
def set_debug(debug:bool):
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
            raise Exception('Name "' + part + '" not found')
        obj = getattr(obj, part)
    return obj

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
    for i in reversed(range(1, len(parts) + 1)):
        cur_module_name = '.'.join(parts[:i])
        try:
            module = importlib.import_module(cur_module_name)
        except ModuleNotFoundError:
            continue
        remaining = '.'.join(parts[i:])
        if remaining:
            return walk_qualname(module, remaining)
        else:
            return module
    return None

def extract_module_from_file(filename: str) -> Tuple[str, str]:
    dirs = [m for m in [inspect.getmodulename(filename)] if m]
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
class IdentityWrapper(Generic[_T]):
    def __init__(self, o: _T):
        self.o = o

    def __hash__(self):
        return id(self.o)

    def __eq__(self, o):
        return hash(self) == hash(o)

class AttributeHolder:
    def __init__(self, attrs: Mapping[str, object]):
        for (k, v) in attrs.items():
            self.__dict__[k] = v

class CrosshairInternal(Exception):
    pass

_MAP = None
def get_subclass_map():
    '''
    Crawls all types presently in memory and makes a map from parent to child classes.
    Only direct children are included.
    Does not yet handle "protocol" subclassing (eg "Iterator", "Mapping", etc).

    >>> CrosshairInternal in get_subclass_map()[Exception]
    True
    '''
    global _MAP
    if _MAP is None:
        classes = [x for x in gc.get_objects() if isinstance(x, type)]
        subclass = collections.defaultdict(list)
        for cls in classes:
            for base in cls.__bases__:
                subclass[base].append(cls)
        _MAP = subclass
    return _MAP

def rebuild_subclass_map():
    global _MAP
    _MAP = None

