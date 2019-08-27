import inspect
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
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


_T = TypeVar('_T')
class IdentityWrapper(Generic[_T]):
    def __init__(self, o: _T):
        self.o = o

    def __hash__(self):
        return id(self.o)

    def __eq__(self, o):
        return hash(self) == hash(o)

class CrosshairInternal(Exception):
    pass
