import collections
import enum
import builtins as orig_builtins
from functools import singledispatch
from typing import *

from crosshair.util import debug

_T = TypeVar('_T')
_VT = TypeVar('_VT')


class _Missing(enum.Enum):
    value = 0


_MISSING = _Missing.value


class _BuiltinsCopy:
    pass


_ORIGINALS: Any = _BuiltinsCopy()
_ORIGINALS.__dict__.update(orig_builtins.__dict__)


# CPython's len() forces the return value to be a native integer.
# Avoid that requirement by making it only call __len__().
def len(l):
    return l.__len__() if hasattr(l, '__len__') else [x for x in l].__len__()

# Avoid calling __len__().__index__() on the input list.


def sorted(l, **kw):
    ret = list(l.__iter__())
    ret.sort()
    return ret

# Trick the system into believing that symbolic values are
# native types.

def issubclass(subclass, superclasses):
    subclass_is_special = hasattr(subclass, '_is_subclass_of_')
    if not subclass_is_special:
        # We could also check superclass(es) for a special method, but
        # the native function won't return True in those cases anyway.
        try:
            ret = _ORIGINALS.issubclass(subclass, superclasses)
            if ret:
                return True
        except TypeError:
            pass
    if type(superclasses) is not tuple:
        superclasses = (superclasses,)
    for superclass in superclasses:
        if hasattr(superclass, '_is_superclass_of_'):
            if superclass._is_superclass_of_(subclass):
                return True
        if subclass_is_special:
            if subclass._is_subclass_of_(superclass):
                return True
    return False

def isinstance(obj, types):
    try:
        ret = _ORIGINALS.isinstance(obj, types)
        if ret:
            return True
    except TypeError:
        pass
    if hasattr(obj, 'python_type'):
        obj_type = obj.python_type
    else:
        obj_type = type(obj)
    return issubclass(obj_type, types)

# Trick the system into believing that symbolic values are
# native types.
#    def patched_type(self, *args):
#        ret = self.originals['type'](*args)
#        if len(args) == 1:
#            ret = _WRAPPER_TYPE_TO_PYTYPE.get(ret, ret)
#        for (original_type, proxied_type) in ProxiedObject.__dict__["_class_proxy_cache"].items():
#            if ret is proxied_type:
#                return original_type
#        return ret


def implies(condition: bool, consequence: bool) -> bool:
    if condition:
        return consequence
    else:
        return True


def hash(obj: Hashable) -> int:
    '''
    post[]: -2**63 <= _ < 2**63
    '''
    # Skip the built-in hash if possible, because it requires the output
    # to be a native int:
    return obj.__hash__() if hasattr(obj, '__hash__') else _ORIGINALS.hash(obj)


#def sum(i: Iterable[_T]) -> Union[_T, int]:
#    '''
#    post[]: _ == 0 or len(i) > 0
#    '''
#    return _ORIGINALS.sum(i)

# def print(*a: object, **kw: Any) -> None:
#    '''
#    post: True
#    '''
#    _ORIGINALS.print(*a, **kw)


def repr(*a: object, **kw: Mapping[object, object]) -> str:
    '''
    post[]: True
    '''
    return _ORIGINALS.repr(*a, **kw)


@singledispatch
def max(*values, key=lambda x: x, default=_MISSING):
    return _max_iter(values, key=key, default=default)


@max.register(collections.Iterable)
def _max_iter(values: Iterable[_T], *, key: Callable = lambda x: x, default: Union[_Missing, _VT] = _MISSING) -> _T:
    '''
    pre: bool(values) or default is not _MISSING
    post[]::
      (_ in values) if default is _MISSING else True
      ((_ in values) or (_ is default)) if default is not _MISSING else True
    '''
    kw = {} if default is _MISSING else {'default': default}
    return _ORIGINALS.max(values, key=key, **kw)


@singledispatch
def min(*values, key=lambda x: x, default=_MISSING):
    return _min_iter(values, key=key, default=default)


@min.register(collections.Iterable)
def _min_iter(values: Iterable[_T], *, key: Callable = lambda x: x, default: Union[_Missing, _VT] = _MISSING) -> _T:
    '''
    pre: bool(values) or default is not _MISSING
    post[]::
      (_ in values) if default is _MISSING else True
      ((_ in values) or (_ is default)) if default is not _MISSING else True
    '''
    kw = {} if default is _MISSING else {'default': default}
    return _ORIGINALS.min(values, key=key, **kw)
