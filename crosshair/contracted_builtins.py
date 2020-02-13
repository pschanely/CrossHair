import collections
import enum
import builtins as orig_builtins
from functools import singledispatch
from typing import *

from crosshair.core import register_patch
from crosshair.util import debug

_T = TypeVar('_T')
_VT = TypeVar('_VT')

class _Missing(enum.Enum):
    value = 0

_MISSING = _Missing.value


class _BuiltinsCopy:
    pass

_TRUE_BUILTINS: Any = _BuiltinsCopy()
_TRUE_BUILTINS.__dict__.update(orig_builtins.__dict__)


# CPython's len() forces the return value to be a native integer.
# Avoid that requirement by making it only call __len__().
def _len(l):
    return l.__len__() if hasattr(l, '__len__') else [x for x in l].__len__()
register_patch(orig_builtins, _len, 'len')

# Avoid calling __len__().__index__() on the input list.


def _sorted(l, **kw):
    ret = list(l.__iter__())
    ret.sort()
    return ret
register_patch(orig_builtins, _sorted, 'sorted')

# Trick the system into believing that symbolic values are
# native types.

def _issubclass(subclass, superclasses):
    subclass_is_special = hasattr(subclass, '_is_subclass_of_')
    if not subclass_is_special:
        # We could also check superclass(es) for a special method, but
        # the native function won't return True in those cases anyway.
        try:
            ret = _TRUE_BUILTINS.issubclass(subclass, superclasses)
            if ret:
                return True
        except TypeError:
            pass
    if type(superclasses) is not tuple:
        superclasses = (superclasses,)
    for superclass in superclasses:
        if hasattr(superclass, '_is_superclass_of_'):
            method = superclass._is_superclass_of_
            if method(subclass) if hasattr(method, '__self__') else method(subclass, superclass):
                return True
        if subclass_is_special:
            method = subclass._is_subclass_of_
            if method(superclass) if hasattr(method, '__self__') else method(subclass, superclass):
                return True
    return False
register_patch(orig_builtins, _issubclass, 'issubclass')

def _isinstance(obj, types):
    try:
        ret = _TRUE_BUILTINS.isinstance(obj, types)
        if ret:
            return True
    except TypeError:
        pass
    if hasattr(obj, 'python_type'):
        obj_type = obj.python_type
        if hasattr(obj_type, '__origin__'):
            obj_type = obj_type.__origin__
    else:
        obj_type = type(obj)
    return issubclass(obj_type, types)
register_patch(orig_builtins, _isinstance, 'isinstance')

#    # TODO: consider tricking the system into believing that symbolic values are
#    # native types.
#    def patched_type(self, *args):
#        ret = self.originals['type'](*args)
#        if len(args) == 1:
#            ret = _WRAPPER_TYPE_TO_PYTYPE.get(ret, ret)
#        for (original_type, proxied_type) in ProxiedObject.__dict__["_class_proxy_cache"].items():
#            if ret is proxied_type:
#                return original_type
#        return ret


def _implies(condition: bool, consequence: bool) -> bool:
    if condition:
        return consequence
    else:
        return True
register_patch(orig_builtins, _implies, 'implies')


def _hash(obj: Hashable) -> int:
    '''
    post[]: -2**63 <= _ < 2**63
    '''
    # Skip the built-in hash if possible, because it requires the output
    # to be a native int:
    if hasattr(obj, '__hash__'):
        # You might think we'd say "return obj.__hash__()" here, but we need some
        # special gymnastics to avoid "metaclass confusion".
        # See: https://docs.python.org/3/reference/datamodel.html#special-method-lookup
        return type(obj).__hash__(obj)
    else:
        return _TRUE_BUILTINS.hash(obj)
register_patch(orig_builtins, _hash, 'hash')

#def sum(i: Iterable[_T]) -> Union[_T, int]:
#    '''
#    post[]: _ == 0 or len(i) > 0
#    '''
#    return _TRUE_BUILTINS.sum(i)

# def print(*a: object, **kw: Any) -> None:
#    '''
#    post: True
#    '''
#    _TRUE_BUILTINS.print(*a, **kw)


def _repr(arg: object) -> str:
    '''
    post[]: True
    '''
    return _TRUE_BUILTINS.repr(arg)
register_patch(orig_builtins, _repr, 'repr')


@singledispatch
def _max(*values, key=lambda x: x, default=_MISSING):
    return _max_iter(values, key=key, default=default)
register_patch(orig_builtins, _max, 'max')


@_max.register(collections.Iterable)
def _max_iter(values: Iterable[_T], *, key: Callable = lambda x: x, default: Union[_Missing, _VT] = _MISSING) -> _T:
    '''
    pre: bool(values) or default is not _MISSING
    post[]::
      (_ in values) if default is _MISSING else True
      ((_ in values) or (_ is default)) if default is not _MISSING else True
    '''
    kw = {} if default is _MISSING else {'default': default}
    return _TRUE_BUILTINS.max(values, key=key, **kw)


@singledispatch
def _min(*values, key=lambda x: x, default=_MISSING):
    return _min_iter(values, key=key, default=default)
register_patch(orig_builtins, _min, 'min')


@_min.register(collections.Iterable)
def _min_iter(values: Iterable[_T], *, key: Callable = lambda x: x, default: Union[_Missing, _VT] = _MISSING) -> _T:
    '''
    pre: bool(values) or default is not _MISSING
    post[]::
      (_ in values) if default is _MISSING else True
      ((_ in values) or (_ is default)) if default is not _MISSING else True
    '''
    kw = {} if default is _MISSING else {'default': default}
    return _TRUE_BUILTINS.min(values, key=key, **kw)
