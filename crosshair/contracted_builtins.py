import collections
import enum
import builtins as orig_builtins
from functools import singledispatch
from typing import *

_T = TypeVar('_T')
_VT = TypeVar('_VT')

class _Missing(enum.Enum):
    value = 0
_MISSING = _Missing.value

class BuiltinsCopy:
    pass
_ORIGINALS:Any = BuiltinsCopy()
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
def isinstance(obj, types):
    ret = _ORIGINALS.isinstance(obj, types)
    if not ret:
        if hasattr(obj, 'python_type'):
            obj_type = obj.python_type
        else:
            obj_type = type(obj)
        if obj_type is types or (type(types) is tuple and any(t is obj_type for t in types)):
            ret = True
    return ret

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



def hash(obj: Hashable) -> int:
    '''
    post: -2**63 <= return < 2**63
    '''
    return _ORIGINALS.hash(obj)

def sum(i: Iterable[_T]) -> Union[_T, int]:
    '''
    post: return == 0 or len(i) > 0
    '''
    return _ORIGINALS.sum(i)

#def print(*a: object, **kw: Any) -> None:
#    '''
#    post: True
#    '''
#    _ORIGINALS.print(*a, **kw)

@singledispatch
def max(*values: _T, key: Callable[[_T], Any] = lambda x:x, default: Union[_Missing, _VT] = _MISSING) -> _T:
    '''
    #pre: len(values) > 0 or default is not _MISSING
    pre: len(values) > 0
    #pre: map(key, values)
    post: default == _MISSING
    post: (return in values) if default is _MISSING else True
    #post: (return in values or return is default) if default is not _MISSING else True
    '''
    kw = {} if default is _MISSING else {'default': default}
    return _ORIGINALS.max(*values, key=key, **kw)

@max.register
def _(values: collections.Iterable, *, key: Callable[[_T], Any] = lambda x:x, default: Union[_Missing, _VT] = _MISSING) -> _T:
    '''
    #pre: len(list(values)) > 0 or default is not _MISSING
    pre: len(values) > 0
    #pre: map(key, values)
    post: default == _MISSING
    post: (return in values) if default is _MISSING else True
    #post: ((return in values) or (return is default)) if default is not _MISSING else True
    '''
    kw = {} if default is _MISSING else {'default': default}
    return _ORIGINALS.max(values, key=key, **kw)
