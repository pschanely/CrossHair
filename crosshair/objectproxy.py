import copy
import operator
import sys

from crosshair.tracers import NoTracing

#
# Adapted from:
# https://github.com/GrahamDumpleton/wrapt/blob/develop/src/wrapt/wrappers.py
# (which is BSD licenced)
#


class ObjectProxy:
    def _wrapped(self):
        raise NotImplementedError

    def __get_module__(self) -> str:
        return self._wrapped().__module__

    def __set_module__(self, value: str) -> None:
        self._wrapped().__module__ = value

    __module__ = property(__get_module__, __set_module__)  # type: ignore

    def __get_doc__(self):
        return self._wrapped().__doc__

    def __set_doc__(self, value):
        self._wrapped().__doc__ = value

    __doc__ = property(__get_doc__, __set_doc__)  # type: ignore

    # We similar use a property for __dict__. We need __dict__ to be
    # explicit to ensure that vars() works as expected.

    @property
    def __dict__(self):
        return self._wrapped().__dict__

    # Need to also propagate the special __weakref__ attribute for case
    # where decorating classes which will define this. If do not define
    # it and use a function like inspect.getmembers() on a decorator
    # class it will fail. This can't be in the derived classes.

    @property
    def __weakref__(self):
        return self._wrapped().__weakref__

    @property
    def __name__(self):
        return self._wrapped().__name__

    @__name__.setter
    def __name__(self, value):
        self._wrapped().__name__ = value

    @property
    def __class__(self):
        return self._wrapped().__class__

    @__class__.setter
    def __class__(self, value):
        self._wrapped().__class__ = value

    def __get_annotations__(self):
        return self._wrapped().__annotations__

    def __set_annotations__(self, value):
        self._wrapped().__annotations__ = value

    __annotations__ = property(__get_annotations__, __set_annotations__)  # type: ignore

    def __dir__(self):
        return dir(self._wrapped())

    def __str__(self):
        return str(self._wrapped())

    def __bytes__(self):
        return bytes(self._wrapped())

    def __repr__(self):
        return repr(self._wrapped())

    def __reversed__(self):
        return reversed(self._wrapped())

    def __round__(self):
        return round(self._wrapped())

    if sys.hexversion >= 0x03070000:

        def __mro_entries__(self, bases):
            return (self._wrapped(),)

    def __lt__(self, other):
        return self._wrapped() < other

    def __le__(self, other):
        return self._wrapped() <= other

    def __eq__(self, other):
        return self._wrapped() == other

    def __ne__(self, other):
        return self._wrapped() != other

    def __gt__(self, other):
        return self._wrapped() > other

    def __ge__(self, other):
        return self._wrapped() >= other

    def __hash__(self):
        return hash(self._wrapped())

    def __nonzero__(self):
        return bool(self._wrapped())

    def __bool__(self):
        return bool(self._wrapped())

    def __setattr__(self, name, value):
        if hasattr(type(self), name):
            object.__setattr__(self, name, value)

        else:
            setattr(self._wrapped(), name, value)

    def __getattr__(self, name):
        with NoTracing():
            if name == "_wrapped":
                return object.__getattribute__(self, "_wrapped")
            else:
                return getattr(self._wrapped(), name)

    def __delattr__(self, name):
        if hasattr(type(self), name):
            object.__delattr__(self, name)

        else:
            delattr(self._wrapped(), name)

    def __add__(self, other):
        return self._wrapped() + other

    def __sub__(self, other):
        return self._wrapped() - other

    def __mul__(self, other):
        return self._wrapped() * other

    def __matmul__(self, other):
        return self._wrapped() @ other

    def __div__(self, other):
        return operator.div(self._wrapped(), other)

    def __truediv__(self, other):
        return operator.truediv(self._wrapped(), other)

    def __floordiv__(self, other):
        return self._wrapped() // other

    def __mod__(self, other):
        return self._wrapped() % other

    def __divmod__(self, other):
        return divmod(self._wrapped(), other)

    def __pow__(self, other, *args):
        return pow(self._wrapped(), other, *args)

    def __lshift__(self, other):
        return self._wrapped() << other

    def __rshift__(self, other):
        return self._wrapped() >> other

    def __and__(self, other):
        return self._wrapped() & other

    def __xor__(self, other):
        return self._wrapped() ^ other

    def __or__(self, other):
        return self._wrapped() | other

    def __radd__(self, other):
        return other + self._wrapped()

    def __rsub__(self, other):
        return other - self._wrapped()

    def __rmul__(self, other):
        return other * self._wrapped()

    def __rmatmul__(self, other):
        return other @ self._wrapped()

    def __rdiv__(self, other):
        return operator.div(other, self._wrapped())

    def __rtruediv__(self, other):
        return operator.truediv(other, self._wrapped())

    def __rfloordiv__(self, other):
        return other // self._wrapped()

    def __rmod__(self, other):
        return other % self._wrapped()

    def __rdivmod__(self, other):
        return divmod(other, self._wrapped())

    def __rpow__(self, other, *args):
        return pow(other, self._wrapped(), *args)

    def __rlshift__(self, other):
        return other << self._wrapped()

    def __rrshift__(self, other):
        return other >> self._wrapped()

    def __rand__(self, other):
        return other & self._wrapped()

    def __rxor__(self, other):
        return other ^ self._wrapped()

    def __ror__(self, other):
        return other | self._wrapped()

    def __iadd__(self, other):
        return operator.iadd(self._wrapped(), other)

    def __isub__(self, other):
        return operator.isub(self._wrapped(), other)

    def __imul__(self, other):
        return operator.imul(self._wrapped(), other)

    def __itruediv__(self, other):
        return operator.itruediv(self._wrapped(), other)

    def __ifloordiv__(self, other):
        return operator.iflootdiv(self._wrapped(), other)

    def __imod__(self, other):
        return operator.imod(self._wrapped(), other)

    def __ipow__(self, other, *args):
        return operator.ipow(self._wrapped(), other, *args)

    def __ilshift__(self, other):
        return operator.ilshift(self._wrapped(), other)

    def __irshift__(self, other):
        return operator.irshift(self._wrapped(), other)

    def __iand__(self, other):
        return operator.iand(self._wrapped(), other)

    def __ixor__(self, other):
        return operator.ixor(self._wrapped(), other)

    def __ior__(self, other):
        return operator.ior(self._wrapped(), other)

    def __neg__(self):
        return -self._wrapped()

    def __pos__(self):
        return +self._wrapped()

    def __abs__(self):
        return abs(self._wrapped())

    def __invert__(self):
        return ~self._wrapped()

    def __int__(self):
        return int(self._wrapped())

    def __float__(self):
        return float(self._wrapped())

    def __complex__(self):
        return complex(self._wrapped())

    def __oct__(self):
        return oct(self._wrapped())

    def __hex__(self):
        return hex(self._wrapped())

    def __index__(self):
        return operator.index(self._wrapped())

    def __len__(self):
        return len(self._wrapped())

    def __contains__(self, value):
        return value in self._wrapped()

    def __getitem__(self, key):
        return self._wrapped()[key]

    def __setitem__(self, key, value):
        self._wrapped()[key] = value

    def __delitem__(self, key):
        del self._wrapped()[key]

    def __getslice__(self, i, j):
        return self._wrapped()[i:j]

    def __setslice__(self, i, j, value):
        self._wrapped()[i:j] = value

    def __delslice__(self, i, j):
        del self._wrapped()[i:j]

    def __enter__(self):
        return self._wrapped().__enter__()

    def __exit__(self, *args, **kwargs):
        return self._wrapped().__exit__(*args, **kwargs)

    def __iter__(self):
        return iter(self._wrapped())

    def __copy__(self):
        return copy.copy(self._wrapped())

    def __deepcopy__(self, memo):
        ret = copy.deepcopy(self._wrapped())
        memo[id(self)] = ret
        return ret

    def __reduce__(self):
        raise NotImplementedError("object proxy must define __reduce_ex__()")

    def __reduce_ex__(self, protocol):
        raise NotImplementedError("object proxy must define __reduce_ex__()")

    def __call__(self, *args, **kwargs):
        return self._wrapped()(*args, **kwargs)
