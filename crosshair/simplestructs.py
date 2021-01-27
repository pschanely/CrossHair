import collections.abc
import dataclasses
import functools
import itertools
import numbers
import operator
import sys
from typing import Callable, Dict, Mapping, MutableMapping, MutableSequence
from typing import Any, Sequence, Set, Tuple, TypeVar, Union
from crosshair.util import debug
from crosshair.util import is_iterable
from crosshair.util import is_hashable
from crosshair.util import name_of_type

class MapBase(collections.abc.MutableMapping):
    def __eq__(self, other):
        # Make our own __eq__ because the one in abc will hash all of our keys.
        if not isinstance(other, collections.abc.Mapping):
            return NotImplemented
        if len(self) != len(other):
            return False
        for (k, self_value) in self.items():
            found = False
            # We do a slow nested loop search because we don't want to hash the key.
            for (other_key, other_value) in other.items():
                if other_key != k:
                    continue
                if self_value == other_value:
                    found = True
                    break
                else:
                    return False
            if not found:
                return False
        return True

    def _is_subclass_of_(cls, other):
        return other is dict


_MISSING = object()

class SimpleDict(MapBase):
    '''
    #inv: set(self.keys()) == set(dict(self.items()).keys())

    >>> d = SimpleDict([(1, 'one'), (2, 'two')])
    >>> d
    {1: 'one', 2: 'two'}
    >>> d[3] = 'three'
    >>> len(d)
    3
    >>> d[2] = 'cat'
    >>> d[2]
    'cat'
    >>> del d[1]
    >>> list(d.keys())
    [2, 3]
    '''
    contents_: MutableSequence

    def __init__(self, contents: MutableSequence):
        ''' `contents` is assumed to not have duplicate keys. '''
        self.contents_ = contents

    def __getitem__(self, key, default=_MISSING):
        if not is_hashable(key):
            raise TypeError('unhashable type')
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                return v
        if default is _MISSING:
            raise KeyError(key)
        return default

    def __setitem__(self, key, value):
        if not is_hashable(key):
            raise TypeError('unhashable type')
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                self.contents_[i] = (k, value)
                return
        self.contents_.append((key, value))

    def __delitem__(self, key):
        if not is_hashable(key):
            raise TypeError('unhashable type')
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                del self.contents_[i]
                return
        raise KeyError(key)

    def __iter__(self):
        return (k for (k, v) in self.contents_)

    def __reversed__(self):
        return (k for (k, v) in reversed(self.contents_))

    def __bool__(self):
        return (len(self.contents_) > 0).__bool__()

    def __len__(self):
        return self.contents_.__len__()

    def __repr__(self):
        return repr(dict(self.items()))

    def items(self):
        return self.contents_

    def popitem(self):
        if not self.contents_:
            raise KeyError
        (k, v) = self.contents_.pop()
        return (k, v)

    def copy(self):
        return SimpleDict(self.contents_[:])


_DELETED = object()
_NOT_FOUND = object()
class ShellMutableMap(MapBase, collections.abc.MutableMapping):
    def __init__(self, inner: Mapping):
        self._mutations: MutableMapping = SimpleDict([])
        self._inner = inner
        self._len = inner.__len__()

    def __getitem__(self, key):
        ret = self._mutations.get(key, _NOT_FOUND)
        if ret is _DELETED:
            raise KeyError(key)
        elif ret is _NOT_FOUND:
            return self._inner.__getitem__(key)
        else:
            return ret

    if sys.version_info >= (3, 8):
        def __reversed__(self):
            return self._reversed()

    def _reversed(self):
        deleted = []
        mutations = self._mutations
        for k in reversed(mutations):
            if mutations[k] is _DELETED:
                deleted.append(k)
                continue
            else:
                yield k
        inner = self._inner
        last = None
        for k in reversed(inner):
            if k in deleted:
                continue
            else:
                yield k

    def __iter__(self):
        mutations = self._mutations
        suppress = list(mutations.keys()) # check against list to avoid hash
        for k in self._inner:
            if k not in suppress:
                yield k
        for k, v in self._mutations.items():
            if v is not _DELETED:
                yield k

    def __eq__(self, other):
        if not self._mutations:
            return self._inner.__eq__(other)
        if not isinstance(other, collections.abc.Mapping):
            return False
        if len(self) != len(other):
            return False
        for k, v in other.items():
            if k not in self or self[k] != v:
                return False
        return True

    def __bool__(self):
        return bool(self._len > 0)

    def __len__(self):
        return self._len

    def __setitem__(self, key, val):
        if key not in self:
            self._len += 1
        self._mutations[key] = val

    def __delitem__(self, key):
        first_hit = self._mutations.get(key, _NOT_FOUND)
        if first_hit is _DELETED:
            raise KeyError(key)
        if first_hit is _NOT_FOUND:
            if key not in self._inner:
                raise KeyError(key)
        self._mutations[key] = _DELETED
        self._len -= 1

    def __repr__(self):
        return repr(dict(self.items()))

    def _lastitem(self):
        raise KeyError

    def pop(self, key, default=_MISSING):
        # CPython checks the empty case before attempting to hash the key.
        # So this must happen before the hash-ability check:
        if self._len == 0:
            raise KeyError(key)
        try:
            value = self[key]
        except KeyError:
            if default is self._MISSING:
                raise
            return default
        else:
            del self[key]
            return value

    def popitem(self):
        for key in self._reversed():
            val = self.__getitem__(key)
            self.__delitem__(key)
            return (key, val)
        raise KeyError

    def copy(self):
        m = ShellMutableMap(self._inner)
        m._mutations = self._mutations.copy()
        return m


def positive_index(idx: Any, container_len: int) -> int:
    if (idx is not None) and (not hasattr(idx, '__index__')):
        raise TypeError('slice indices must be integers or None or have an __index__ method')
    return idx if idx >= 0 else container_len + idx

def indices(s: slice, container_len: int) -> Tuple[int, int, int]:
    '''
    Pure python version of slice.indices() that doesn't force integers into
    existence.
    '''
    start, stop, step = s.start, s.stop, s.step
    if (step is not None) and (not hasattr(step, '__index__')):
        raise TypeError('slice indices must be integers or None or have an __index__ method')
    if step is None:
        step = 1
    elif step <= 0:
        # fallback to python implementation (this will realize values)
        return s.indices(container_len)
    return (0 if start is None else positive_index(start, container_len),
            container_len if stop is None else positive_index(stop, container_len),
            step)
        
def unidirectional_slice(start: int, stop: int, step: int) -> slice:
    return slice(max(0, start), None if stop < 0 else stop, step)

def unidirectional_slice2(start: int, stop: int, step: int) -> slice:
    return slice(None if start < 0 else start, max(0, stop), step)


@functools.total_ordering
class SeqBase:
    def __hash__(self):
        return hash(list(self))

    def __eq__(self, other):
        if self is other:
            return True
        if not is_iterable(other):
            return False
        if len(self) != len(other):
            return False
        for myval, otherval in zip(self, other):
            if myval is otherval:
                continue
            if myval != otherval:
                return False
        return True

    def __lt__(self, other):
        # NOTE: subclasses will need further type restrictions.
        # For example, `[1,2] <= (1,2)` raises a TypeError.
        if not is_iterable(other):
            return NotImplemented
        for v1, v2 in zip(self, other):
            if v1 == v2:
                continue
            return v1 < v2
        return len(self) < len(other)

    def __bool__(self):
        return bool(self.__len__() > 0)

    def __add__(self, other):
        if isinstance(other, collections.abc.Sequence):
            return SequenceConcatenation(self, other)
        raise TypeError(f'unsupported operand type(s) for +')

    def __radd__(self, other):
        if isinstance(other, collections.abc.Sequence):
            return SequenceConcatenation(other, self)
        raise TypeError(f'unsupported operand type(s) for +')

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("can't multiply by non-int xx")
        if other <= 0:
            # A trick to get an empty thing of the same type!:
            return self[0:0]
        ret = self
        for idx in range(1, other):
            ret = self.__add__(ret)
        return ret

    def __rmul__(self, other):
        return self.__mul__(other)


@dataclasses.dataclass(eq=False)
class SequenceConcatenation(collections.abc.Sequence, SeqBase):
    _first: Sequence
    _second: Sequence

    def __getitem__(self, i:Union[int, slice]):
        '''
        raises: IndexError
        post: _ == (self._first + self._second)[i]
        '''
        first, second = self._first, self._second
        firstlen, secondlen = len(first), len(second)
        totallen = firstlen + secondlen
        if isinstance(i, int):
            if not (0 <= i < self.__len__()):
                raise IndexError(i)
            i = positive_index(i, totallen)
            return first[i] if i < firstlen else second[i - firstlen]
        else:
            start, stop, step = indices(i, totallen)
            bump = 0
            if step > 0:
                if start >= firstlen:
                    return second[unidirectional_slice2(start - firstlen, stop - firstlen, step)]
                if stop <= firstlen:
                    return first[unidirectional_slice2(start, stop, step)]
                if step > 1:
                    bump = ((firstlen - start) % step)
                    if bump != 0:
                        bump = step - bump
                first_output = first[start : stop : step]
                second_output = second[bump + max(0, start - firstlen) : max(0, stop - firstlen) : step]
            else:
                if stop >= firstlen:
                    return second[unidirectional_slice(start - firstlen, stop - firstlen, step)]
                if start < firstlen:
                    return first[unidirectional_slice(start, stop, step)]
                if step < -1:
                    bump = (1 + start - firstlen) % -step
                    if bump != 0:
                        bump = (-step) - bump
                first_output = second[unidirectional_slice(start - firstlen, stop - firstlen, step)]
                second_output = first[unidirectional_slice(firstlen - (1 + bump), stop, step)]
            return SequenceConcatenation(first_output, second_output)

    def __contains__(self, item):
        return self._first.__contains__(item) or self._second.__contains__(item)

    def __iter__(self):
        return itertools.chain(self._first, self._second)

    def __len__(self):
        return len(self._first) + len(self._second)


@dataclasses.dataclass(init=False, eq=False) # type: ignore # (https://github.com/python/mypy/issues/5374)
class SliceView(collections.abc.Sequence, SeqBase):
    seq: Sequence
    start: int
    stop: int

    def __init__(self, seq: Sequence, start: int, stop: int):
        seqlen = seq.__len__()
        if start < 0:
            start = 0
        if stop > seqlen:
            stop = seqlen
        if stop < start:
            stop = start
        self.seq = seq
        self.start = start
        self.stop = stop

    def __getitem__(self, key):
        mylen = self.stop - self.start
        if type(key) is slice:
            start, stop, step = indices(key, mylen)
            if step == 1:
                # Move truncation into indices helper to avoid the nesting of slices here
                return SliceView(self, start, stop)
            else:
                return list(self)[key]
        else:
            key = self.start + positive_index(key, mylen)
            if key < self.start or key >= self.stop:
                raise IndexError(key)
            return self.seq[key]

    def __len__(self) -> int:
        return self.stop - self.start

    def __iter__(self):
        for i in range(self.start, self.stop):
            yield self.seq[i]


@dataclasses.dataclass(eq=False)
class ShellMutableSequence(collections.abc.MutableSequence, SeqBase):
    '''
    A class that wraps a sequence and provides mutating operations, but 
    does not modify the original sequence. It reuses portions of the 
    original list as best it can.
    '''
    inner: Sequence

    __hash__ = None # type: ignore

    def __setitem__(self, k, v):
        inner = self.inner
        old_len = len(inner)
        if isinstance(k, slice):
            if not isinstance(v, collections.abc.Iterable):
                raise TypeError('can only assign an iterable')
            start, stop, step = indices(k, old_len)
            if step != 1:
                # abort cleverness:
                newinner = list(inner)
                newinner[k] = v
                self.inner = newinner
                return
            else:
                newinner = v
        elif isinstance(k, numbers.Integral):
            k = positive_index(k, old_len)
            if not (0 <= k < old_len):
                raise IndexError('list index out of range')
            start, stop = k, k + 1
            newinner = [v]
        else:
            raise TypeError(
                f'indices must be integers or slices, not "{name_of_type(k)}"')

        if stop < start:
            stop = start
        # At this point, `stop` >= `start`
        if start > 0:
            newinner = SequenceConcatenation(inner[:start], newinner)
        elif stop <= 0:
            stop = 0
        # At this point, `stop` must be >= 0
        if stop < old_len:
            newinner = SequenceConcatenation(newinner, inner[stop:])
        self.inner = newinner

    def __delitem__(self, k):
        if isinstance(k, slice):
            self.__setitem__(k, [])
        else:
            mylen = self.inner.__len__()
            idx = positive_index(k, mylen)
            if idx < 0 or idx >= mylen:
                raise IndexError(k)
            self.__setitem__(slice(idx, idx + 1, 1), [])

    def __add__(self, other):
        if isinstance(other, collections.abc.Sequence):
            return ShellMutableSequence(SequenceConcatenation(self, other))
        raise TypeError(f'unsupported operand type(s) for +')

    def __radd__(self, other):
        if isinstance(other, collections.abc.Sequence):
            return ShellMutableSequence(SequenceConcatenation(other, self))
        raise TypeError(f'unsupported operand type(s) for +')

    def __imul__(self, other):
        return ShellMutableSequence(self * other)

    def extend(self, other):
        if not isinstance(other, collections.abc.Iterable):
            raise TypeError('object is not iterable')
        self.inner = SequenceConcatenation(self.inner, other)

    def sort(self, key=None, reverse=False):
        self.inner = sorted(self.inner, key=key, reverse=reverse)

    def copy(self):
        return self[:]

    def __len__(self):
        return self.inner.__len__()
    
    def insert(self, index, item):
        self.__setitem__(slice(index, index, 1), [item])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return ShellMutableSequence(self.inner.__getitem__(key))
        else:
            return self.inner.__getitem__(key)

    def __repr__(self):
        return repr(list(self.__iter__()))

    def __contains__(self, other):
        return self.inner.__contains__(other)

    def __iter__(self):
        return self.inner.__iter__()

    def reverse(self):
        self.inner = list(reversed(self.inner))


AbcSet = collections.abc.Set
class SetBase:
    python_type = set
    def __repr__(self):
        return set(self).__repr__()
    def __hash__(self):
        return hash(set(self))

    def __and__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        AbcSet.__and__(self, x)
    def __or__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        AbcSet.__or__(self, x)
    def __xor__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        AbcSet.__xor__(self, x)
    def __sub__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        AbcSet.__sub__(self, x)
    def __rsub__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        AbcSet.__rsub__(self, x)


class SingletonSet(SetBase, AbcSet):
    '''
    Primarily this exists to avoid hashing values.
    '''
    def __init__(self, item):
        self._item = item
    def __contains__(self, x):
        return x == self._item
    def __iter__(self):
        yield self._item
    def __len__(self):
        return 1


class LazySetCombination(SetBase, AbcSet):
    '''
    An immutable set that is a view over two other sets, and a logical
    operation between them.

    >>> a = {2, 4,    6   }
    >>> b = {   4, 5, 6, 7}
    >>> s = LazySetCombination(lambda a,b: (a and b), a, b)
    >>> sorted(s)
    [4, 6]
    >>> a.add(5)
    >>> sorted(s)
    [4, 5, 6]

    '''
    def __init__(self, op: Callable[[bool, bool], bool], a: Set, b: Set):
        self._op = op
        self._a = a
        self._b = b
    def __contains__(self, x):
        ina = self._a.__contains__(x)
        inb = self._b.__contains__(x)
        return self._op(ina, inb)
    def __iter__(self):
        op, a, b = self._op, self._a, self._b
        def afilter(a_item):
            return op(True, a_item in b)
        def bfilter(b_item):
            ina = b_item in a
            if ina:
                # We've already seen this item and would have returned it
                # while traversing a, if we were supposed to.
                return False
            return op(ina, True)
        return itertools.chain(filter(afilter, a), filter(bfilter, b))
    def __len__(self):
        return sum(1 for i in self.__iter__())


class ShellMutableSet(SetBase, collections.abc.MutableSet):
    '''
    A view over an immutable set, giving it mutating operations
    that replace the underlying datastructure entirely.
    This set also attempts to preserve insertion order of the set,
    assuming the underlying set(s) do so as well.
    '''
    _inner: Set

    def __init__(self, inner=frozenset()):
        if isinstance(inner, AbcSet):
            self._inner = inner
        elif is_iterable(inner):
            # Piggyback on ordered-ness of dictionaries:
            self._inner = {k:None for k in inner}.keys()
            # TODO: this hashes the elements;
            #       we likely want a dedicated ordered set class.
        else:
            raise TypeError

    # methods that just defer to _inner
    def __contains__(self, x):
        return self._inner.__contains__(x)
    def __iter__(self):
        return self._inner.__iter__()
    def __len__(self):
        return self._inner.__len__()
    def __le__(self, x):
        return self._inner.__le__(x)
    def __lt__(self, x):
        return self._inner.__lt__(x)
    def __eq__(self, x):
        return self._inner.__eq__(x)
    def __ne__(self, x):
        return self._inner.__ne__(x)
    def __gt__(self, x):
        return self._inner.__gt__(x)
    def __ge__(self, x):
        return self._inner.__ge__(x)
    def isdisjoint(self, x):
        return self._inner.isdisjoint(x)

    # mutation operations
    def add(self, x):
        self.__ior__(SingletonSet(x))
    def clear(self):
        self._inner = frozenset()
    def pop(self):
        if self:
            x = next(iter(self))
            self.remove(x)
            return x
        else:
            raise KeyError
    def discard(self, x):
        self.__isub__(SingletonSet(x))
    def remove(self, x):
        if x not in self:
            raise KeyError
        self.discard(x)
    def __or__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        return ShellMutableSet(LazySetCombination(operator.or_, self._inner, x))
    __ror__ = __or__
    def __and__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        return ShellMutableSet(LazySetCombination(operator.and_, self._inner, x))
    __rand__ = __and__
    def __xor__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        return ShellMutableSet(LazySetCombination(operator.xor, self._inner, x))
    __rxor__ = __xor__
    def __sub__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        return ShellMutableSet(LazySetCombination(lambda x, y: (x and not y), self._inner, x))
    def __rsub__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        return ShellMutableSet(LazySetCombination(lambda x, y: (y and not x), self._inner, x))

    def __ior__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        self._inner = LazySetCombination(operator.or_, self._inner, x)
        return self
    def  __iand__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        self._inner = LazySetCombination(operator.and_, self._inner, x)
        return self
    def __ixor__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        self._inner = LazySetCombination(operator.xor, self._inner, x)
        return self
    def __isub__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        self._inner = LazySetCombination(lambda x, y: (x and not y), self._inner, x)
        return self
