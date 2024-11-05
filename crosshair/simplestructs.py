import collections.abc
import copy
import dataclasses
import functools
import itertools
import numbers
import operator
import sys
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from crosshair.core import deep_realize
from crosshair.tracers import NoTracing, ResumedTracing, tracing_iter
from crosshair.util import (
    CrossHairValue,
    assert_tracing,
    is_hashable,
    is_iterable,
    name_of_type,
)


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

    def copy(self):
        raise NotImplementedError

    def __ch_pytype__(self):
        return dict

    def __ch_realize__(self):
        memo = {}
        return {deep_realize(k, memo): v for k, v in tracing_iter(self.items())}

    def __ch_deep_realize__(self, memo):
        return {
            deep_realize(k, memo): deep_realize(v, memo)
            for k, v in tracing_iter(self.items())
        }

    def __repr__(self):
        contents = ", ".join(f"{repr(k)}: {repr(v)}" for (k, v) in self.items())
        return "{" + contents + "}"

    if sys.version_info >= (3, 9):

        def __or__(self, other: Mapping) -> Mapping:
            if not isinstance(other, Mapping):
                raise TypeError
            union_map = self.copy()
            union_map.update(other)
            return union_map

        __ror__ = __or__


_MISSING = object()


class SimpleDict(MapBase):
    """
    A pure Python implementation of a dictionary.
    Intentionally does no hashing (linear searches).

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
    """

    contents_: MutableSequence

    def __init__(self, contents: MutableSequence):
        """
        Initialize with (key, value) pairs.

        ``contents`` is assumed to not have duplicate keys.
        """
        self.contents_ = contents

    def __getitem__(self, key, default=_MISSING):
        if not is_hashable(key):
            raise TypeError("unhashable type")
        for (k, v) in self.contents_:
            if k == key:
                return v
        if default is _MISSING:
            raise KeyError
        return default

    def __setitem__(self, key, value):
        if not is_hashable(key):
            raise TypeError("unhashable type")
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                self.contents_[i] = (k, value)
                return
        self.contents_.append((key, value))

    def __delitem__(self, key):
        if not is_hashable(key):
            raise TypeError("unhashable type")
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                del self.contents_[i]
                return
        raise KeyError

    def __iter__(self):
        return (k for (k, v) in self.contents_)

    def __reversed__(self):
        return (k for (k, v) in reversed(self.contents_))

    def __bool__(self):
        return (len(self.contents_) > 0).__bool__()

    def __len__(self):
        return self.contents_.__len__()

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
            raise KeyError
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
        for k in reversed(inner):
            if k in deleted:
                continue
            else:
                yield k

    def __iter__(self):
        mutations = self._mutations
        suppress = list(mutations.keys())  # check against list to avoid hash
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
            raise KeyError
        if first_hit is _NOT_FOUND:
            if key not in self._inner:
                raise KeyError
        self._mutations[key] = _DELETED
        self._len -= 1

    def _lastitem(self):
        raise KeyError

    def pop(self, key, default=_MISSING):
        # CPython checks the empty case before attempting to hash the key.
        # So this must happen before the hash-ability check:
        if self._len > 0:
            try:
                value = self[key]
            except KeyError:
                pass
            else:
                del self[key]
                return value
        # Not found:
        if default is _MISSING:
            raise KeyError(key)
        return default

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


def normalize_idx(idx: Any, container_len: int) -> int:
    if (idx is not None) and (not hasattr(idx, "__index__")):
        raise TypeError("indices must be integers or slices")
    if idx < 0:
        return idx + container_len
    return idx


def check_idx(idx: Any, container_len: int) -> int:
    if not hasattr(idx, "__index__"):
        raise TypeError("indices must be integers or slices, not str")
    normalized_idx = normalize_idx(idx, container_len)
    if 0 <= normalized_idx < container_len:
        return normalized_idx
    raise IndexError


def clamp_slice(s: slice, container_len: int) -> slice:
    if s.step < 0:
        if s.start < 0 or s.stop >= container_len - 1:
            return slice(0, 0, s.step)

        def clamper(i):
            if i < 0:
                return None
            if i >= container_len:
                return container_len - 1
            return i

    else:

        def clamper(i):
            if i < 0:
                return 0
            if i > container_len:
                return container_len
            return i

    return slice(clamper(s.start), clamper(s.stop), s.step)


def offset_slice(s: slice, offset: int) -> slice:
    return slice(s.start + offset, s.stop + offset, s.step)


def compose_slices(prelen: int, postlen: int, s: slice):
    """Transform a slice to apply to a larger sequence."""
    start, stop = s.start, s.stop
    if start >= 0:
        start += prelen
    else:
        start -= prelen
    if stop >= 0:
        stop += prelen
    else:
        stop -= postlen
    return slice(start, stop, s.step)


def cut_slice(start: int, stop: int, step: int, cut: int) -> Tuple[slice, slice]:
    backwards = step < 0
    if backwards:
        start, stop, step, cut = -start, -stop, -step, -cut
    # Modulous with negatives is super hard to reason about, shift everything >= 0:
    delta = -min(start, stop, cut)
    start, stop, cut = start + delta, stop + delta, cut + delta
    if cut < start:
        lstart, lstop = cut, cut
        rstart, rstop = start, stop
    elif cut > stop:
        lstart, lstop = start, stop
        rstart, rstop = cut, cut
    else:
        mid = min(cut, stop)
        lstart, lstop = start, mid
        empties_at_tail = mid % step
        if empties_at_tail > 0:
            mid += step - empties_at_tail
        rstart = mid
        rstop = stop
    lstart, lstop = lstart - delta, lstop - delta
    rstart, rstop = rstart - delta, rstop - delta
    if backwards:
        lstart, lstop = -lstart, -lstop
        rstart, rstop = -rstart, -rstop
        step = -step
    return (slice(lstart, lstop, step), slice(rstart, rstop, step))


def indices(s: slice, container_len: int) -> Tuple[int, int, int]:
    """
    (Mostly) mimic ``slice.indices``.

    This is a pure Python version of ``slice.indices()`` that doesn't force integers
    into existence.
    Note that, unlike `slice.indices`, this function does not "clamp" the index to the
    range [0, container_len).
    """
    start, stop, step = s.start, s.stop, s.step
    if (step is not None) and (not hasattr(step, "__index__")):
        raise TypeError(
            "slice indices must be integers or None or have an __index__ method"
        )
    if step is None:
        step = 1
    elif step <= 0:
        # fallback to python implementation (this will realize values)
        return s.indices(container_len)
    return (
        0 if start is None else normalize_idx(start, container_len),
        container_len if stop is None else normalize_idx(stop, container_len),
        step,
    )


@functools.total_ordering
class SeqBase(CrossHairValue):
    def __hash__(self):
        # TODO: test
        return hash(tuple(self))

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
            return concatenate_sequences(self, other)
        raise TypeError(f"unsupported operand type(s) for +")

    def __radd__(self, other):
        if isinstance(other, collections.abc.Sequence):
            return concatenate_sequences(other, self)
        raise TypeError(f"unsupported operand type(s) for +")

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
    _len: Optional[int] = None

    def __ch_pytype__(self):
        return tuple

    def __getitem__(self, i: Union[int, slice]):
        """Get the item from the concatenation."""
        first, second = self._first, self._second
        firstlen, secondlen = len(first), len(second)
        totallen = firstlen + secondlen
        if isinstance(i, int):
            i = check_idx(i, totallen)
            return first[i] if i < firstlen else second[i - firstlen]
        else:
            if i.step is None or i.step > 0:
                # This block is functionally redundant with the more general
                # logic afterwards. It exists for additional efficency when
                # we can easily slice using one side or the other.
                if i.stop is not None and 0 <= i.stop <= firstlen:
                    if i.start is None or i.start >= 0:
                        return first.__getitem__(i)
                    if i.start < -secondlen:
                        return first.__getitem__(
                            slice(i.start + secondlen, i.stop, i.step)
                        )
                if i.start is not None and i.start >= firstlen:
                    return second.__getitem__(
                        slice(
                            i.start - firstlen,
                            i.stop
                            if i.stop is None or i.stop < 0
                            else i.stop - firstlen,
                            i.step,
                        )
                    )
            start, stop, step = i.indices(totallen)
            cutpoint = firstlen if step > 0 else firstlen - 1
            slice1, slice2 = cut_slice(start, stop, step, cutpoint)
            if step > 0:
                slice1 = clamp_slice(slice1, firstlen)
                slice2 = clamp_slice(offset_slice(slice2, -firstlen), secondlen)
                return concatenate_sequences(first[slice1], second[slice2])
            else:
                slice1 = clamp_slice(offset_slice(slice1, -firstlen), secondlen)
                slice2 = clamp_slice(slice2, firstlen)
                return concatenate_sequences(second[slice1], first[slice2])

    def __eq__(self, other):
        with NoTracing():
            if not hasattr(other, "__len__"):
                return False
            first, second = self._first, self._second
        if self.__len__() != other.__len__():
            return False
        firstlen = first.__len__()
        return first == other[:firstlen] and second == other[firstlen:]

    def __contains__(self, item):
        return self._first.__contains__(item) or self._second.__contains__(item)

    def __iter__(self):
        return itertools.chain(self._first, self._second)

    def __len__(self):
        if self._len is None:
            self._len = len(self._first) + len(self._second)
        return self._len


@dataclasses.dataclass(eq=False)  # type: ignore # (https://github.com/python/mypy/issues/5374)
class SliceView(collections.abc.Sequence, SeqBase):
    seq: Sequence
    start: int
    stop: int

    def __ch_pytype__(self):
        return tuple

    @staticmethod
    def slice(seq: Sequence, start: int, stop: int) -> Sequence:
        seqlen = seq.__len__()
        left_at_end = start <= 0
        right_at_end = stop >= seqlen
        if left_at_end:
            if right_at_end:
                return seq
            start = 0
        if right_at_end:
            stop = seqlen
        if stop <= start:
            stop = start
        return SliceView(seq, start, stop)

    def __getitem__(self, key):
        mystart = self.start
        mylen = self.stop - mystart
        if type(key) is slice:
            start, stop, step = indices(key, mylen)
            if step == 1:
                clamped = clamp_slice(slice(start, stop, step), mylen)
                slice_start = mystart + clamped.start
                slice_stop = mystart + clamped.stop
                if slice_stop <= slice_start:
                    return SliceView((), 0, 0)
                return SliceView(self.seq, slice_start, slice_stop)
            else:
                return list(self)[key]
        else:
            key = self.start + check_idx(key, mylen)
            return self.seq[key]

    def __len__(self) -> int:
        return self.stop - self.start

    def __iter__(self):
        for i in range(self.start, self.stop):
            yield self.seq[i]


def concatenate_sequences(a: Sequence, b: Sequence) -> Sequence:
    with NoTracing():
        if isinstance(a, list):
            if isinstance(b, list):
                return a + b
            elif isinstance(b, SequenceConcatenation) and isinstance(b._first, list):
                return SequenceConcatenation(a + b._first, b._second)
        elif (
            isinstance(a, SequenceConcatenation)
            and isinstance(b, list)
            and isinstance(a._second, list)
        ):
            return SequenceConcatenation(a._first, a._second + b)
        return SequenceConcatenation(a, b)


def sequence_evaluation(seq: Sequence):
    with NoTracing():
        if is_hashable(seq):
            return seq  # immutable datastructures are fine
        elif isinstance(seq, ShellMutableSequence):
            return seq.inner
        else:
            return list(seq)  # TODO: use tracing_iter() here?


@dataclasses.dataclass(eq=False)
class ShellMutableSequence(collections.abc.MutableSequence, SeqBase):
    """
    Wrap a sequence and provide mutating operations without modifying the original.

    It reuses portions of the original list as best it can.
    """

    inner: Sequence

    __hash__ = None  # type: ignore

    def _spawn(self, items: Sequence) -> "ShellMutableSequence":
        # For overriding in subclasses.
        return ShellMutableSequence(items)

    def __eq__(self, other):
        with NoTracing():
            if isinstance(other, ShellMutableSequence):
                other = other.inner
        return self.inner.__eq__(other)

    def __setitem__(self, k, v):
        inner = self.inner
        old_len = len(inner)
        if isinstance(k, slice):
            if not isinstance(v, collections.abc.Iterable):
                raise TypeError("can only assign an iterable")
            # Make a copy if the argument is a mutable container:
            v = sequence_evaluation(v)
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
            k = check_idx(k, old_len)
            start, stop = k, k + 1
            newinner = [v]
        else:
            raise TypeError(
                f'indices must be integers or slices, not "{name_of_type(k)}"'
            )

        if stop < start:
            stop = start
        # At this point, `stop` >= `start`
        if start > 0:
            newinner = concatenate_sequences(inner[:start], newinner)
        elif stop <= 0:
            stop = 0
        # At this point, `stop` must be >= 0
        if stop < old_len:
            newinner = concatenate_sequences(newinner, inner[stop:])
        self.inner = newinner

    def __delitem__(self, k):
        if isinstance(k, slice):
            if k.step in (None, 1):
                self.__setitem__(k, [])
            else:
                self.inner = list(self.inner)
                self.inner.__delitem__(k)
        else:
            mylen = self.inner.__len__()
            idx = check_idx(k, mylen)
            self.__setitem__(slice(idx, idx + 1, 1), [])

    def __add__(self, other):
        if isinstance(other, collections.abc.Sequence):
            return self._spawn(
                concatenate_sequences(self.inner, sequence_evaluation(other))
            )
        raise TypeError(f"unsupported operand type(s) for +")

    def __radd__(self, other):
        if isinstance(other, collections.abc.Sequence):
            return self._spawn(
                concatenate_sequences(sequence_evaluation(other), self.inner)
            )
        raise TypeError(f"unsupported operand type(s) for +")

    def __imul__(self, other):
        return self._spawn(self * other)

    def append(self, item):
        inner = self.inner
        self.inner = concatenate_sequences(inner, [item])

    def extend(self, other):
        if not isinstance(other, collections.abc.Iterable):
            raise TypeError("object is not iterable")
        self.inner = concatenate_sequences(self.inner, sequence_evaluation(other))

    def index(self, *a) -> int:
        return self.inner.index(*a)

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
            return self._spawn(self.inner.__getitem__(key))
        else:
            return self.inner.__getitem__(key)

    def __repr__(self):
        contents = ", ".join(map(repr, self))
        return f"[{contents}]"

    def __contains__(self, other):
        return self.inner.__contains__(other)

    def __iter__(self):
        return self.inner.__iter__()

    def reverse(self):
        self.inner = list(reversed(self.inner))


AbcSet = collections.abc.Set
AbcMutableSet = collections.abc.MutableSet


def _force_arg_to_set(x: object) -> AbcSet:
    with NoTracing():
        if isinstance(x, AbcSet):
            while isinstance(x, ShellMutableSet):
                x = x._inner
            if isinstance(x, AbcMutableSet):
                # Already known to have unique elements:
                return LinearSet(list(tracing_iter(x)))
            elif isinstance(x, (frozenset, FrozenSetBase)):
                return x  # Immutable set
        if is_iterable(x):
            with ResumedTracing():
                return LinearSet.check_unique_and_create(x)
        raise TypeError


class SetBase(CrossHairValue):
    def __bool__(self):
        itr = iter(self)
        try:
            next(itr)
            return True
        except StopIteration:
            return False

    def __repr__(self):
        return deep_realize(self).__repr__()

    def __and__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        return AbcSet.__and__(self, x)

    def __or__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        return AbcSet.__or__(self, x)

    def __xor__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        return AbcSet.__xor__(self, x)

    __rxor__ = __xor__
    __ror__ = __or__
    __rand__ = __and__

    def __sub__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        return AbcSet.__sub__(self, x)

    def __rsub__(self, x):
        if not isinstance(x, AbcSet):
            return NotImplemented
        return AbcSet.__rsub__(self, x)

    def copy(self):
        return copy.copy(self)

    def difference(self, *itrs):
        for itr in itrs:
            self = self.__sub__(_force_arg_to_set(itr))
        return self

    def intersection(self, *itrs):
        for itr in itrs:
            self = self.__and__(_force_arg_to_set(itr))
        return self

    def isdisjoint(self, x):
        return not (self.intersection(x))

    def issubset(self, x):
        return self <= _force_arg_to_set(x)

    def issuperset(self, x):
        return self >= _force_arg_to_set(x)

    def symmetric_difference(self, x):
        return self.__xor__(_force_arg_to_set(x))

    def union(self, *itrs):
        for itr in itrs:
            self = self.__or__(_force_arg_to_set(itr))
        return self


class FrozenSetBase(SetBase, AbcSet):
    def __ch_realize__(self):
        # We are going to have to hash all of our contents,
        # so just realize everything.
        return self.__ch_deep_realize__({})

    @assert_tracing(False)
    def __ch_deep_realize__(self, memo):
        contents = []
        for item in tracing_iter(self):
            contents.append(deep_realize(item, memo))
        # TODO: This fails to preserve the iteration order;
        # should we do something about that?:
        return frozenset(contents)

    def __ch_pytype__(self):
        return frozenset

    @classmethod
    def _from_iterable(cls, it):
        # overrides collections.abc.Set's version
        return LinearSet.check_unique_and_create(it)

    def __hash__(self):
        return hash(deep_realize(self))


class SingletonSet(FrozenSetBase):
    # Primarily this exists to avoid hashing values.
    # TODO: should we fold uses of this into LinearSet, below?

    def __init__(self, item):
        self._item = item

    def __contains__(self, x):
        return x == self._item

    def __iter__(self):
        yield self._item

    def __len__(self):
        return 1


class EmptySet(FrozenSetBase):
    def __contains__(self, x):
        if not is_hashable(x):
            raise TypeError
        return False

    def __iter__(self):
        return
        yield

    def __len__(self):
        return 0


class LinearSet(FrozenSetBase):
    # Primarily this exists to avoid hashing values.
    # Presumes that its arguments are already unique.

    def __init__(self, items: Iterable):
        self._items = items

    @staticmethod
    def check_unique_and_create(seq):
        accepted = []
        # duplicate detection:
        # (alternatively, we could defer using LazySetCombination)
        for item in seq:
            if not is_hashable(item):
                raise TypeError
            if item not in accepted:
                accepted.append(item)
        return LinearSet(accepted)

    def __contains__(self, x):
        if not is_hashable(x):
            raise TypeError
        for item in self._items:
            if x == item:
                return True
        return False

    def __iter__(self):
        for item in self._items:
            yield item

    def __len__(self):
        return len(self._items)


class LazySetCombination(FrozenSetBase):
    """
    Provide a view over two sets and a logical operation in-between.

    The view itself is an immutable set.

    >>> a = {2, 4,    6   }
    >>> b = {   4, 5, 6, 7}
    >>> s = LazySetCombination(lambda a,b: (a and b), a, b)
    >>> sorted(s)
    [4, 6]
    >>> a.add(5)
    >>> sorted(s)
    [4, 5, 6]
    """

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


class ShellMutableSet(SetBase, AbcMutableSet):
    """
    Provide a mutable view over an immutable set.

    The mutating operations simply replace the underlying
    data structure entirely.
    This set also attempts to preserve insertion order of the set,
    assuming the underlying set(s) do so as well.
    """

    _inner: Union[frozenset, FrozenSetBase]

    @assert_tracing(False)
    def __init__(self, inner: Iterable = EmptySet()):
        if isinstance(inner, (frozenset, FrozenSetBase)):
            self._inner = inner
        elif is_iterable(inner):
            with ResumedTracing():
                self._inner = LinearSet.check_unique_and_create(inner)
        else:
            raise TypeError

    def __ch_realize__(self):
        # Deep realize contents because the real set will want to hash them:
        return set(map(deep_realize, tracing_iter(self)))

    def __ch_pytype__(self):
        return set

    # methods that just defer to _inner
    def __contains__(self, x):
        return self._inner.__contains__(x)

    def __iter__(self):
        # TODO: replace _inner after iteration (to avoid recalculation of lazy sub-sets)
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
        if not is_hashable(x):
            raise TypeError(f"unhashable type: '{name_of_type(type(x))}'")
        self.__ior__(SingletonSet(x))

    def clear(self):
        self._inner = frozenset()

    def difference_update(self, x):
        self._inner = self._inner.difference(x)

    def discard(self, x):
        if not is_hashable(x):
            raise TypeError(f"unhashable type: '{name_of_type(type(x))}'")
        self.__isub__(SingletonSet(x))

    def intersection_update(self, x):
        self._inner = self._inner.intersection(x)

    def pop(self):
        if self:
            x = next(iter(self))
            self.remove(x)
            return x
        else:
            raise KeyError

    def remove(self, x):
        if x not in self:
            raise KeyError
        self.discard(x)

    def symmetric_difference_update(self, x):
        self._inner = self._inner.symmetric_difference(x)

    def update(self, *iterables):
        for itr in iterables:
            additions = _force_arg_to_set(itr)
            self._inner = LazySetCombination(operator.or_, self._inner, additions)

    def __or__(self, x):
        with NoTracing():
            if not isinstance(x, AbcSet):
                return NotImplemented
            return ShellMutableSet(
                LazySetCombination(operator.or_, self._inner, _force_arg_to_set(x))
            )

    __ror__ = __or__

    def __and__(self, x):
        with NoTracing():
            if not isinstance(x, AbcSet):
                return NotImplemented
            return ShellMutableSet(
                LazySetCombination(operator.and_, self._inner, _force_arg_to_set(x))
            )

    __rand__ = __and__

    def __xor__(self, x):
        with NoTracing():
            if not isinstance(x, AbcSet):
                return NotImplemented
            return ShellMutableSet(
                LazySetCombination(operator.xor, self._inner, _force_arg_to_set(x))
            )

    __rxor__ = __xor__

    def __sub__(self, x):
        with NoTracing():
            if not isinstance(x, AbcSet):
                return NotImplemented
            # TODO: why not lazy set combination here?
            return ShellMutableSet(
                LazySetCombination(lambda x, y: (x and not y), self._inner, x)
            )

    def __rsub__(self, x):
        with NoTracing():
            if not isinstance(x, AbcSet):
                return NotImplemented
            return ShellMutableSet(
                LazySetCombination(lambda x, y: (y and not x), self._inner, x)
            )

    def __ior__(self, x):
        with NoTracing():
            if not isinstance(x, AbcSet):
                return NotImplemented
            self._inner = LazySetCombination(
                operator.or_, self._inner, _force_arg_to_set(x)
            )
            return self

    def __iand__(self, x):
        with NoTracing():
            if not isinstance(x, AbcSet):
                return NotImplemented
            self._inner = LazySetCombination(
                operator.and_, self._inner, _force_arg_to_set(x)
            )
            return self

    def __ixor__(self, x):
        with NoTracing():
            if not isinstance(x, AbcSet):
                return NotImplemented
            self._inner = LazySetCombination(
                operator.xor, self._inner, _force_arg_to_set(x)
            )
            return self

    def __isub__(self, x):
        with NoTracing():
            if not isinstance(x, AbcSet):
                return NotImplemented
            self._inner = LazySetCombination(
                lambda x, y: (x and not y), self._inner, _force_arg_to_set(x)
            )
            return self
