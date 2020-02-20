import collections.abc
import dataclasses
import itertools
from typing import Mapping, MutableSequence, Sequence, Tuple, TypeVar, Union
from crosshair.util import is_iterable

_MISSING = object()


class SimpleDict(collections.abc.MutableMapping):
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
        # TODO: assumes initial data has no duplicate keys. Is that right?
        self.contents_ = contents

    def _is_subclass_of_(cls, other):
        return other is dict

    def __getitem__(self, key, default=_MISSING):
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                return v
        if default is _MISSING:
            raise KeyError(key)
        return default

    def __setitem__(self, key, value):
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                self.contents_[i] = (k, value)
                return
        self.contents_.append((key, value))

    def __delitem__(self, key):
        for (i, (k, v)) in enumerate(self.contents_):
            if k == key:
                del self.contents_[i]
                return

    def __iter__(self):
        return (k for (k, v) in self.contents_)

    def __eq__(self, other):
        # Make our own __eq__ because the one in abc will hash all of our keys.
        if not isinstance(other, Mapping):
            return NotImplemented
        if len(self) != len(other):
            return False
        for (k, self_value) in self.contents_:
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

    def __bool__(self):
        return (len(self.contents_) > 0).__bool__()

    def __len__(self):
        return len(self.contents_)

    def __repr__(self):
        return str(dict(self.items()))



def positive_index(idx: int, container_len: int) -> int:
    return idx if idx >= 0 else container_len + idx

def indices(s: slice, container_len: int) -> Tuple[int, int, int]:
    '''
    Pure python version of slice.indices() that doesn't force integers into
    existence.
    '''
    start, stop, step = s.start, s.stop, s.step
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
        

class SeqBase:
    def __hash__(self):
        return hash(list(self))

    def __eq__(self, other):
        if not is_iterable(other):
            return False
        if len(self) != len(other):
            return False
        for myval, otherval in zip(self, other):
            if myval != otherval:
                return False
        return True

    def __bool__(self):
        return bool(self.__len__() > 0)

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

    def __add__(self, other):
        return SequenceConcatenation(self, other)

    def __radd__(self, other):
        return SequenceConcatenation(other, self)


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

    def __add__(self, other):
        return SequenceConcatenation(self, other)

    def __radd__(self, other):
        return SequenceConcatenation(other, self)

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
            start, stop, step = indices(k, old_len)
            if step != 1:
                # abort cleverness:
                newinner = list(inner)
                newinner[k] = v
                self.inner = newinner
                return
            else:
                newinner = v
        else:
            k = positive_index(k, old_len)
            start, stop = k, k + 1
            newinner = [v]
        if start != 0:
            newinner = SequenceConcatenation(inner[:start], newinner)
        if stop < old_len:
            newinner = SequenceConcatenation(newinner, inner[stop:])
        self.inner = newinner

    def __delitem__(self, k):
        if isinstance(k, slice):
            self.__setitem__(k, [])
        else:
            if k < 0:
                k = self.inner.__len__() + k
            self.__setitem__(slice(k, k + 1, 1), [])

    def __add__(self, other):
        return ShellMutableSequence(SequenceConcatenation(self, other))

    def __radd__(self, other):
        return ShellMutableSequence(SequenceConcatenation(other, self))

    def __imul__(self, other):
        return ShellMutableSequence(self * other)

    def extend(self, other):
        self.inner = SequenceConcatenation(self.inner, other)

    def sort(self):
        self.inner = sorted(self.inner)

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
