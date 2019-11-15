import collections.abc
import dataclasses
from typing import MutableSequence, Sequence, TypeVar, Union

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

    def __bool__(self):
        return (len(self.contents_) > 0).__bool__()

    def __len__(self):
        '''
        post: _ >= 0
        '''
        return len(self.contents_)

    def __repr__(self):
        return str(dict(self.items()))

# TODO: implement collections.deque on top of SmtList:
# class SimpleDeque(collections.abc.MutableSequence): ...
#   def __init__(self, backing_store: List):

'''
class MutableSequenceOverImmutablePieces(collections.abc.MutableSequence):
    __getitem__
    __setitem__
    __delitem__
    __len__
    insert
'''

def positive_index(idx: int, container_len: int) -> int:
    return idx if idx >= 0 else container_len + idx

def unidirectional_slice(start: int, stop: int, step: int) -> slice:
    return slice(max(0, start), None if stop < 0 else stop, step)

def unidirectional_slice2(start: int, stop: int, step: int) -> slice:
    return slice(None if start < 0 else start, max(0, stop), step)
        

T = TypeVar('T')
@dataclasses.dataclass
class SequenceConcatenation(Sequence[T]):
    _first: Sequence[T]
    _second: Sequence[T]

    def __getitem__(self, i:Union[int, slice]):
        '''
        pre: 0 <= i < len(self) if isinstance(i, int) else True
        pre: i.step != 0 if isinstance(i, slice) else True
        post: _ == (self._first + self._second)[i]
        '''
        first, second = self._first, self._second
        firstlen, secondlen = len(first), len(second)
        totallen = firstlen + secondlen
        if isinstance(i, int):
            i = positive_index(i, totallen)
            return first[i] if i < firstlen else second[i - firstlen]
        else:
            start, stop, step = i.indices(totallen)
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

    def __len__(self):
        return len(self._first) + len(self._second)


#@dataclasses.dataclass(init=False) # type: ignore # (https://github.com/python/mypy/issues/5374)
class SliceView(collections.abc.Sequence):
    seq: Sequence
    rng: Sequence[int]

    def __init__(self, seq, rng=None):
        if rng is None:
            rng = range(len(seq))
        self.seq = seq
        self.rng = rng

    def __getitem__(self, key):
        if type(key) == slice:
            return SliceView(self.seq, self.rng[key])
        else:
            return self.seq[self.rng[key]]

    def __len__(self) -> int:
        return len(self.rng)

    def __iter__(self):
        for i in self.rng:
            yield self.seq[i]


