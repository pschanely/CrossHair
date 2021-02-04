import collections
from functools import total_ordering
from typing import *

from crosshair import register_type
from crosshair.abcstring import AbcString

T = TypeVar("T")


class ListBasedDeque:
    def __init__(self, contents: List[T], maxlen: Optional[int] = None):
        self._contents = contents
        self._maxlen = maxlen

    def __len__(self) -> int:
        return len(self._contents)

    def _has_room(self) -> bool:
        maxlen = self._maxlen
        return maxlen is None or len(self._contents) < maxlen

    def appendleft(self, item: T) -> None:
        if self._has_room():
            self._contents = [item] + self._contents
        else:
            del self._contents[-1]
            self._contents = [item] + self._contents

    def append(self, item: T) -> None:
        if self._has_room():
            self._contents = self._contents + [item]
        else:
            del self._contents[0]
            self._contents = self._contents + [item]

    def clear(self) -> None:
        self._contents = []

    def count(self, item: T) -> int:
        c = 0
        for i in self._contents:
            if i == item:
                c += 1
        return c

    def index(
        self, item: T, start: Optional[int] = None, end: Optional[int] = None
    ) -> int:
        if start is not None and end is None:
            return self._contents.index(item, start)
        if start is not None and end is not None:
            return self._contents.index(item, start, end)
        return self._contents.index(item)

    def insert(self, index: int, item: T) -> None:
        self._contents.insert(index, item)

    def pop(self) -> T:
        x = self._contents[-1]
        del self._contents[-1]
        return x

    def popleft(self) -> T:
        x = self._contents[0]
        del self._contents[0]
        return x

    def remove(self, item: T) -> None:
        self._contents.remove(item)

    def reverse(self) -> None:
        self._contents.reverse()

    def rotate(self, n: Optional[int] = 1) -> None:
        if not self._contents or n % len(self._contents) == 0:
            pass
        self._contents = (
            self._contents[-n % len(self._contents) :]
            + self._contents[: -n % len(self._contents)]
        )

    def maxlen(self) -> int:
        return self._maxlen


class PureDefaultDict(collections.abc.MutableMapping):
    def __init__(self, factory, internal):
        self.default_factory = factory
        self._internal = internal

    def __getitem__(self, k):
        try:
            return self._internal.__getitem__(k)
        except KeyError:
            return self.__missing__(k)

    def __setitem__(self, k, v):
        return self._internal.__setitem__(k, v)

    def __delitem__(self, k):
        return self._internal.__delitem__(k)

    def __iter__(self):
        return self._internal.__iter__()

    def __len__(self):
        return self._internal.__len__()

    def __repr__(self):
        return "defaultdict({!r}, {!r})".format(self.default_factory, self._internal)

    def __missing__(self, k):
        if self.default_factory is None:
            raise KeyError(k)
        value = self.default_factory()
        self._internal[k] = value
        return value


# TODO: We use AbcString as a superclass here, but it probably isn't fully
# appropriate for bytes. Investigate.
@total_ordering
class ListBasedByteString(collections.abc.ByteString, AbcString):
    def __init__(self, l):
        self.l = l

    data = property(lambda s: bytes(s.l))

    def __len__(self):
        return len(self.l)

    def __getitem__(self, *a, **kw):
        return self.l.__getitem__(*a, **kw)

    def __repr__(self):
        return repr(bytes(self))

    def __iter__(self):
        return self.l.__iter__()

    def __eq__(self, other) -> bool:
        if isinstance(other, collections.abc.ByteString):
            return self.l == list(other)
        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, collections.abc.ByteString):
            return self.l < list(other)
        else:
            raise TypeError

    def __copy__(self):
        return ListBasedByteString(self.l)

    def __deepcopy__(self, memo):
        return ListBasedByteString(self.l)

    def decode(self, encoding="utf-8", errors="strict"):
        self.data.decode(encoding=encoding, errors=errors)


def make_byte_string(p: Callable[[type], object]):
    # alternatively, we might realize the byte length and then we can constrain
    # the values from the begining. Using a quantifier is also possible.
    values = ListBasedByteString(p(List[int]))
    p.space.defer_assumption(
        "bytes are valid bytes", lambda: all(0 <= v < 256 for v in values)
    )
    return values


def make_registrations():
    register_type(collections.defaultdict, lambda p, kt=Any, vt=Any: PureDefaultDict(p(Optional[Callable[[], vt]]), p(Dict[kt, vt])))  # type: ignore
    register_type(collections.ChainMap, lambda p, kt=Any, vt=Any: collections.ChainMap(*p(Tuple[Dict[kt, vt], ...])))  # type: ignore
    register_type(collections.abc.Mapping, lambda p, kt=Any, vt=Any: p(Dict[kt, vt]))  # type: ignore
    register_type(collections.abc.MutableMapping, lambda p, kt=Any, vt=Any: p(Dict[kt, vt]))  # type: ignore
    register_type(collections.OrderedDict, lambda p, kt=Any, vt=Any: collections.OrderedDict(p(Dict[kt, vt])))  # type: ignore
    register_type(collections.Counter, lambda p, t=Any: collections.Counter(p(Dict[t, int])))  # type: ignore
    # TODO: MappingView is missing
    register_type(collections.abc.ItemsView, lambda p, kt=Any, vt=Any: p(Set[Tuple[kt, vt]]))  # type: ignore
    register_type(collections.abc.KeysView, lambda p, t=Any: p(Set[t]))  # type: ignore
    register_type(collections.abc.ValuesView, lambda p, t=Any: p(List[t]))  # type: ignore

    register_type(collections.abc.Container, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Collection, lambda p, t=Any: p(Tuple[t, ...]))

    register_type(collections.deque, lambda p, t=Any: ListBasedDeque(p(List[t])))  # type: ignore

    register_type(collections.abc.Iterable, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Iterator, lambda p, t=Any: iter(p(Iterable[t])))  # type: ignore

    register_type(collections.abc.MutableSequence, lambda p, t=Any: p(List[t]))  # type: ignore
    register_type(collections.abc.Reversible, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Sequence, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Sized, lambda p, t=Any: p(Tuple[t, ...]))

    register_type(collections.abc.MutableSet, lambda p, t=Any: p(Set[t]))  # type: ignore

    register_type(collections.abc.ByteString, make_byte_string)
    register_type(collections.abc.Hashable, lambda p: p(int))
