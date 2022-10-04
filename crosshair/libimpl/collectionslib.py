import collections
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from crosshair import register_type
from crosshair.core import CrossHairValue, realize
from crosshair.tracers import NoTracing, ResumedTracing
from crosshair.util import is_iterable

T = TypeVar("T")


class ListBasedDeque(collections.abc.MutableSequence, CrossHairValue, Generic[T]):
    def __init__(self, contents: List[T], maxlen: Optional[int] = None):
        self._contents = contents
        self._maxlen = maxlen

    def __ch_pytype__(self):
        return collections.deque

    def __ch_realize__(self):
        with ResumedTracing():
            return collections.deque(self._contents, maxlen=realize(self._maxlen))

    def __add__(self, other):
        if not isinstance(other, collections.deque):
            raise TypeError
        ret = self.copy()
        ret.extend(other)
        return ret

    def __eq__(self, other: object) -> bool:
        with NoTracing():
            mycontents = self._contents
            if isinstance(other, ListBasedDeque):
                with ResumedTracing():
                    return mycontents == other._contents
            elif isinstance(other, collections.deque):
                with ResumedTracing():
                    return mycontents == list(other)
            return False

    def __len__(self) -> int:
        return len(self._contents)

    def __mul__(self, count):
        if not isinstance(count, int):
            raise TypeError
        ret = ListBasedDeque([], self._maxlen)
        for _ in range(count):
            ret.extend(self._contents)
        return ret

    def __repr__(self) -> str:
        return repr(realize(self))

    def __getitem__(self, k):
        if isinstance(k, slice):  # slicing isn't supported on deque
            raise TypeError
        return self._contents.__getitem__(k)

    def __setitem__(self, k, v):
        if isinstance(k, slice):  # slicing isn't supported on deque
            raise TypeError
        return self._contents.__setitem__(k, v)

    def __delitem__(self, k):
        if isinstance(k, slice):  # slicing isn't supported on deque
            raise TypeError
        return self._contents.__delitem__(k)

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

    def copy(self):
        return ListBasedDeque(self._contents[:], self._maxlen)

    def count(self, item: T) -> int:
        c = 0
        for i in self._contents:
            if i == item:
                c += 1
        return c

    def extend(self, items: Iterable[T]) -> None:
        if not is_iterable(items):
            raise TypeError
        self._contents += list(items)

    def extendleft(self, items: Iterable[T]) -> None:
        if not is_iterable(items):
            raise TypeError
        prefix = list(items)
        prefix.reverse()
        self._contents = prefix + self._contents

    def index(self, item: T, *bounds) -> int:
        return self._contents.index(item, *bounds)

    def insert(self, index: int, item: T) -> None:
        self._contents.insert(index, item)

    def pop(self) -> T:  # type: ignore
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

    def rotate(self, n: int = 1) -> None:
        if not self._contents or n % len(self._contents) == 0:
            return
        self._contents = (
            self._contents[-n % len(self._contents) :]
            + self._contents[: -n % len(self._contents)]
        )

    def maxlen(self) -> Optional[int]:
        return self._maxlen


class PureDefaultDict(collections.abc.MutableMapping, CrossHairValue):
    def __init__(self, factory, internal):
        self.default_factory = factory
        self._internal = internal

    def __ch_pytype__(self):
        return collections.defaultdict

    def __ch_realize__(self):
        with ResumedTracing():
            return collections.defaultdict(self.default_factory, self._internal)

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


def make_registrations():
    register_type(collections.defaultdict, lambda p, kt=Any, vt=Any: PureDefaultDict(p(Optional[Callable[[], vt]], "_initalizer"), p(Dict[kt, vt])))  # type: ignore
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

    register_type(collections.abc.ByteString, lambda p: p(bytes))
    register_type(collections.abc.Hashable, lambda p: p(int))
