import collections
from typing import *

from crosshair import register_type

T = TypeVar('T')
class ListBasedDeque:
    def __init__(self, contents: List[T], maxlen: Optional[int]=None):
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

    def index(self, item :T, start: Optional[int]=None, end: Optional[int]=None) -> int:
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

    def rotate(self, n: Optional[int]=1) -> None:
        if n == 0:
            # No rotation
            pass
        if n > 0:
            # Rotate to the right
            self._contents = self._contents[-n:] + self._contents[:-n]
        if n < 0:
            # Rotate to the left
            self._contents = self._contents[n:] + self._contents[:n]

    def maxlen(self) -> int:
        return self._maxlen

def make_registrations():
    # NOTE: defaultdict could be symbolic (but note the default_factory is changable/stateful):
    register_type(collections.defaultdict, lambda p, kt=Any, vt=Any: collections.defaultdict(p(Callable[[], vt]), p(Dict[kt, vt]))) # type: ignore
    register_type(collections.ChainMap, lambda p, kt=Any, vt=Any: collections.ChainMap(*p(Tuple[Dict[kt, vt], ...]))) # type: ignore
    register_type(collections.abc.Mapping, lambda p, t=Any: p(Dict[t]))  # type: ignore
    register_type(collections.abc.MutableMapping, lambda p, t=Any: p(Dict[t]))  # type: ignore
    register_type(collections.OrderedDict, lambda p, kt=Any, vt=Any: collections.OrderedDict(p(Dict[kt, vt]))) # type: ignore
    register_type(collections.Counter, lambda p, t=Any: collections.Counter(p(Dict[t, int]))) # type: ignore
    # TODO: MappingView is missing
    register_type(collections.abc.ItemsView, lambda p, kt=Any, vt=Any: p(Set[Tuple[kt, vt]]))  # type: ignore
    register_type(collections.abc.KeysView, lambda p, t=Any: p(Set[t]))  # type: ignore
    register_type(collections.abc.ValuesView, lambda p, t=Any: p(Set[t]))  # type: ignore

    register_type(collections.abc.Container, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Collection, lambda p, t=Any: p(Tuple[t, ...]))

    register_type(collections.deque, lambda p, t=Any: ListBasedDeque(p(List[t, ...]))) # type: ignore

    register_type(collections.abc.Iterable, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Iterator, lambda p, t=Any: iter(p(Iterable[t])))  # type: ignore

    register_type(collections.abc.MutableSequence, lambda p, t=Any: p(List[t]))  # type: ignore
    register_type(collections.abc.Reversible, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Sequence, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Sized, lambda p, t=Any: p(Tuple[t, ...]))

    register_type(collections.abc.MutableSet, lambda p, t=Any: p(Set[t]))  # type: ignore

    register_type(collections.abc.ByteString, lambda p: bytes(b % 256 for b in p(List[int])))
    register_type(collections.abc.Hashable, lambda p: p(int))
