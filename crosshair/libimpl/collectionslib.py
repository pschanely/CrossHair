import collections
from typing import *

from crosshair import register_type


def make_registrations():
    # NOTE: could be symbolic (but note the default_factory is changable/stateful):
    register_type(collections.defaultdict, lambda p, kt=Any, vt=Any: collections.defaultdict(p(Callable[[], vt]), p(Dict[kt, vt]))) # type: ignore
    register_type(collections.ChainMap, lambda p, kt=Any, vt=Any: collections.ChainMap(*p(Tuple[Dict[kt, vt], ...]))) # type: ignore
    register_type(collections.abc.Mapping, lambda p, t=Any: p(Dict[t]))  # type: ignore
    register_type(collections.abc.MutableMapping, lambda p, t=Any: p(Dict[t]))  # type: ignore
    register_type(collections.OrderedDict, lambda p, kt=Any, vt=Any: collections.OrderedDict(p(Dict[kt, vt]))) # type: ignore
    register_type(collections.Counter, lambda p, t=Any: collections.Counter(p(Dict[t, int]))) # type: ignore
    # MappingView: (as instantiated origin)
    register_type(collections.abc.ItemsView, lambda p, kt=Any, vt=Any: p(Set[Tuple[kt, vt]]))  # type: ignore
    register_type(collections.abc.KeysView, lambda p, t=Any: p(Set[t]))  # type: ignore
    register_type(collections.abc.ValuesView, lambda p, t=Any: p(Set[t]))  # type: ignore

    register_type(collections.abc.Container, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Collection, lambda p, t=Any: p(Tuple[t, ...]))
    # TODO: a custom impl in simplestructs.py
    register_type(collections.deque, lambda p, t=Any: collections.deque(p(Tuple[t, ...])))
    register_type(collections.abc.Iterable, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Iterator, lambda p, t=Any: iter(p(Iterable[t])))  # type: ignore

    register_type(collections.abc.MutableSequence, lambda p, t=Any: p(List[t]))  # type: ignore
    register_type(collections.abc.Reversible, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Sequence, lambda p, t=Any: p(Tuple[t, ...]))
    register_type(collections.abc.Sized, lambda p, t=Any: p(Tuple[t, ...]))

    register_type(collections.abc.MutableSet, lambda p, t=Any: p(Set[t]))  # type: ignore

    register_type(collections.abc.ByteString, lambda p: bytes(b % 256 for b in p(List[int])))
    register_type(collections.abc.Hashable, lambda p: p(int))
