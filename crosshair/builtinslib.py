import collections
import contextlib
import io
import re
import typing
from typing import *

from crosshair.core import register_type
from crosshair.util import IgnoreAttempt

def make_raiser(exc, *a) -> Callable:
    def do_raise(*ra, **rkw) -> NoReturn:
        raise exc(*a)
    return do_raise

def make_builtin_registrations():

    # Note that the following builtin types are handled natively by CrossHair,
    # and so aren't listed here:
    #   List
    #   Set
    #   FrozenSet
    #   Dict
    #   Optional
    #   Callable
    #   ClassVar
    
    register_type(complex, lambda p: complex(p(float), p(float)))
    register_type(type(None), lambda p: None)
    register_type(slice, lambda p: slice(p(Optional[int]), p(Optional[int]), p(Optional[int])))
    register_type(NoReturn, make_raiser(IgnoreAttempt, 'Attempted to short circuit a NoReturn function')) # type: ignore
    
    # AsyncContextManager, lambda p: p(contextlib.AbstractAsyncContextManager),
    # AsyncGenerator: ,
    # AsyncIterable,
    # AsyncIterator,
    # Awaitable,
    # Coroutine: (handled via typeshed)
    # Generator: (handled via typeshed)
    
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
    register_type(NamedTuple, lambda p, *t: p(Tuple.__getitem__(tuple(t))))
    
    register_type(collections.abc.MutableSet, lambda p, t=Any: p(Set[t]))  # type: ignore
    
    register_type(re.Pattern, lambda p, t=None: p(re.compile))  # type: ignore
    register_type(re.Match, lambda p, t=None: p(re.match))  # type: ignore
    
    # Text: (elsewhere - identical to str)
    register_type(collections.abc.ByteString, lambda p: bytes(b % 256 for b in p(List[int])))
    register_type(bytes, lambda p: p(ByteString))
    register_type(bytearray, lambda p: p(ByteString))
    register_type(memoryview, lambda p: p(ByteString))
    # AnyStr,  (it's a type var)
    
    register_type(typing.BinaryIO, lambda p: io.BytesIO(p(ByteString)))
    # TODO: handle Any/AnyStr with a custom class that accepts str/bytes interchangably?:
    register_type(typing.IO, lambda p, t=Any: p(BinaryIO) if t == 'bytes' else p(TextIO))
    # TODO: StringIO (and BytesIO) won't accept SmtStr writes.
    # Consider clean symbolic implementations of these.
    register_type(typing.TextIO, lambda p: io.StringIO(str(p(str))))
    
    register_type(collections.abc.Hashable, lambda p: p(int))
    register_type(SupportsAbs, lambda p: p(int))
    register_type(SupportsFloat, lambda p: p(float))
    register_type(SupportsInt, lambda p: p(int))
    register_type(SupportsRound, lambda p: p(float))
    register_type(SupportsBytes, lambda p: p(ByteString))
    register_type(SupportsComplex, lambda p: p(complex))

