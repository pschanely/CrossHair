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

def make_registrations():

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
    
    register_type(NamedTuple, lambda p, *t: p(Tuple.__getitem__(tuple(t))))
    
    register_type(re.Pattern, lambda p, t=None: p(re.compile))  # type: ignore
    register_type(re.Match, lambda p, t=None: p(re.match))  # type: ignore
    
    # Text: (elsewhere - identical to str)
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
    
    register_type(SupportsAbs, lambda p: p(int))
    register_type(SupportsFloat, lambda p: p(float))
    register_type(SupportsInt, lambda p: p(int))
    register_type(SupportsRound, lambda p: p(float))
    register_type(SupportsBytes, lambda p: p(ByteString))
    register_type(SupportsComplex, lambda p: p(complex))

