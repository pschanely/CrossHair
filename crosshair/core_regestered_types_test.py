import re
from dataclasses import dataclass
from typing import (
    Any,
    BinaryIO,
    Callable,
    Final,
    List,
    NamedTuple,
    NoReturn,
    SupportsBytes,
    SupportsComplex,
    Union,
)

import pytest  # type: ignore
import typing_inspect  # type: ignore

from crosshair.core import _SIMPLE_PROXIES, proxy_for_type
from crosshair.dynamic_typing import origin_of
from crosshair.tracers import ResumedTracing
from crosshair.util import (
    CrosshairUnsupported,
    IgnoreAttempt,
    debug,
    name_of_type,
)

"""
Tests that the builtin and standard library types behave like their
counterparts.
"""


possible_args = [
    (),
    (Any,),
    (Any, Any),
    (Any, Any, Any),
]

untested_types = {
    # These types won't check at runtime:
    Union,
    NoReturn,
    Final,
    BinaryIO,
    SupportsBytes,
    SupportsComplex,
    # TODO: make a symbolic NamedTuple that intercepts attribute access:
    NamedTuple,
}


comparisons: List = []
for typ, creator in _SIMPLE_PROXIES.items():
    if typ in untested_types:
        continue
    name = name_of_type(typ)
    comparisons.append(pytest.param(typ, creator, id=name))
    if typing_inspect.is_generic_type(typ) and hasattr(typ, "__getitem__"):
        for nargs in range(1, 3):
            type_args = tuple([Any for _ in range(nargs)])
            try:
                typ_paramed = typ.__getitem__(type_args)
            except TypeError:
                pass
            else:
                comparisons.append(
                    pytest.param(typ_paramed, creator, id=name + "_with_{nargs}_args")
                )


@pytest.mark.parametrize("typ,creator", comparisons)
def test_patch(typ: type, creator: Callable, space):
    debug("Patch test:", typ, creator)
    try:
        proxy = proxy_for_type(typ, "x")
    except CrosshairUnsupported:
        debug("Ignored pass - CrossHair explicitly does not support this type")
        return
    except IgnoreAttempt:
        # Some factories (e.g. re.Match) only produce a value on a subset of
        # paths and abandon the rest; there is nothing to check on those.
        debug("Ignored pass - proxy creation abandoned this path")
        return
    origin = origin_of(typ)
    with ResumedTracing():
        assert isinstance(proxy, origin)
