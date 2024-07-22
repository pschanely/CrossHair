from types import MappingProxyType
from typing import Dict

import pytest

from crosshair import ResumedTracing
from crosshair.core import deep_realize, proxy_for_type
from crosshair.libimpl.builtinslib import SymbolicInt


def test_mappingproxy_repr(space):
    d = proxy_for_type(MappingProxyType[int, int], "d")
    with ResumedTracing():
        assert repr(d).startswith("mappingproxy(")


def test_mappingproxy_deep_realize(space):
    inner = proxy_for_type(Dict[int, int], "inner")
    space.add(inner._inner.__len__().var == 1)
    with ResumedTracing():
        key = next(iter(inner.keys()))
    assert type(key) is SymbolicInt
    orig = MappingProxyType(inner)
    assert type(orig) is MappingProxyType
    copy = deep_realize(orig)
    assert type(copy) is MappingProxyType
    with ResumedTracing():
        val_from_orig = orig[key]
        val_from_copy = copy[key]
    assert type(val_from_orig) is SymbolicInt
    assert type(val_from_copy) is int
