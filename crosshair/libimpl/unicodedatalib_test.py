import unicodedata
from typing import *

import pytest

from crosshair.core_and_libs import NoTracing, proxy_for_type, standalone_statespace
from crosshair.libimpl.builtinslib import LazyIntSymbolicStr
from crosshair.tracers import ResumedTracing


def test_numeric():
    with standalone_statespace as space:
        with NoTracing():
            fourstr = LazyIntSymbolicStr(list(map(ord, "4")))
            halfstr = LazyIntSymbolicStr(list(map(ord, "\u00bd")))  # (1/2 character)
        four = unicodedata.numeric(fourstr)
        half = unicodedata.numeric(halfstr)
        assert type(four) is float
        assert four == 4.0
        assert half == 0.5


def test_decimal():
    with standalone_statespace as space:
        with NoTracing():
            thai4 = LazyIntSymbolicStr(list(map(ord, "\u0e54")))  # (Thai numerial 4)
            super4 = LazyIntSymbolicStr(list(map(ord, "\u2074")))  # (superscript 4)
        four = unicodedata.decimal(thai4)
        assert type(four) is int
        assert four == 4
        with pytest.raises(ValueError):
            unicodedata.decimal(super4)


def test_digit():
    with standalone_statespace as space:
        with NoTracing():
            thai4 = LazyIntSymbolicStr(list(map(ord, "\u0e54")))  # (Thai numerial 4)
            super4 = LazyIntSymbolicStr(list(map(ord, "\u2074")))  # (superscript 4)
        four = unicodedata.digit(thai4)
        assert type(four) is int
        assert four == 4
        assert unicodedata.digit(super4) == 4


@pytest.mark.parametrize(
    "fn", [unicodedata.decimal, unicodedata.digit, unicodedata.numeric]
)
def test_does_not_realize(space, fn):
    s = proxy_for_type(str, "s")
    with ResumedTracing():
        space.add(len(s) == 1)
        space.add(ord(s) >= ord("0"))
        space.add(ord(s) <= ord("9"))
        fn(s)
        assert space.is_possible(s == "1")
        assert space.is_possible(s == "2")
