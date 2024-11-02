import math
from fractions import Fraction

from crosshair.core import deep_realize
from crosshair.core_and_libs import proxy_for_type
from crosshair.statespace import POST_FAIL
from crosshair.test_util import check_states
from crosshair.tracers import ResumedTracing


def test_fraction_realize(space):
    n = proxy_for_type(int, "n")
    d = proxy_for_type(int, "d")
    with ResumedTracing():
        space.add(d != 0)
        deep_realize(Fraction(n, d))


def test_fraction_copy_doesnt_realize(space):
    n = proxy_for_type(int, "n")
    with ResumedTracing():
        space.add(n >= 0)
        f1 = Fraction(n, 1)
        f2 = f1.__copy__()
        assert space.is_possible(f2.numerator == 2)
        assert space.is_possible(f2.numerator == 3)


def test_fraction_can_be_one_half() -> None:
    def f(f: Fraction):
        """post:_"""
        return f != Fraction(1, 2)

    check_states(f, POST_FAIL)


def test_int_from_fraction(space) -> None:
    n = proxy_for_type(int, "n")
    d = proxy_for_type(int, "d")
    with ResumedTracing():
        space.add(d > 0)
        space.add(n == d * 3)
        f = Fraction(n, d)
        assert space.is_possible(d == 3)
        assert not space.is_possible(f.denominator != 1)
        truncated_fraction = int(f)
        assert space.is_possible(truncated_fraction == 3)
        assert not space.is_possible(truncated_fraction != 3)


def test_fraction_ceil_does_not_explode(space) -> None:
    f = proxy_for_type(Fraction, "f")
    with ResumedTracing():
        math.ceil(f)


# TODO: The math module is patched with deep_realize, but many of the usual operators may not work. Test.
