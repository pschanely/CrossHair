from fractions import Fraction

from crosshair.core import deep_realize, proxy_for_type, realize
from crosshair.tracers import ResumedTracing


def test_fraction_create(space):
    symbolic = proxy_for_type(Fraction, "symbolic")
    space.is_possible(symbolic.numerator == 3)


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
