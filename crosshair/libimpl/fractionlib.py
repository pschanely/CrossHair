from fractions import Fraction

from crosshair.core import SymbolicFactory, register_type
from crosshair.statespace import force_true
from crosshair.tracers import ResumedTracing


def _make_fraction(factory: SymbolicFactory):
    n, d = factory(int, "_numerator"), factory(int, "_denominator")
    with ResumedTracing():
        force_true(d > 0)
        return Fraction(n, d)


def make_registrations() -> None:
    register_type(Fraction, _make_fraction)
