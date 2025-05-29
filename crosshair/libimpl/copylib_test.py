import copy
import datetime

from crosshair.core import proxy_for_type
from crosshair.statespace import StateSpace
from crosshair.tracers import ResumedTracing
from crosshair.util import debug


def test_date_copy(space: StateSpace) -> None:
    concrete_date = datetime.date(2000, 2, 3)
    symbolic_date = proxy_for_type(datetime.date, "d")
    with ResumedTracing():
        copied_symbolic = copy.deepcopy(symbolic_date)
        copied_concrete = copy.deepcopy(concrete_date)
        assert not space.is_possible(copied_concrete != concrete_date)
        assert not space.is_possible(copied_concrete != datetime.date(2000, 2, 3))
        assert not space.is_possible(copied_symbolic != symbolic_date)
