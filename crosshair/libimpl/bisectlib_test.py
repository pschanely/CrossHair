import bisect
from typing import List

from crosshair.core import proxy_for_type, standalone_statespace
from crosshair.tracers import NoTracing

# NOTE: We don't patch anything in bisect, but it has lgic written in C, so make sure it
# works with symbolics.


def test_bisect_left():
    with standalone_statespace as space:
        with NoTracing():
            lst = proxy_for_type(List[int], "lst")
            space.add(lst.__len__().var == 2)
            space.add(lst[0].var < 10)
            space.add(lst[1].var > 20)
            x = proxy_for_type(int, "x")
            space.add(x.var >= 10)
            space.add(x.var <= 20)
        assert bisect.bisect_left(lst, x) == 1
        assert bisect.bisect_left([0, 100], x) == 1
        assert bisect.bisect_left(lst, 15) == 1
