import itertools
from typing import List

from crosshair.core import proxy_for_type, standalone_statespace
from crosshair.statespace import POST_FAIL, MessageType
from crosshair.test_util import check_states
from crosshair.tracers import NoTracing


def test_keyfn_is_intercepted():
    with standalone_statespace as space:
        with NoTracing():
            two = proxy_for_type(int, "two")
            space.add(two.var == 2)
        ret = list((k, tuple(v)) for k, v in itertools.groupby([1, two, 3.0], type))
        assert ret == [(int, (1, 2)), (float, (3.0,))]


def test_accumulate():
    def f(x: int) -> List[int]:
        """
        post: __return__[-1] != 10  # (false when x == 6)
        """
        return list(itertools.accumulate([1, x, 3]))

    check_states(f, POST_FAIL)
