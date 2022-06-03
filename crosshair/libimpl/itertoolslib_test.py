import itertools

from crosshair.core import proxy_for_type, standalone_statespace
from crosshair.tracers import NoTracing


def test_keyfn_is_intercepted():
    with standalone_statespace as space:
        with NoTracing():
            two = proxy_for_type(int, "two")
            space.add(two.var == 2)
        ret = list((k, tuple(v)) for k, v in itertools.groupby([1, two, 3.0], type))
        assert ret == [(int, (1, 2)), (float, (3.0,))]
