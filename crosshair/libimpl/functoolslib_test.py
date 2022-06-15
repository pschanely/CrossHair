import functools

from crosshair.core import proxy_for_type, standalone_statespace
from crosshair.libimpl.builtinslib import LazyIntSymbolicStr
from crosshair.tracers import NoTracing


def test_reduce():
    with standalone_statespace as space:
        with NoTracing():
            string = LazyIntSymbolicStr(list(map(ord, "12 oofoo 12")))
            tostrip = LazyIntSymbolicStr(list(map(ord, "2")))
        ret = functools.reduce(str.strip, [string, "1", "2"])
        assert ret == " oofoo 1"
