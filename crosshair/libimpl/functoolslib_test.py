import functools

from crosshair.core import proxy_for_type, standalone_statespace
from crosshair.libimpl.builtinslib import LazyIntSymbolicStr
from crosshair.tracers import NoTracing


def test_partial():
    with standalone_statespace as space:
        with NoTracing():
            abc = LazyIntSymbolicStr(list(map(ord, "abc")))
            xyz = LazyIntSymbolicStr(list(map(ord, "xyz")))
        joiner = functools.partial(str.join, ",")
        ret = joiner([abc, xyz])
        assert ret == "abc,xyz"


def test_reduce():
    with standalone_statespace as space:
        with NoTracing():
            string = LazyIntSymbolicStr(list(map(ord, "12 oofoo 12")))
            tostrip = LazyIntSymbolicStr(list(map(ord, "2")))
        ret = functools.reduce(str.strip, [string, "1", "2"])
        assert ret == " oofoo 1"
