import functools
import inspect

from crosshair.core import proxy_for_type, standalone_statespace
from crosshair.libimpl.builtinslib import LazyIntSymbolicStr
from crosshair.tracers import NoTracing, ResumedTracing


def test_partial(space):
    abc = LazyIntSymbolicStr(list(map(ord, "abc")))
    xyz = LazyIntSymbolicStr(list(map(ord, "xyz")))
    with ResumedTracing():
        joiner = functools.partial(str.join, ",")
        ret = joiner([abc, xyz])
        assert ret == "abc,xyz"


def test_partial_is_interceptable(space):
    x = proxy_for_type(str, "x")
    y = proxy_for_type(str, "y")
    with ResumedTracing():
        joiner = functools.partial(str.startswith, x)
        # Ensure we don't explode
        list(map(joiner, ["foo", y]))


def test_partial_arg_is_inspectable(space):
    with ResumedTracing():
        joiner = functools.partial(str.join, ",")
        assert isinstance(joiner, functools.partial)
        assert inspect.getdoc(joiner.func) == inspect.getdoc(str.join)


def test_reduce():
    with standalone_statespace as space:
        with NoTracing():
            string = LazyIntSymbolicStr(list(map(ord, "12 oofoo 12")))
            tostrip = LazyIntSymbolicStr(list(map(ord, "2")))
        ret = functools.reduce(str.strip, [string, "1", "2"])
        assert ret == " oofoo 1"


_global_state = [42]


@functools.lru_cache()
def whaa(x: int) -> int:
    _global_state[0] += 1
    return _global_state[0]


def test_lru_cache_is_ignored():
    with standalone_statespace as space:
        assert whaa(0) == 43
        assert whaa(1) == 44
        assert whaa(1) == 45
