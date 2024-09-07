import math
import sys
import unittest

from crosshair.core import proxy_for_type, standalone_statespace
from crosshair.libimpl.builtinslib import RealBasedSymbolicFloat
from crosshair.tracers import NoTracing
from crosshair.util import set_debug


def test_isfinite():
    with standalone_statespace:
        with NoTracing():
            x = RealBasedSymbolicFloat("symfloat")
        assert math.isfinite(x)
        assert math.isfinite(2.3)
        assert not math.isfinite(float("nan"))


def test_isinf():
    with standalone_statespace:
        with NoTracing():
            x = RealBasedSymbolicFloat("symfloat")
        assert not math.isinf(x)
        assert not math.isinf(float("nan"))
        assert math.isinf(float("-inf"))


def test_log():
    with standalone_statespace as space:
        with NoTracing():
            i = proxy_for_type(int, "i")
            f = proxy_for_type(float, "f")
        space.add(i > 0)
        space.add(f > 0)
        math.log(i)
        math.log(f)


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
