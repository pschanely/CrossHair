import math
import sys
import unittest

from crosshair.core import standalone_statespace
from crosshair.libimpl.builtinslib import SymbolicFloat
from crosshair.tracers import NoTracing
from crosshair.util import set_debug


class MathLibTests(unittest.TestCase):
    def test_isfinite(self):
        with standalone_statespace:
            with NoTracing():
                x = SymbolicFloat("symfloat")
            self.assertTrue(math.isfinite(x))
            self.assertTrue(math.isfinite(2.3))
            self.assertFalse(math.isfinite(float("nan")))


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
