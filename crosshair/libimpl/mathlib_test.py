import math
import sys
import unittest
from typing import *

from crosshair.core_and_libs import *
from crosshair.core import Patched
from crosshair.test_util import check_ok
from crosshair.test_util import check_exec_err
from crosshair.test_util import check_post_err
from crosshair.test_util import check_fail
from crosshair.test_util import check_unknown
from crosshair.test_util import check_messages
from crosshair.util import set_debug

from crosshair.libimpl.builtinslib import SymbolicFloat
from crosshair.statespace import SimpleStateSpace
from crosshair.statespace import StateSpaceContext
from crosshair.tracers import NoTracing


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
