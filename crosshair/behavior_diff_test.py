import unittest
import sys

from crosshair.behavior_diff import diff_behavior
from crosshair.core import AnalysisOptions
from crosshair.util import set_debug

def foo1(x: int) -> int:
    if x >= 100:
        return 100
    return x

def foo2(x: int) -> int:
    return min(x, 100)

def foo3(x: int) -> int:
    if x > 1000:
        return 1000
    elif x > 100:
        return 100
    else:
        return x


class BehaviorDiffTest(unittest.TestCase):

    def test_diff_behavior_same(self) -> None:
        lines = list(diff_behavior(foo1, foo2, AnalysisOptions(max_iterations=10)))
        # if we're already running under coverage, no coverage will be generated:
        if sys.gettrace() is None:
            self.assertEqual(lines, [
                '(achieved 100% opcode coverage)'])
        else:
            self.assertEqual(lines, [])

    def test_diff_behavior_different(self) -> None:
        lines = list(diff_behavior(foo1, foo3, AnalysisOptions(max_iterations=10)))
        self.assertTrue('- returns: 100' in lines)
        self.assertTrue('+ returns: 1000' in lines)


if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
