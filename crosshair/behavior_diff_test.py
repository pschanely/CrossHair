import dis
import unittest
import sys
from typing import List

from crosshair.behavior_diff import BehaviorDiff
from crosshair.behavior_diff import diff_behavior
from crosshair.behavior_diff import diff_opcodes
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

    def test_diff_opcodes(self) -> None:
        def foo(x: int) -> int:
            return x * 3 + 1
        def bar(x: int) -> int:
            return (x * 2 + 1) * 10
        foo_i = [(i.offset, i.opname, i.argrepr) for i in dis.get_instructions(foo)]
        bar_i = [(i.offset, i.opname, i.argrepr) for i in dis.get_instructions(bar)]
        self.assertEqual(foo_i, [(0, 'LOAD_FAST', 'x'),
                                 (2, 'LOAD_CONST', '3'),
                                 (4, 'BINARY_MULTIPLY', ''),
                                 (6, 'LOAD_CONST', '1'),
                                 (8, 'BINARY_ADD', ''),
                                 (10, 'RETURN_VALUE', '')])
        self.assertEqual(bar_i, [(0, 'LOAD_FAST', 'x'),
                                 (2, 'LOAD_CONST', '2'),
                                 (4, 'BINARY_MULTIPLY', ''),
                                 (6, 'LOAD_CONST', '1'),
                                 (8, 'BINARY_ADD', ''),
                                 (10, 'LOAD_CONST', '10'),
                                 (12, 'BINARY_MULTIPLY', ''),
                                 (14, 'RETURN_VALUE', '')])
        self.assertEqual(diff_opcodes(foo, bar), ({2}, {2, 10, 12}))

    def test_diff_behavior_same(self) -> None:
        diffs = diff_behavior(foo1, foo2, AnalysisOptions(max_iterations=10))
        self.assertEqual(diffs, [])

    def test_diff_behavior_different(self) -> None:
        diffs = diff_behavior(foo1, foo3, AnalysisOptions(max_iterations=10))
        self.assertEqual(len(diffs), 1)
        diff = diffs[0]
        assert isinstance(diff, BehaviorDiff)
        self.assertGreater(int(diff.args['x']), 1000)
        self.assertEqual(diff.result1.return_repr, '100')
        self.assertEqual(diff.result2.return_repr, '1000')

    def test_diff_behavior_arg_mutation(self) -> None:
        def cut_out_item1(a: List[int], i: int):
            a[i:i+1] = []
        def cut_out_item2(a: List[int], i: int):
            a[:] = a[:i] + a[i+1:]
        # TODO: this takes longer than it should:
        opts = AnalysisOptions(max_iterations=100,
                               per_path_timeout=10,
                               per_condition_timeout=10)
        diffs = diff_behavior(cut_out_item1, cut_out_item2, opts)
        assert not isinstance(diffs, str)
        self.assertEqual(len(diffs), 1)
        diff = diffs[0]
        self.assertGreater(len(diff.args['a']), 1)
        self.assertEqual(diff.args['i'], '-1')

if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
