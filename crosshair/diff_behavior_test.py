import dis
import unittest
import sys
from typing import List
from typing import Optional

from crosshair.diff_behavior import BehaviorDiff
from crosshair.diff_behavior import diff_behavior
from crosshair.diff_behavior import diff_opcodes
from crosshair.fnutil import walk_qualname
from crosshair.fnutil import FunctionInfo
from crosshair.options import AnalysisOptions
from crosshair.options import DEFAULT_OPTIONS
from crosshair.util import debug
from crosshair.util import set_debug


def _foo1(x: int) -> int:
    if x >= 100:
        return 100
    return x


foo1 = FunctionInfo.from_fn(_foo1)


def _foo2(x: int) -> int:
    return min(x, 100)


foo2 = FunctionInfo.from_fn(_foo2)


def _foo3(x: int) -> int:
    if x > 1000:
        return 1000
    elif x > 100:
        return 100
    else:
        return x


foo3 = FunctionInfo.from_fn(_foo3)


class Base:
    def foo(self):
        return 10

    @staticmethod
    def staticfoo(x: int) -> int:
        return min(x, 100)


class Derived(Base):
    def foo(self):
        return 11


class BehaviorDiffTest(unittest.TestCase):
    def test_diff_method(self) -> None:
        diffs = diff_behavior(
            walk_qualname(Base, "foo"),
            walk_qualname(Derived, "foo"),
            DEFAULT_OPTIONS.overlay(max_iterations=10),
        )
        assert isinstance(diffs, list)
        self.assertEqual(
            [(d.result1.return_repr, d.result2.return_repr) for d in diffs],
            [("10", "11")],
        )

    def test_diff_staticmethod(self) -> None:
        diffs = diff_behavior(
            walk_qualname(Base, "staticfoo"),
            foo2,
            DEFAULT_OPTIONS.overlay(max_iterations=10),
        )
        self.assertEqual(diffs, [])

    def test_diff_opcodes(self) -> None:
        def foo(x: int) -> int:
            return x * 3 + 1

        def bar(x: int) -> int:
            return (x * 2 + 1) * 10

        foo_i = [(i.offset, i.opname, i.argrepr) for i in dis.get_instructions(foo)]
        bar_i = [(i.offset, i.opname, i.argrepr) for i in dis.get_instructions(bar)]
        self.assertEqual(
            foo_i,
            [
                (0, "LOAD_FAST", "x"),
                (2, "LOAD_CONST", "3"),
                (4, "BINARY_MULTIPLY", ""),
                (6, "LOAD_CONST", "1"),
                (8, "BINARY_ADD", ""),
                (10, "RETURN_VALUE", ""),
            ],
        )
        self.assertEqual(
            bar_i,
            [
                (0, "LOAD_FAST", "x"),
                (2, "LOAD_CONST", "2"),
                (4, "BINARY_MULTIPLY", ""),
                (6, "LOAD_CONST", "1"),
                (8, "BINARY_ADD", ""),
                (10, "LOAD_CONST", "10"),
                (12, "BINARY_MULTIPLY", ""),
                (14, "RETURN_VALUE", ""),
            ],
        )
        self.assertEqual(diff_opcodes(foo, bar), ({2}, {2, 10, 12}))

    def test_diff_behavior_same(self) -> None:
        diffs = diff_behavior(foo1, foo2, DEFAULT_OPTIONS.overlay(max_iterations=10))
        self.assertEqual(diffs, [])

    def test_diff_behavior_different(self) -> None:
        diffs = diff_behavior(foo1, foo3, DEFAULT_OPTIONS.overlay(max_iterations=10))
        self.assertEqual(len(diffs), 1)
        diff = diffs[0]
        assert isinstance(diff, BehaviorDiff)
        self.assertGreater(int(diff.args["x"]), 1000)
        self.assertEqual(diff.result1.return_repr, "100")
        self.assertEqual(diff.result2.return_repr, "1000")

    def test_diff_behavior_mutation(self) -> None:
        def cut_out_item1(a: List[int], i: int):
            a[i : i + 1] = []

        def cut_out_item2(a: List[int], i: int):
            a[:] = a[:i] + a[i + 1 :]

        # TODO: this takes longer than I'd like (few iterations though):
        opts = DEFAULT_OPTIONS.overlay(
            max_iterations=20, per_path_timeout=10, per_condition_timeout=10
        )
        diffs = diff_behavior(
            FunctionInfo.from_fn(cut_out_item1),
            FunctionInfo.from_fn(cut_out_item2),
            opts,
        )
        assert not isinstance(diffs, str)
        self.assertEqual(len(diffs), 1)
        diff = diffs[0]
        self.assertGreater(len(diff.args["a"]), 1)
        self.assertEqual(diff.args["i"], "-1")

    def test_example_coverage(self) -> None:
        # Try to get examples that highlist the differences in the code.
        # Here, we add more conditions for the `return True` path and
        # another case where we used to just `return False`.
        def isack1(s: str) -> bool:
            if s in ("y", "yes"):
                return True
            return False

        def isack2(s: str) -> Optional[bool]:
            if s in ("y", "yes", "Y", "YES"):
                return True
            if s in ("n", "no", "N", "NO"):
                return False
            return None

        diffs = diff_behavior(
            FunctionInfo.from_fn(isack1),
            FunctionInfo.from_fn(isack2),
            DEFAULT_OPTIONS.overlay(max_iterations=20, per_condition_timeout=5),
        )
        debug("diffs=", diffs)
        assert not isinstance(diffs, str)
        return_vals = set((d.result1.return_repr, d.result2.return_repr) for d in diffs)
        self.assertEqual(return_vals, {("False", "None"), ("False", "True")})


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
