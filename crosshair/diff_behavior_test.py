import sys
import unittest
from typing import Callable, List, Optional

from crosshair.diff_behavior import BehaviorDiff, diff_behavior
from crosshair.fnutil import FunctionInfo, walk_qualname
from crosshair.options import DEFAULT_OPTIONS
from crosshair.util import IgnoreAttempt, debug, set_debug


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
    def test_diff_method(self):
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

    def test_diff_staticmethod(self):
        diffs = diff_behavior(
            walk_qualname(Base, "staticfoo"),
            foo2,
            DEFAULT_OPTIONS.overlay(max_iterations=10),
        )
        self.assertEqual(diffs, [])

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

        # TODO: this takes longer than I'd like:
        opts = DEFAULT_OPTIONS.overlay(max_iterations=40)
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
            DEFAULT_OPTIONS.overlay(max_iterations=20),
        )
        debug("diffs=", diffs)
        assert not isinstance(diffs, str)
        return_vals = set((d.result1.return_repr, d.result2.return_repr) for d in diffs)
        self.assertEqual(return_vals, {("False", "None"), ("False", "True")})


def test_diff_behavior_lambda() -> None:
    def f(a: Optional[Callable[[int], int]]):
        if a:
            return a(2) + 4
        else:
            return "hello"

    diffs = diff_behavior(
        FunctionInfo.from_fn(f),
        FunctionInfo.from_fn(f),
        DEFAULT_OPTIONS,
    )
    assert diffs == []


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
