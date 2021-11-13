import unittest

import pytest

from crosshair.simplestructs import *
from crosshair.test_util import summarize_execution


class SimpleStructTests(unittest.TestCase):
    def test_ShellMutableMap(self) -> None:
        m = ShellMutableMap({4: 4, 5: 5, 6: 6})
        m[3] = 3
        m[4] = None
        del m[5]
        self.assertEqual(len(m), 3)
        self.assertEqual(list(m), [6, 3, 4])
        self.assertEqual(m[4], None)
        with self.assertRaises(KeyError):
            m[5]
        self.assertEqual(m, {3: 3, 4: None, 6: 6})

    def test_ShellMutableMap_popitem_ordering(self) -> None:
        self.assertEqual({"c": "d", "a": "b"}.popitem(), ("a", "b"))
        self.assertEqual(SimpleDict([("c", "d"), ("a", "b")]).popitem(), ("a", "b"))

    def test_ShellMutableMap_poo(self) -> None:
        m = ShellMutableMap({2: 0})
        self.assertEqual(0, m.setdefault(2.0, {True: "0"}))

    def test_SequenceConcatenation_comparison(self) -> None:
        compound = SequenceConcatenation((11, 22), (33, 44))
        self.assertEqual(compound, (11, 22, 33, 44))
        self.assertLess(compound, (22, 33, 44))
        self.assertGreaterEqual(compound, (11, 22, 33))

    def test_SliceView_comparison(self) -> None:
        sliced = SliceView((00, 11, 22, 33, 44, 55), 1, 5)
        self.assertEqual(sliced, (11, 22, 33, 44))
        self.assertLess(sliced, (22, 33, 44))
        self.assertGreaterEqual(sliced, (11, 22, 33))

    def test_slice_view(self) -> None:
        nums = ["0", "1", "2", "3", "4", "5"]
        ctr = 0
        for start in [0, 1, 2, 3, 4, 5, 6]:
            for stop in range(start, 7):
                view = SliceView(nums, start, stop)
                concrete = nums[start:stop]
                self.assertEqual(
                    list(view), concrete, f"{ctr}: {start}:{stop}: {view} vs {concrete}"
                )
                if stop - start > 0:
                    self.assertEqual(view[0], nums[start])
                ctr += 1

    def test_LazySetCombination_xor(self) -> None:
        a = {2, 4, 6}
        b = {4, 5, 6, 7}
        s = LazySetCombination(operator.xor, a, b)
        self.assertEqual(s, {2, 5, 7})
        self.assertTrue(4 not in s)
        self.assertTrue(5 in s)

    def test_ShellMutableSequence_slice_assignment(self) -> None:
        l = ["0", "1", "2", "3"]
        shell = ShellMutableSequence(l)
        self.assertEqual(shell, shell)
        self.assertEqual(shell, l)
        shell[1:3] = ["1", "1.5", "2"]
        self.assertEqual(shell, ["0", "1", "1.5", "2", "3"])
        self.assertEqual(shell, shell)

    def test_ShellMutableSequence_assignment_negative_index(self) -> None:
        l = ShellMutableSequence(["a", "a"])
        l[-1] = "b"
        self.assertEqual(l, ["a", "b"])

    def test_ShellMutableSequence_assignment_bad_index(self) -> None:
        with self.assertRaises(TypeError):
            ShellMutableSequence([])["whoa"] = 3

    def test_ShellMutableSequence_assignment_out_of_bounds(self) -> None:
        with self.assertRaises(IndexError):
            ShellMutableSequence(["b", "c"])[-3] = "a"

    def test_ShellMutableSequence_sort_invalid_args(self) -> None:
        s = ShellMutableSequence(SequenceConcatenation([], []))
        with self.assertRaises(TypeError):
            s.sort(reverse="badvalue")

    def test_SequenceConcatenation_operators(self) -> None:
        s = SequenceConcatenation([4], [6]) + [8]
        self.assertEqual(s, [4, 6, 8])
        self.assertTrue(type(s) == SequenceConcatenation)
        s = [2] + SequenceConcatenation([4], [6])
        self.assertEqual(s, [2, 4, 6])
        self.assertTrue(type(s) == SequenceConcatenation)

    def test_SingletonSet_arg_types(self) -> None:
        s = SingletonSet(42)
        with self.assertRaises(TypeError):
            s | [3, 5]
        with self.assertRaises(TypeError):
            [3, 5] ^ s  # type: ignore
        with self.assertRaises(TypeError):
            s - [3, 5]

    def test_ShellMutableSet_members(self) -> None:
        shell = ShellMutableSet([6, 4, 8, 2])
        self.assertEqual(shell, shell)
        self.assertEqual(shell, {2, 4, 6, 8})
        shell.pop()
        self.assertEqual(list(shell), [4, 8, 2])
        shell.add(0)
        self.assertEqual(list(shell), [4, 8, 2, 0])
        shell.clear()
        self.assertEqual(list(shell), [])

    def test_ShellMutableSet_comparisons(self) -> None:
        # Truths
        self.assertTrue(ShellMutableSet([2, 4]) >= {2, 4})
        self.assertTrue(ShellMutableSet([2, 4]) < {2, 3, 4})
        self.assertTrue(ShellMutableSet([2, 4]) == {4, 2})
        self.assertTrue(ShellMutableSet([2, 4]) != {})
        self.assertTrue(ShellMutableSet([2, 4]).isdisjoint({1, 3, 5}))
        # Falsehoods
        self.assertFalse(ShellMutableSet([2, 4]) > {2, 4})
        self.assertFalse(ShellMutableSet([2, 4]) <= {4, 5})
        self.assertFalse(ShellMutableSet([2, 4]).isdisjoint({1, 4, 5}))

    def test_ShellMutableSet_operators(self) -> None:
        shell = ShellMutableSet([2, 4])
        shell = shell | {6}
        self.assertEqual(len(shell), 3)
        self.assertEqual(list(shell), [2, 4, 6])
        shell = shell & {2, 6, 8}
        self.assertEqual(list(shell), [2, 6])
        shell = shell ^ {0, 2, 4, 6}
        self.assertEqual(list(shell), [0, 4])
        shell = shell - {0, 1}
        self.assertEqual(list(shell), [4])
        shell = {3, 4, 5} - shell  # type: ignore
        self.assertTrue(isinstance(shell, ShellMutableSet))
        self.assertEqual(list(shell), [3, 5])

    def test_ShellMutableSet_mutating_operators(self) -> None:
        shell = ShellMutableSet([2, 4])
        shell |= {6}
        self.assertEqual(list(shell), [2, 4, 6])
        shell &= {2, 6, 8}
        self.assertEqual(list(shell), [2, 6])
        shell ^= {0, 2, 4, 6}
        self.assertEqual(list(shell), [0, 4])
        shell -= {0, 1}
        self.assertEqual(list(shell), [4])

    def test_ShellMutableSet_errors(self) -> None:
        with self.assertRaises(KeyError):
            ShellMutableSet([2]).remove(3)
        with self.assertRaises(KeyError):
            ShellMutableSet([]).pop()
        with self.assertRaises(TypeError):
            ShellMutableSet(4)


@pytest.mark.parametrize("cut", [-2, 1, 3, 7])
@pytest.mark.parametrize("step", [-2, -1, 1, 2, 3])
@pytest.mark.parametrize("stop", [0, 2, 4, 6])
@pytest.mark.parametrize("start", [0, 1, 2, 3, 4, 5, 6])
def test_cut_slice(start, stop, step, cut):
    left, right = cut_slice(start, stop, step, cut)
    litems = list(range(left.start, left.stop, left.step))
    ritems = list(range(right.start, right.stop, right.step))
    expected = list(range(start, stop, step))
    assert expected == litems + ritems, f"{litems}+{ritems}, expected {expected}"


@pytest.mark.parametrize("idx", [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
def test_SequenceConcatenation_indexed(idx) -> None:
    c1 = SequenceConcatenation(SequenceConcatenation((11, 22), ()), (33,))
    # c1 = SequenceConcatenation((11, 22), (33,))
    c2 = [11, 22, 33]
    r1 = summarize_execution(lambda x: c1[x], (idx,), detach_path=False)
    expected = summarize_execution(lambda x: c2[x], (idx,), detach_path=False)
    assert r1 == expected


@pytest.mark.parametrize("step", [None, -1, 1, 2, -2, 3, -3])
@pytest.mark.parametrize("stop", [None, 0, 2, 4, 6, -2, -4, -6])
@pytest.mark.parametrize("start", [None, 0, 1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6])
def test_SequenceConcatenation_sliced(start, stop, step) -> None:
    c1 = SequenceConcatenation((11, 22, 33), (44, 55))
    c2 = [11, 22, 33, 44, 55]
    s = slice(start, stop, step)
    r1 = list(c1[s])
    expected = c2[s]
    assert r1 == expected


if __name__ == "__main__":
    unittest.main()
