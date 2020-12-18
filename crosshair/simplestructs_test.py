import unittest

from crosshair.simplestructs import *

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
        self.assertEqual({'c': 'd', 'a': 'b'}.popitem(), ('a', 'b'))
        self.assertEqual(SimpleDict([('c', 'd'), ('a', 'b')]).popitem(), ('a', 'b'))

    def test_ShellMutableMap_poo(self) -> None:
        m = ShellMutableMap({2: 0})
        self.assertEqual(0, m.setdefault(2.0, {True: '0'}))

    def test_sequence_concatenation(self) -> None:
        c1 = SequenceConcatenation((11,22,33), (44,55,66))
        c2 =                       [11,22,33,   44,55,66]
        ctr = 0
        for start in [None,0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6]:
            for stop in [None,0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6]:
                for step in [None, -1, 1, 2, -2, 3, -3]:
                    s = slice(start, stop, step)
                    r1 = list(c1[s])
                    r2 = c2[s]
                    self.assertEqual(r1, r2, f'{ctr}: {s}: {r1} vs {r2}')
                    ctr += 1

    def test_slice_view(self) -> None:
        nums = ['0', '1', '2', '3', '4', '5']
        ctr = 0
        for start in [0,1,2,3,4,5,6]:
            for stop in range(start, 7):
                view = SliceView(nums, start, stop)
                concrete = nums[start : stop]
                self.assertEqual(list(view), concrete, f'{ctr}: {start}:{stop}: {view} vs {concrete}')
                if stop - start > 0:
                    self.assertEqual(view[0], nums[start])
                ctr += 1

    def test_ShellMutableSequence_slice_assignment(self) -> None:
        l = ['0', '1', '2', '3']
        shell = ShellMutableSequence(l)
        self.assertEqual(shell, shell)
        self.assertEqual(shell, l)
        shell[1:3] = ['1', '1.5', '2']
        self.assertEqual(shell, ['0', '1', '1.5', '2', '3'])
        self.assertEqual(shell, shell)

    def test_ShellMutableSequence_bad_index(self) -> None:
        with self.assertRaises(TypeError):
            ShellMutableSequence([])["whoa"] = 3

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
            [3, 5] ^ s
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
        shell = {3, 4, 5} - shell
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

if __name__ == '__main__':
    unittest.main()
