import unittest

from crosshair.simplestructs import *

class SimpleStructTests(unittest.TestCase):
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

    def test_ShellMutableSequence(self) -> None:
        l = ['0', '1', '2', '3']
        shell = ShellMutableSequence(l)
        self.assertEqual(shell, shell)
        self.assertEqual(shell, l)
        shell[1:3] = ['1', '1.5', '2']
        self.assertEqual(shell, ['0', '1', '1.5', '2', '3'])
        self.assertEqual(shell, shell)

if __name__ == '__main__':
    unittest.main()
