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

if __name__ == '__main__':
    unittest.main()
