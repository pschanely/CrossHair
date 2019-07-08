import unittest

from asthelpers import *

class AstHelpersTest(unittest.TestCase):

    def test_index_by_position(self):
        tree = ast.parse('''x = 1
def foo(y):
  return y''')
        index = index_by_position(tree)
        self.assertEqual(index[(1,0)], set([tree.body[0], tree.body[0].targets[0]]))
        self.assertEqual(index[(2,0)], set([tree.body[1]]))
        self.assertEqual(index[(3,2)], set([tree.body[1].body[0]]))

if __name__ == '__main__':
    unittest.main()
