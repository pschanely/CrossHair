import unittest

from asthelpers import *

class AstHelpersTest(unittest.TestCase):

    def test_semantic_hash(self):
        self.assertTrue(all(
            semantic_hash(exprparse("(lambda x:x+1)(2)")) >=
            semantic_hash(exprparse("(lambda f:f+1)"))))
        self.assertNotEqual(
            list(semantic_hash(exprparse("range(1)"))),
            list(semantic_hash(exprparse(  "len(1)"))))
        self.assertNotEqual(
            list(semantic_hash(exprparse("map(isint, ())"))),
            list(semantic_hash(exprparse("map(isbool, ())"))))
        self.assertNotEqual(
            list(semantic_hash(exprparse("range(x)"))),
            list(semantic_hash(exprparse("range(1)"))))
        self.assertEqual(
            list(semantic_hash(exprparse("range(1)"))),
            list(semantic_hash(exprparse("range(2)"))))
        self.assertEqual(
            list(semantic_hash(exprparse("x +  1 if (True) else ()"))),
            list(semantic_hash(exprparse("x + 0  if False  else ()"))))
        self.assertEqual(11,
            len([x for x in semantic_hash(exprparse("x+1 if True else ()")) if x]))

if __name__ == '__main__':
    unittest.main()
