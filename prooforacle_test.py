import ast
import collections
import unittest

from astunparse import unparse

from asthelpers import exprparse
from prooforacle import *

class ProofOracleTest(unittest.TestCase):

    def test_gen_data(self):
        r = BiasedRandom(2, 100.0)
        counters = collections.defaultdict(int)
        for _ in range(100):
            counters[gen_data(r)] += 1
        print (counters.items())
        self.assertTrue(counters[0] > 10)
        self.assertTrue(counters[False] > 10)
        self.assertTrue(counters[True] > 10)

    def test_data_to_ast(self):
        self.assertEqual(True, data_to_ast(True).value)
        self.assertEqual(None, data_to_ast(None).value)
        self.assertEqual(0, data_to_ast(0).n)
        self.assertEqual('()\n', unparse(data_to_ast(())))
        self.assertEqual('(0, 1)\n', unparse(data_to_ast((0,1))))

    def test_crosshair_ast_eval(self):
        self.assertEqual(1, crosshair_ast_eval(data_to_ast(1)))
        self.assertEqual(False, crosshair_ast_eval(data_to_ast(False)))
        self.assertEqual((None, 0), crosshair_ast_eval(data_to_ast((None,0))))
        self.assertEqual((None, 0), crosshair_ast_eval(data_to_ast((None,0))))
        r = BiasedRandom(7, 100.0)
        self.assertTrue(crosshair_ast_eval(gen_predicate(r)) in (True, False))
        self.assertTrue(crosshair_ast_eval(gen_predicate(r)) in (True, False))
        self.assertTrue(crosshair_ast_eval(gen_predicate(r)) in (True, False))

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
        self.assertEqual(12,
            len([x for x in semantic_hash(exprparse("x+1 if True else ()")) if x]))

    def test_order_axioms(self):
        self.assertEqual([
            'filter(isint, (2, b, c))\n',
            'foo(filter(isint, (2, b, c)))\n',
            '(2, 3, 4)\n',
            '(2, x, 4)\n',
            'istuple(2, x, 4)\n',
        ], list(map(unparse, order_axioms([
                    exprparse('(2, x, 4)'),
                    exprparse('istuple(2, x, 4)'),
                    exprparse('(2, 3, 4)'),
                    exprparse('filter(isint, (2, b, c))'),
                    exprparse('foo(filter(isint, (2, b, c)))'),
                ],
                exprparse('filter(isint, (2, 3, 4))')
            ))))

if __name__ == '__main__':
    unittest.main()
