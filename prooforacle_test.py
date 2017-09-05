import ast
import collections
import unittest

from astunparse import unparse

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
        
if __name__ == '__main__':
    unittest.main()
