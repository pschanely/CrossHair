import ast
import unittest
from crosshair.rewriting import preprocess_pattern, matches, astparse, unparse, replace_pattern_vars

class AstHelpersTest(unittest.TestCase):

    def test_basic(self):
        bindings = {}
        patt, repl = preprocess_pattern(astparse('0 + X'), astparse('X + 1'))
        self.assertEqual('(0 + $X)', unparse(patt))
        self.assertTrue(matches(astparse('0 + 2'), patt, bindings))
        self.assertEqual('2', unparse(bindings['X']))
        self.assertEqual('(2 + 1)', unparse(replace_pattern_vars(repl, bindings)))
        patt = preprocess_pattern(ast.parse('if C:\n  X\nY\n'))
        repl = preprocess_pattern(ast.parse('if C:\n  X\n  Y\nelse:\n  Y'))
        self.assertTrue(matches(ast.parse('if True:\n  42\n20'), patt, bindings))
        self.assertEqual(
            'if True:\n    42\n    20\nelse:\n    20',
            unparse(replace_pattern_vars(repl, bindings)))
        bindings = {}
        self.assertTrue(matches(ast.parse("""if True:
  42
10
20"""), patt, bindings))
        self.assertEqual(
            'if True:\n    42\n    10\n    20\nelse:\n    10\n    20',
            unparse(replace_pattern_vars(repl, bindings)))

if __name__ == '__main__':
    unittest.main()
