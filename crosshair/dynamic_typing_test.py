import collections
import unittest
from typing import *

from crosshair.dynamic_typing import unify, realize

_T = TypeVar('_T')
_U = TypeVar('_U')
class UnifyTest(unittest.TestCase):
    def test_typevars(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Tuple[int, str, List[int]],
                              Tuple[int, _T, _U], bindings))
        self.assertEqual(realize(Mapping[_U, _T], bindings),
                         Mapping[List[int], str])
    
    def test_callable(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Callable[[int, str], List[int]],
                              Callable[[int, _T], _U], bindings))
        self.assertEqual(realize(Callable[[_U], _T], bindings),
                         Callable[[List[int]], str])
    
    def test_plain_callable(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Callable[[int, str], List[int]],
                              Callable, bindings))
    
    def test_uniform_tuple(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Iterable[int], Tuple[_T, ...], bindings))
        self.assertEqual(bindings[_T], int)

    def test_union_fail(self):
        bindings = collections.ChainMap()
        self.assertFalse(unify(Iterable[int], Union[int, Dict[_T, str]], bindings))
        
    def test_union_ok(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Iterable[int], Union[int, Tuple[_T, ...]], bindings))
        self.assertEqual(bindings[_T], int)

    

if __name__ == '__main__':
    unittest.main()
