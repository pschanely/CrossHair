import collections
import unittest
from typing import *

from crosshair.dynamic_typing import unify, realize

_T = TypeVar('_T')
_U = TypeVar('_U')
class UnifyTest(unittest.TestCase):
    def test_raw_tuple(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(tuple, Iterable[_T], bindings))

    def test_typevars(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Tuple[int, str, List[int]],
                              Tuple[int, _T, _U], bindings))
        self.assertEqual(realize(Mapping[_U, _T], bindings),
                         Mapping[List[int], str])
    
    def test_bound_vtypears(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Dict[str, int], Dict[_T, _U]))
        self.assertFalse(unify(Dict[str, int], Dict[_T, _T]))

    def test_zero_type_args_ok(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(map, Iterable[_T]))
        self.assertFalse(unify(map, Iterable[int]))
        
    def test_callable(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Callable[[Iterable], bool],
                              Callable[[List], bool], bindings))
        self.assertFalse(unify(Callable[[List], bool],
                              Callable[[Iterable], bool], bindings))
        self.assertTrue(unify(Callable[[int, _T], List[int]],
                              Callable[[int, str], _U], bindings))
        self.assertEqual(realize(Callable[[_U], _T], bindings),
                         Callable[[List[int]], str])
    
    def test_plain_callable(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Callable[[int, str], List[int]],
                              Callable, bindings))
    
    def test_uniform_tuple(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Tuple[int, int], Tuple[_T, ...], bindings))
        self.assertEqual(bindings[_T], int)
        self.assertFalse(unify(Tuple[int, str], Tuple[_T, ...], bindings))

    def test_tuple(self):
        bindings = collections.ChainMap()
        self.assertFalse(unify(tuple, Tuple[int, str], bindings))

    def test_union_fail(self):
        bindings = collections.ChainMap()
        self.assertFalse(unify(Iterable[int], Union[int, Dict[str, _T]], bindings))
        
    def test_union_ok(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(int, Union[str, int], bindings))
        self.assertTrue(unify(Tuple[int, ...], Union[int, Iterable[_T]], bindings))
        self.assertEqual(bindings[_T], int)

    def test_union_into_union(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(Union[str, int], Union[str, int, float],  bindings))
        self.assertFalse(unify(Union[str, int, float], Union[str, int],  bindings))

    def test_nested_union(self):
        bindings = collections.ChainMap()
        self.assertTrue(unify(List[str], Sequence[Union[str, int]], bindings))

if __name__ == '__main__':
    unittest.main()
