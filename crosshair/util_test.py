import unittest

from crosshair.util import *


class UtilTest(unittest.TestCase):

    def test_is_pure_python_functions(self):
        self.assertTrue(is_pure_python(is_pure_python))
        self.assertFalse(is_pure_python(map))

    def test_is_pure_python_classes(self):
        class RegularClass:
            pass
        class ClassWithSlots:
            __slots__ = ('x',)
        self.assertTrue(is_pure_python(RegularClass))
        self.assertTrue(is_pure_python(ClassWithSlots))
        self.assertFalse(is_pure_python(list))

    def test_is_pure_python_other_stuff(self):
        self.assertTrue(is_pure_python(7))
        self.assertTrue(is_pure_python(unittest))

    def test_dynamic_scope_var_basic(self):
        var = DynamicScopeVar(int, 'height')
        with var.open(7):
            self.assertEqual(var.get(), 7)

    def test_dynamic_scope_var_bsic(self):
        var = DynamicScopeVar(int, 'height')
        self.assertEqual(var.get_if_in_scope(), None)
        with var.open(7):
            self.assertEqual(var.get_if_in_scope(), 7)
        self.assertEqual(var.get_if_in_scope(), None)

    def test_dynamic_scope_var_error_cases(self):
        var = DynamicScopeVar(int, 'height')
        with var.open(100):
            with self.assertRaises(AssertionError, msg='Already in a height context'):
                with var.open(500, reentrant=False):
                    pass
        with self.assertRaises(AssertionError, msg='Not in a height context'):
            var.get()

    def test_tiny_stack(self):
        FS = traceback.FrameSummary
        s = tiny_stack([
            FS('a.py',                1, 'fooa'),
            FS('/crosshair/b.py',     2, 'foob'),
            FS('/crosshair/c.py',     3, 'fooc'),
            FS('/other/package/d.py', 4, 'food'),
            FS('/crosshair/e.py',     5, 'fooe'),
        ])
        self.assertEqual(s, '(fooa@a.py:1) (...x2) (food@d.py:4) (...x1)')

if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    else:
        unittest.main()
