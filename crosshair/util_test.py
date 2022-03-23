import unittest

from crosshair.util import (
    CrosshairInternal,
    DynamicScopeVar,
    is_pure_python,
    measure_fn_coverage,
    set_debug,
    sourcelines,
    sys,
    traceback,
)
from crosshair.util import _tiny_stack_frames


class UtilTest(unittest.TestCase):
    def test_is_pure_python_functions(self):
        self.assertTrue(is_pure_python(is_pure_python))
        self.assertFalse(is_pure_python(map))

    def test_is_pure_python_classes(self):
        class RegularClass:
            pass

        class ClassWithSlots:
            __slots__ = ("x",)

        self.assertTrue(is_pure_python(RegularClass))
        self.assertTrue(is_pure_python(ClassWithSlots))
        self.assertFalse(is_pure_python(list))

    def test_is_pure_python_other_stuff(self):
        self.assertTrue(is_pure_python(7))
        self.assertTrue(is_pure_python(unittest))

    def test_dynamic_scope_var_basic(self):
        var = DynamicScopeVar(int, "height")
        with var.open(7):
            self.assertEqual(var.get(), 7)

    def test_dynamic_scope_var_bsic(self):
        var = DynamicScopeVar(int, "height")
        self.assertEqual(var.get_if_in_scope(), None)
        with var.open(7):
            self.assertEqual(var.get_if_in_scope(), 7)
        self.assertEqual(var.get_if_in_scope(), None)

    def test_dynamic_scope_var_error_cases(self):
        var = DynamicScopeVar(int, "height")
        with var.open(100):
            with self.assertRaises(AssertionError, msg="Already in a height context"):
                with var.open(500, reentrant=False):
                    pass
        with self.assertRaises(AssertionError, msg="Not in a height context"):
            var.get()

    def test_dynamic_scope_var_with_exception(self):
        var = DynamicScopeVar(int, "height")
        try:
            with var.open(7):
                raise NameError()
        except NameError:
            pass
        self.assertEqual(var.get_if_in_scope(), None)

    def test_tiny_stack_frames(self):
        FS = traceback.FrameSummary
        s = _tiny_stack_frames(
            [
                FS("a.py", 1, "fooa"),
                FS("/crosshair/b.py", 2, "foob"),
                FS("/crosshair/c.py", 3, "fooc"),
                FS("/other/package/d.py", 4, "food"),
                FS("/crosshair/e.py", 5, "fooe"),
            ]
        )
        self.assertEqual(s, "(fooa@a.py:1) (...x2) (food@d.py:4) (...x1)")

    def test_measure_fn_coverage(self) -> None:
        def called_by_foo(x: int) -> int:
            return x

        def foo(x: int) -> int:
            if called_by_foo(x) < 50:
                return x
            else:
                return (x - 50) + (called_by_foo(2 + 1) > 3) + -abs(x)

        def calls_foo(x: int) -> int:
            return foo(x)

        with measure_fn_coverage(foo) as coverage:
            calls_foo(5)
        self.assertGreater(0.4, coverage().opcode_coverage, 0.1)

        with measure_fn_coverage(foo) as coverage:
            calls_foo(100)
        self.assertGreater(0.95, coverage().opcode_coverage, 0.6)

        with measure_fn_coverage(foo) as coverage:
            calls_foo(5)
            calls_foo(100)
        # Note that we can't get 100% - there's an extra "return None"
        # at the end that's unreachable.
        self.assertGreater(coverage().opcode_coverage, 0.85)


class UnhashableCallable:
    def __hash__(self):
        raise CrosshairInternal("Do not hash")

    def __call__(self):
        return 42


def test_sourcelines_on_unhashable_callable():
    # Ensure we never trigger hashing when getting source code.
    sourcelines(UnhashableCallable())


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    else:
        unittest.main()
