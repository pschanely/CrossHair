import sys
import traceback
import unittest
from enum import Enum

import numpy
import pytest

from crosshair.util import (
    CrosshairInternal,
    DynamicScopeVar,
    _tiny_stack_frames,
    eval_friendly_repr,
    is_pure_python,
    renamed_function,
    set_debug,
    sourcelines,
)


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
        self.assertEqual(s, "(fooa a.py:1) (...x2) (food d.py:4) (...x1)")


class UnhashableCallable:
    def __hash__(self):
        raise CrosshairInternal("Do not hash")

    def __call__(self):
        return 42


def test_sourcelines_on_unhashable_callable():
    # Ensure we never trigger hashing when getting source code.
    sourcelines(UnhashableCallable())


def test_eval_friendly_repr():
    # Class
    assert eval_friendly_repr(unittest.TestCase) == "unittest.case.TestCase"
    # Pure-python method:
    assert (
        eval_friendly_repr(unittest.TestCase.assertTrue)
        == "unittest.case.TestCase.assertTrue"
    )
    # Builtin function:
    assert eval_friendly_repr(print) == "print"
    # Object:
    assert eval_friendly_repr(object()) == "object()"
    # Special float values:
    assert eval_friendly_repr(float("nan")) == 'float("nan")'
    # MethodDescriptorType
    assert (
        eval_friendly_repr(numpy.random.RandomState.randint)
        == "numpy.random.mtrand.RandomState.randint"
    )
    # enum
    class Color(Enum):
        RED = 0

    assert eval_friendly_repr(Color.RED) == "Color.RED"

    # We return to original repr() behaviors afterwards:
    assert repr(float("nan")) == "nan"
    assert repr(Color.RED) == "<Color.RED: 0>"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Python 3.8+ required")
def test_renamed_function():
    def crash_on_seven(x):
        if x == 7:
            raise IOError
        return x

    hello = renamed_function(crash_on_seven, "hello")
    hello(6)
    try:
        hello(7)
    except IOError as e:
        assert traceback.extract_tb(e.__traceback__)[-1].name == "hello"


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    else:
        unittest.main()
