import builtins
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from typing import Optional

import numpy
import pytest

from crosshair.util import (
    CrosshairInternal,
    DynamicScopeVar,
    _tiny_stack_frames,
    eval_friendly_repr,
    format_boundargs,
    is_pure_python,
    renamed_function,
    sourcelines,
)


def test_is_pure_python_functions():
    assert is_pure_python(is_pure_python)
    assert not is_pure_python(map)


def test_is_pure_python_classes():
    class RegularClass:
        pass

    class ClassWithSlots:
        __slots__ = ("x",)

    assert is_pure_python(RegularClass)
    assert is_pure_python(ClassWithSlots)
    assert not is_pure_python(list)


def test_is_pure_python_other_stuff():
    assert is_pure_python(7)
    assert is_pure_python(pytest)


def test_dynamic_scope_var_basic():
    var = DynamicScopeVar(int, "height")
    with var.open(7):
        assert var.get() == 7


def test_dynamic_scope_var_bsic():
    var = DynamicScopeVar(int, "height")
    assert var.get_if_in_scope() is None
    with var.open(7):
        assert var.get_if_in_scope() == 7
    assert var.get_if_in_scope() is None


def test_dynamic_scope_var_error_cases():
    var = DynamicScopeVar(int, "height")
    with var.open(100):
        with pytest.raises(AssertionError, match="Already in a height context"):
            with var.open(500, reentrant=False):
                pass
    with pytest.raises(AssertionError, match="Not in a height context"):
        var.get()


def test_dynamic_scope_var_with_exception():
    var = DynamicScopeVar(int, "height")
    try:
        with var.open(7):
            raise NameError()
    except NameError:
        pass
    assert var.get_if_in_scope() is None


def test_tiny_stack_frames():
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
    assert s == "(fooa a.py:1) (...x2) (food d.py:4) (...x1)"


class UnhashableCallable:
    def __hash__(self):
        raise CrosshairInternal("Do not hash")

    def __call__(self):
        return 42


def test_sourcelines_on_unhashable_callable():
    # Ensure we never trigger hashing when getting source code.
    sourcelines(UnhashableCallable())


def eat_things(p1, *varp, kw1=4, kw2="default", **varkw):
    pass


def test_format_boundargs():
    bound = signature(eat_things).bind(1, 2, 3, kw2=5, other=6)
    assert format_boundargs(bound) == "1, 2, 3, kw1=4, kw2=5, other=6"


class Color(Enum):
    RED = 0


@dataclass
class Pair:
    x: Optional["Pair"] = None
    y: Optional["Pair"] = None

    def __repr__(self):  # TODO tracing on avoids this?
        return f"Pair({builtins.repr(self.x)}, {builtins.repr(self.y)})"


def test_eval_friendly_repr():
    # Class
    assert eval_friendly_repr(Color) == "Color"
    # Pure-python method:
    assert (
        eval_friendly_repr(UnhashableCallable.__hash__) == "UnhashableCallable.__hash__"
    )
    # Builtin function:
    assert eval_friendly_repr(print) == "print"
    # Object:
    assert eval_friendly_repr(object()) == "object()"
    # Special float values:
    assert eval_friendly_repr(float("nan")) == 'float("nan")'
    # MethodDescriptorType
    assert eval_friendly_repr(numpy.random.RandomState.randint) == "RandomState.randint"
    # Preserve identical objects
    a = Pair()
    assert eval_friendly_repr(Pair(a, a)) == "Pair(v1:=Pair(None, None), v1)"
    # do not attempt to re-use ReferencedIdentifiers
    assert eval_friendly_repr(Pair(Pair, Pair)) == "Pair(Pair, Pair)"
    # enums:
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
