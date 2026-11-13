import sys
from copy import deepcopy
from threading import RLock
from typing import Dict, List, Tuple

import pytest

from crosshair.copyext import CopyMode, copy_failures, deepcopyext
from crosshair.core_and_libs import proxy_for_type, standalone_statespace
from crosshair.libimpl.builtinslib import SymbolicInt
from crosshair.tracers import NoTracing


def test_deepcopyext_best_effort():
    lock = RLock()
    input = [lock]

    with pytest.raises(TypeError):
        deepcopy([lock])
    with pytest.raises(TypeError):
        deepcopyext(input, CopyMode.REGULAR, {})

    output = deepcopyext(input, CopyMode.BEST_EFFORT, {})
    assert input is not output
    assert input[0] is output[0]


class RaisesOnCopy:
    def __deepcopy__(self, memo):
        raise ValueError("no copies")


def test_deepcopyext_best_effort_reports_failures():
    uncopyable = RaisesOnCopy()
    input = [1, [uncopyable]]
    memo = {}
    output = deepcopyext(input, CopyMode.BEST_EFFORT, memo)
    assert output[1][0] is uncopyable
    (failure,) = copy_failures(input, memo)
    assert failure.typ is RaisesOnCopy
    assert str(failure) == "ValueError: no copies"
    assert copy_failures(input[1], memo) == [failure]


def test_deepcopyext_best_effort_reports_failures_through_shared_objects():
    shared = [RaisesOnCopy()]
    first, second, clean = [shared], [shared], [[]]
    memo = {}
    for obj in (first, second, clean):
        deepcopyext(obj, CopyMode.BEST_EFFORT, memo)
    assert len(copy_failures(first, memo)) == 1
    assert copy_failures(second, memo) == copy_failures(first, memo)
    assert copy_failures(clean, memo) == []


def test_deepcopyext_reports_lazy_heap_failures():
    with standalone_statespace as space:
        with NoTracing():
            handles = proxy_for_type(List[RaisesOnCopy], "handles")
            memo = {}
            handles_copy = deepcopyext(handles, CopyMode.BEST_EFFORT, memo)
            assert copy_failures(handles, memo) == []
        space.add(len(handles_copy) == 1)
        handles_copy[0]
        with NoTracing():
            assert len(copy_failures(handles, memo)) == 1


def test_deepcopyext_symbolic_set():
    with standalone_statespace:
        symbolic_int = proxy_for_type(int, "symbolic_int")
        s = {i for i in (symbolic_int, 42)}
        with NoTracing():
            # Ensure this doesn't crash with "Numeric operation on symbolic...":
            deepcopyext(s, CopyMode.REALIZE, {})


def test_deepcopyext_realize_simple(space):
    x = SymbolicInt("x")
    input = (x,)
    output = deepcopyext(input, CopyMode.REALIZE, {})
    assert input is not output
    assert input[0] is not output[0]
    assert type(input[0]) is SymbolicInt
    assert type(output[0]) is int


def test_deepcopyext_realize(space):
    x = SymbolicInt("x")
    lock = RLock()
    lockarray = [lock]
    input = {"a": ([x], lockarray, lockarray)}
    output = deepcopyext(input, CopyMode.REALIZE, {})
    assert input is not output
    assert input["a"] is not output["a"]
    assert output["a"][1] is output["a"][2]  # memo preserves identity
    assert type(input["a"][0][0]) is SymbolicInt
    assert type(output["a"][0][0]) is int


def test_deepcopyext_tuple_type():
    assert deepcopy(Tuple) is Tuple
    assert deepcopyext(Tuple, CopyMode.REALIZE, {}) is Tuple


class RecursiveType:
    a: "RecursiveType"

    def set(self, a):
        self.a = a


def test_deepcopyext_recursivetype(space):
    recursive_obj = RecursiveType()
    recursive_obj.set(recursive_obj)
    deepcopyext(recursive_obj, CopyMode.REALIZE, {})


class SlottedWithGetstate:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return (self.x + 1,)

    def __setstate__(self, state):
        self.x = state[0] - 1


class WithGetnewargs:
    def __new__(cls, x):
        obj = super().__new__(cls)
        obj.x = x
        return obj

    def __getnewargs__(self):
        return (self.x + 0,)

    def __getstate__(self):
        return None


@pytest.mark.parametrize("cls", [SlottedWithGetstate, WithGetnewargs])
def test_deepcopyext_realize_runs_reduction_hooks_traced(cls, space):
    x = SymbolicInt("x")
    output = deepcopyext(cls(x), CopyMode.REALIZE, {})
    assert type(output.x) is int


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="dict views not inspect-able before python 3.10"
)
def test_fsum_accepts_dict_values_argument(space):
    d = {"a": proxy_for_type(int, "b")}
    realized_vals = deepcopyext(d.values(), CopyMode.REALIZE, {})
    assert type(next(iter(realized_vals))) is int


def test_stdlib_deepcopy_of_symbolic_object_in_heap():
    with standalone_statespace as space:
        with NoTracing():
            mapping = proxy_for_type(Dict[int, object], "mapping")
        space.add(len(mapping) == 1)
        value = next(iter(mapping.values()))
        deepcopy((mapping, value))
