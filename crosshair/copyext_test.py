from copy import deepcopy
from threading import RLock
from typing import Tuple

import pytest

from crosshair.copyext import CopyMode, deepcopyext
from crosshair.core_and_libs import proxy_for_type, standalone_statespace
from crosshair.libimpl.builtinslib import SymbolicInt
from crosshair.tracers import NoTracing, ResumedTracing


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


def test_deepcopyext_symbolic_set():
    with standalone_statespace:
        symbolic_int = proxy_for_type(int, "symbolic_int")
        s = {i for i in (symbolic_int, 42)}
        with NoTracing():
            # Ensure this doesn't crash with "Numeric operation on symbolic...":
            deepcopyext(s, CopyMode.REALIZE, {})


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
