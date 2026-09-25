from array import array

import pytest

from crosshair.libimpl.arraylib import INT_TYPE_BOUNDS, SymbolicArray
from crosshair.statespace import MessageType
from crosshair.test_util import check_states
from crosshair.tracers import ResumedTracing

TARGET = array("h", [-33, 0])


def test_mutated_array_compares_equal_to_concrete_array():
    def f(a: array) -> None:
        """post: a != TARGET"""
        a.append(0)

    check_states(f, MessageType.POST_FAIL)


def test_extended_array_compares_equal_to_concrete_array():
    def f(a: array) -> None:
        """post: a != TARGET"""
        a.extend([0])

    check_states(f, MessageType.POST_FAIL)


def test_array_with_insertion_compares_equal_to_concrete_array():
    def f(a: array) -> None:
        """
        pre: a.typecode == 'h'
        post: a != TARGET
        """
        a.insert(0, -33)

    check_states(f, MessageType.POST_FAIL)


@pytest.mark.parametrize("typecode", INT_TYPE_BOUNDS.keys())
def test_int_type_bounds_match_cpython(typecode):
    minval, maxval = INT_TYPE_BOUNDS[typecode]
    array(typecode, [minval, maxval - 1])
    with pytest.raises(OverflowError):
        array(typecode, [minval - 1])
    with pytest.raises(OverflowError):
        array(typecode, [maxval])


@pytest.mark.parametrize("typecode", INT_TYPE_BOUNDS.keys())
def test_frombytes_decodes_full_itemsize_values(typecode, space):
    itemsize = array(typecode).itemsize
    data = b"\x7f" * itemsize + b"\x80" * itemsize
    expected = array(typecode)
    expected.frombytes(data)
    a = SymbolicArray(typecode, [])
    with ResumedTracing():
        a.frombytes(data)
        assert list(a) == list(expected)


def test_frombytes_rejects_misaligned_length(space):
    a = SymbolicArray("h", [])
    with ResumedTracing():
        with pytest.raises(ValueError):
            a.frombytes(b"\x00")


def test_insert_negative_raises_on_unsigned(space):
    a = SymbolicArray("H", [])
    with ResumedTracing():
        with pytest.raises(OverflowError):
            a.insert(0, -3)


def test_setitem_negative_raises_on_unsigned(space):
    a = SymbolicArray("H", [0])
    with ResumedTracing():
        with pytest.raises(OverflowError):
            a[0] = -3


def test_setitem_checks_index_before_value(space):
    a = SymbolicArray("H", [])
    with ResumedTracing():
        with pytest.raises(IndexError):
            a[5] = -1


def test_add_same_typecode_concatenates(space):
    a = SymbolicArray("h", [1])
    with ResumedTracing():
        assert list(a + array("h", [2])) == [1, 2]


def test_type_error_cases(space):
    a = SymbolicArray("h", [])
    with ResumedTracing():
        with pytest.raises(TypeError):
            a.frombytes([1, 2, 3])  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            _ = a + array("b", [1])
        with pytest.raises(TypeError):
            a + [1]  # type: ignore[operator]
        with pytest.raises(TypeError):
            [1] + a  # type: ignore[operator]
        with pytest.raises(TypeError):
            a.extend(array("b", [1]))
        with pytest.raises(TypeError):
            a += array("b", [1])
        with pytest.raises(TypeError):
            a += [1]  # type: ignore[arg-type]
