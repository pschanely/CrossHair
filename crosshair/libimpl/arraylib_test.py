import sys
from array import array

from crosshair.statespace import MessageType
from crosshair.test_util import check_states

TARGET = array("h", [-33, 0])

# The machine values 1 and -2 for typecode "h", in native byte order.
FROMBYTES_H_DATA = (1).to_bytes(2, sys.byteorder) + (-2).to_bytes(
    2, sys.byteorder, signed=True
)


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


def test_frombytes_decodes_machine_values():
    def f(a: array) -> None:
        """
        pre: a.typecode == 'h'
        post: list(a) != [1, -2]
        """
        a.frombytes(FROMBYTES_H_DATA)

    check_states(f, MessageType.POST_FAIL)


def test_frombytes_rejects_misaligned_length():
    def f(a: array) -> bool:
        """post: _"""
        if a.itemsize == 1:
            return True
        try:
            a.frombytes(b"\x00")
        except ValueError:
            return False
        return True

    check_states(f, MessageType.POST_FAIL)


def test_frombytes_rejects_non_bytes():
    def f(a: array) -> bool:
        """post: _"""
        try:
            a.frombytes([1, 2, 3])  # type: ignore[arg-type]
        except TypeError:
            return False
        return True

    check_states(f, MessageType.POST_FAIL)


def test_insert_negative_raises_on_unsigned():
    def f(a: array) -> bool:
        """post: _"""
        if a.typecode not in "BHILQ":
            return True
        try:
            a.insert(0, -3)
        except OverflowError:
            return False
        return True

    check_states(f, MessageType.POST_FAIL)


def test_setitem_negative_raises_on_unsigned():
    def f(a: array) -> bool:
        """post: _"""
        if a.typecode not in "BHILQ" or len(a) == 0:
            return True
        try:
            a[0] = -3
        except OverflowError:
            return False
        return True

    check_states(f, MessageType.POST_FAIL)


def test_add_different_typecode_raises():
    def f(a: array) -> bool:
        """post: _"""
        if a.typecode == "b":
            return True
        try:
            a + array("b", [1])
        except TypeError:
            return False
        return True

    check_states(f, MessageType.POST_FAIL)


def test_add_same_typecode_concatenates():
    def f(a: array) -> None:
        """
        pre: a.typecode == 'h'
        post: list(a) != [1, 2]
        """
        a = a + array("h", [1, 2])

    check_states(f, MessageType.POST_FAIL)
