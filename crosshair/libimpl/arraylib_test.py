from array import array

from crosshair.statespace import MessageType
from crosshair.test_util import check_states

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
        """post: a != TARGET"""
        a.insert(0, -33)

    check_states(f, MessageType.POST_FAIL)
