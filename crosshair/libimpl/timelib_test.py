import time

from crosshair.statespace import CANNOT_CONFIRM, CONFIRMED, POST_FAIL, MessageType
from crosshair.test_util import check_states


def test_time_time():
    def f():
        """post: _ >= 0"""
        start = time.time()
        return time.time() - start

    check_states(f, POST_FAIL)


def test_time_time_ns():
    def f():
        """post: _ >= 0"""
        start = time.time_ns()
        return time.time_ns() - start

    check_states(f, POST_FAIL)


def test_time_monotonic():
    def f():
        """post: _ >= 0"""
        start = time.monotonic()
        return time.monotonic() - start

    check_states(f, CANNOT_CONFIRM)


def test_time_monotonic_ns():
    def f():
        """post: _ >= 0"""
        start = time.monotonic_ns()
        return time.monotonic_ns() - start

    check_states(f, CONFIRMED)
