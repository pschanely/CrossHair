import time

from crosshair.statespace import MessageType
from crosshair.test_util import check_states


def test_time_time():
    def f():
        """post: _ >= 0"""
        start = time.time()
        return time.time() - start

    assert check_states(f) == {MessageType.POST_FAIL}


def test_time_time_ns():
    def f():
        """post: _ >= 0"""
        start = time.time_ns()
        return time.time_ns() - start

    assert check_states(f) == {MessageType.POST_FAIL}


def test_time_monotonic():
    def f():
        """post: _ >= 0"""
        start = time.monotonic()
        return time.monotonic() - start

    # TODO: Get this to MessageType.CONFIRMED
    # (float result capping causes problems when created below a ParallelNode;
    # instead we should ensure that we ONLY cap at the top of the tree)
    assert check_states(f) == {MessageType.CANNOT_CONFIRM}


def test_time_monotonic_ns():
    def f():
        """post: _ >= 0"""
        start = time.monotonic_ns()
        return time.monotonic_ns() - start

    assert check_states(f) == {MessageType.CONFIRMED}
