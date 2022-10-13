import time

import pytest

from crosshair.statespace import CANNOT_CONFIRM, CONFIRMED, POST_FAIL, MessageType
from crosshair.test_util import check_states


@pytest.mark.demo
def test_time():
    def f():
        """
        Can time go backwards?

        NOTE: CrossHair allows time() to produce ANY value.
        Although highly unlikely, it's possible that the system clock
        is set backwards while a program executes.
        (BTW: use time.monotonic if you don't want it to go backwards!)

        CrossHair's counterexample includes a monkey-patching context
        manager that lets you reproduce the issue, e.g.:
            with crosshair.patch_to_return({time.time: [2.0, 1.0]}):
                f()

        post: _ >= 0
        """
        start = time.time()
        return time.time() - start

    check_states(f, POST_FAIL)


def test_time_ns():
    def f():
        """post: _ >= 0"""
        start = time.time_ns()
        return time.time_ns() - start

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_monotonic():
    def f():
        """
        Can time increase by one second between monotonic() calls?

        post: _ != 1.0
        """
        start = time.monotonic()
        end = time.monotonic()
        return end - start

    check_states(f, POST_FAIL)


def test_monotonic_confirm():
    def f():
        """post: _ >= 0"""
        start = time.monotonic()
        return time.monotonic() - start

    check_states(f, CANNOT_CONFIRM)


def test_monotonic_ns():
    def f():
        """post: _ >= 0"""
        start = time.monotonic_ns()
        return time.monotonic_ns() - start

    check_states(f, CONFIRMED)
