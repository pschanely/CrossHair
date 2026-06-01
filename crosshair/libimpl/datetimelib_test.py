import datetime

import pytest

from crosshair.core import proxy_for_type
from crosshair.statespace import CONFIRMED, EXEC_ERR, POST_FAIL, StateSpace
from crosshair.test_util import check_states
from crosshair.tracers import ResumedTracing


def test_timedelta_symbolic_months_fail() -> None:
    def f(num_months: int) -> datetime.date:
        """
        pre: 0 <= num_months <= 100
        post: _.year != 2003
        """
        dt = datetime.date(2000, 1, 1)
        return dt + datetime.timedelta(days=30 * num_months)

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_date___init___method() -> None:
    def f(dt: datetime.date) -> datetime.date:
        """
        Is February 29th ever part of a valid date?
        (it is on leap years)

        post: (dt.month, dt.day) != (2, 29)
        """
        return dt

    check_states(f, POST_FAIL)


def test_time_fail() -> None:
    def f(dt: datetime.time) -> int:
        """
        post: _ != 14
        """
        return dt.hour

    check_states(f, POST_FAIL)


def test_datetime_fail() -> None:
    def f(dtime: datetime.datetime) -> int:
        """
        post: _ != 22
        """
        return dtime.second

    check_states(f, POST_FAIL)


def test_timedelta_fail() -> None:
    def f(d: datetime.timedelta) -> int:
        """
        post: _ != 9
        """
        return d.seconds

    check_states(f, POST_FAIL)


def test_date_plus_delta_fail() -> None:
    def f(delta: datetime.timedelta) -> datetime.date:
        """
        post: _.year != 2033
        raises: OverflowError
        """
        return datetime.date(2000, 1, 1) + delta

    check_states(f, POST_FAIL)


def test_date_plus_delta_overflow_err() -> None:
    def f(delta: datetime.timedelta) -> datetime.date:
        """
        post: True
        """
        return datetime.date(2000, 1, 1) + delta

    check_states(f, EXEC_ERR)


@pytest.mark.demo("yellow")
def test_timedelta___add___method() -> None:
    def f(delta: datetime.timedelta) -> datetime.date:
        """
        Can we increment 2000-01-01 by some time delta to reach 2001?

        NOTE: Although this counterexample is found relatively quickly,
        most date arithmetic solutions will require at least a minute
        of processing time.

        post: _.year != 2001
        raises: OverflowError
        """
        return datetime.date(2000, 1, 1) + delta

    check_states(f, POST_FAIL)


def _assert_both_reachable(space, value, needle) -> None:
    # The point of keeping comparisons symbolic is that the search can still
    # reach *both* sides; realizing an operand would pin it and make one side
    # unreachable (so e.g. a `!= needle` postcondition could never be
    # falsified).  We check reachability via independent satisfiability
    # queries -- no reliance on the result's internal type.
    with ResumedTracing():
        assert space.is_possible(value == needle)  # can equal the needle
        assert space.is_possible(value != needle)  # ...and can differ
        assert space.is_possible(value < needle)  # can be below
        assert space.is_possible(value >= needle)  # ...and at-or-above


def test_date_comparisons_reach_both_sides(space: StateSpace) -> None:
    d = proxy_for_type(datetime.date, "d")
    _assert_both_reachable(space, d, datetime.date(2030, 2, 14))


def test_datetime_comparisons_reach_both_sides(space: StateSpace) -> None:
    d = proxy_for_type(datetime.datetime, "d")
    _assert_both_reachable(space, d, datetime.datetime(2030, 2, 14, 9, 30, 0))


def test_timedelta_comparisons_reach_both_sides(space: StateSpace) -> None:
    d = proxy_for_type(datetime.timedelta, "d")
    _assert_both_reachable(space, d, datetime.timedelta(days=5, seconds=42))


def TODO_test_leap_year() -> None:
    # The solver returns unknown when adding a delta to a symbolic date. (nonlinear I think)
    def f(start: datetime.date) -> datetime.date:
        """
        post: _.year == start.year + 1
        """
        return start + datetime.timedelta(days=365)

    check_states(f, POST_FAIL)
