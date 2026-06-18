import calendar
import datetime

import pytest

from crosshair.core import proxy_for_type
from crosshair.libimpl.builtinslib import make_bounded_int
from crosshair.libimpl.datetimelib import _is_leap_int
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


@pytest.mark.parametrize(
    "year",
    [1, 4, 100, 400, 1900, 1996, 2000, 2001, 2004, 2023, 2100, 2400, 9999],
)
def test_is_leap_int_matches_calendar(year: int) -> None:
    # Concrete inputs must produce the same answer as the stdlib (as a 0/1 int).
    assert _is_leap_int(year) == int(calendar.isleap(year))


def test_is_leap_int_stays_symbolic(space: StateSpace) -> None:
    # The whole point of the arithmetic form: it must NOT fork/realize the
    # year.  After the call, both a known leap year and a known common year
    # remain reachable, and the result can be both 1 and 0.  The old
    # ``_is_leap`` (which uses ``and``/``or``) would pin the year here.
    year = make_bounded_int("year", 1, 9999)
    with ResumedTracing():
        result = _is_leap_int(year)
        assert space.is_possible((year == 2000).var)  # leap still reachable
        assert space.is_possible((year == 2001).var)  # common still reachable
        assert space.is_possible((result == 1).var)  # result can be leap
        assert space.is_possible((result == 0).var)  # ...and common


def test_symbolic_date_year_stays_symbolic(space: StateSpace) -> None:
    # Reading a calendar field off an ordinal-backed date must stay symbolic and
    # non-forking.  The bridge decomposition (#428) introduces a clean symbolic
    # `year` linked to the ordinal, so the year is not pinned -- both a leap and a
    # common year remain reachable after the field access.
    d = proxy_for_type(datetime.date, "d")
    with ResumedTracing():
        year = d.year
        assert space.is_possible(year == 2000)  # leap reachable
        assert space.is_possible(year == 2001)  # common reachable


def test_symbolic_date_feb29_only_on_leap_years(space: StateSpace) -> None:
    # Feb 29 is reachable exactly on leap years.  The bridge decomposition adds
    # the day-validity constraint over the clean symbolic fields, so Feb 29 is
    # satisfiable only when the year is a leap year -- verified here directly
    # through the fields (which are non-forking under the bridge).
    d = proxy_for_type(datetime.date, "d")
    with ResumedTracing():
        feb29 = (d.month == 2) & (d.day == 29)
        assert space.is_possible((d.year == 2000) & feb29)  # leap: ok
        assert space.is_possible((d.year == 2400) & feb29)  # leap (400yr): ok
        assert not space.is_possible((d.year == 2001) & feb29)  # common: no
        assert not space.is_possible((d.year == 1900) & feb29)  # century non-leap


def test_leap_year() -> None:
    def f(start: datetime.date) -> datetime.date:
        """
        Adding 365 days does not always land on the next year (leap years
        have 366), so this postcondition is falsifiable.

        pre: start.month == 1 and start.day == 1
        post: _.year == start.year + 1
        raises: OverflowError
        """
        return start + datetime.timedelta(days=365)

    # Pin month/day to Jan 1 so the search only has to find a leap year, rather
    # than exploring all month/day combinations. Without this the counterexample
    # is found but timing-sensitively (~12-22s; the per-path Z3 timeout is
    # wall-clock), which flaked across CI runners. The symbolic year still
    # exercises the date + timedelta arithmetic that crosses the calendar.
    check_states(f, POST_FAIL)
