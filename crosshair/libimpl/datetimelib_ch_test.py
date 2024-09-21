import operator
import sys
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Tuple, Union

import pytest  # type: ignore

from crosshair.core_and_libs import MessageType, analyze_function, run_checkables
from crosshair.test_util import ResultComparison, compare_results, compare_returns

# crosshair: max_uninteresting_iterations=10


def _invoker(method_name):
    def invoke(obj, *args):
        return getattr(obj, method_name)(*args)

    return invoke


# TODO: test replace(); test realize?


# special methods


def check_datetimelib_lt(
    p: Union[
        Tuple[timedelta, timedelta],
        Tuple[date, datetime],
        Tuple[datetime, datetime],
    ]
) -> ResultComparison:
    """post: _"""
    return compare_results(operator.lt, *p)


def check_datetimelib_add(
    p: Union[
        Tuple[timedelta, timedelta],
        Tuple[date, timedelta],
        Tuple[timedelta, datetime],
    ]
) -> ResultComparison:
    """post: _"""
    return compare_results(operator.add, *p)


def check_datetimelib_subtract(
    p: Union[
        Tuple[timedelta, timedelta],
        Tuple[date, timedelta],
        Tuple[datetime, timedelta],
        Tuple[datetime, datetime],
    ]
) -> ResultComparison:
    """post: _"""
    return compare_results(operator.sub, *p)


def check_datetimelib_str(
    obj: Union[timedelta, timezone, date, time, datetime]
) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("__str__"), obj)


def check_datetimelib_repr(
    # TODO: re-enable time, datetime repr checking after fixing in Python 3.11
    obj: Union[timedelta, timezone, date]
) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("__repr__"), obj)


# timedelta


def check_timedelta_new(
    # `days` and `secs` can be floats but imprecision can cause this test to fail
    # right now. Start testing floats once we fix this issue:
    # https://github.com/pschanely/CrossHair/issues/230
    days: int,
    secs: int,
    microsecs: Union[int, float],
) -> ResultComparison:
    """post: _"""
    if days % 1 != 0 or secs % 1 != 0 or microsecs % 1 != 0:
        pass
    return compare_returns(lambda *a: timedelta(*a), days, secs, microsecs)


def check_timedelta_bool(td: timedelta) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("__bool__"), td)


def check_timedelta_neg(td: timedelta) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("__neg__"), td)


def check_timedelta_abs(td: timedelta) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("__abs__"), td)


def check_timedelta_truediv(
    numerator: timedelta, denominator: Union[timedelta, int]
) -> ResultComparison:
    """post: _"""
    return compare_results(operator.truediv, numerator, denominator)


def check_timedelta_floordiv(
    numerator: timedelta, denominator: Union[timedelta, int]
) -> ResultComparison:
    """post: _"""
    return compare_results(operator.floordiv, numerator, denominator)


def check_timedelta_multiply(td: timedelta, factor: int) -> ResultComparison:
    """post: _"""
    return compare_results(operator.mul, td, factor)


def check_timedelta_divmod(
    numerator: timedelta, denominator: timedelta
) -> ResultComparison:
    """post: _"""
    return compare_results(divmod, numerator, denominator)


def check_timedelta_total_seconds(td: timedelta) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("total_seconds"), td)


# date


def check_date_replace(dt: date, year: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda d, m, u: dt.replace(year=year), dt, year)


def check_date_timetuple(dt: date) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("timetuple"), dt)


def check_date_toordinal(dt: date) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("toordinal"), dt)


def check_date_weekday(dt: date) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("weekday"), dt)


def check_date_isoweekday(dt: date) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("isoweekday"), dt)


def check_date_isocalendar(dt: date) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("isocalendar"), dt)


def check_date_isoformat(dt: date) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("isoformat"), dt)


def check_date_ctime(dt: date) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("ctime"), dt)


def check_date_strftime(dt: date, fmt: str) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("strftime"), dt, fmt)


# datetime


def check_datetime_date(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("date"), dt)


def check_datetime_time(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("time"), dt)


def check_datetime_timetz(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("timetz"), dt)


def check_datetime_replace(
    dt: datetime, month: int, microsecond: int
) -> ResultComparison:
    """post: _"""
    return compare_results(
        lambda d, m, u: d.replace(month=m, microsecond=u), dt, month, microsecond
    )


def check_datetime_astimezone(dt: datetime, tz: tzinfo) -> ResultComparison:
    """post: _"""
    # crosshair: max_uninteresting_iterations=3
    return compare_results(_invoker("astimezone"), dt, tz)


def check_datetime_utcoffset(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("utcoffset"), dt)


def check_datetime_dst(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("dst"), dt)


def check_datetime_tzname(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("tzname"), dt)


def check_datetime_timetuple(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("timetuple"), dt)


def check_datetime_utctimetuple(dt: datetime) -> ResultComparison:
    """post: _"""
    # crosshair: max_uninteresting_iterations=5
    return compare_results(_invoker("utctimetuple"), dt)


def check_datetime_toordinal(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("toordinal"), dt)


def check_datetime_timestamp(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("timestamp"), dt)


def check_datetime_weekday(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("weekday"), dt)


def check_datetime_isoweekday(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("isoweekday"), dt)


def check_datetime_isocalendar(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("isocalendar"), dt)


def check_datetime_isoformat(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("isoformat"), dt)


def check_datetime_ctime(dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("ctime"), dt)


def check_datetime_strftime(dt: datetime, fmt: str) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("strftime"), dt, fmt)


# time


def check_time_replace(tm: time, hour: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t, h: t.replace(hour=h), tm, hour)


def check_time_isoformat(tm: time) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("isoformat"), tm)


def check_time_strftime(tm: time, fmt: str) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("strftime"), tm, fmt)


def check_time_utcoffset(tm: time) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("utcoffset"), tm)


def check_time_dst(tm: time) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("dst"), tm)


def check_time_tzname(tm: time) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("tzname"), tm)


# timezone


def check_timezone_utcoffset(tz: timezone, dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("utcoffset"), tz, dt)


def check_timezone_dst(tz: timezone, dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("dst"), tz, dt)


def check_timezone_tzname(tz: timezone, dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("tzname"), tz, dt)


def check_timezone_fromutc(tz: timezone, dt: datetime) -> ResultComparison:
    """post: _"""
    return compare_results(_invoker("fromutc"), tz, dt)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    this_module = sys.modules[__name__]
    messages = run_checkables(analyze_function(getattr(this_module, fn_name)))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
