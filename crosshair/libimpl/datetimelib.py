import datetime
import importlib
import sys
from typing import Any, Optional, Tuple

from crosshair import deep_realize
from crosshair import realize
from crosshair import register_type
from crosshair import IgnoreAttempt
from crosshair import ResumedTracing
from crosshair.core import SymbolicFactory
from crosshair.libimpl.builtinslib import make_bounded_int


def _raises_value_error(fn, args):
    try:
        fn(*args)
        return False
    except ValueError:
        return True


def _timedelta_skip_construct(days, seconds, microseconds):
    # timedelta's constructor is convoluted to guide C implementations.
    # We use something simpler, and just ensure elsewhere that the inputs fall in the right ranges:
    delta = datetime.timedelta()
    delta._days = days  # type: ignore
    delta._seconds = seconds  # type: ignore
    delta._microseconds = microseconds  # type: ignore
    return delta


def _time_skip_construct(hour, minute, second, microsecond, tzinfo, fold):
    tm = datetime.time()
    tm._hour = hour
    tm._minute = minute
    tm._second = second
    tm._microsecond = microsecond
    tm._tzinfo = tzinfo
    tm._fold = fold
    return tm


def _date_skip_construct(year, month, day):
    dt = datetime.date(2020, 1, 1)
    dt._year = year
    dt._month = month
    dt._day = day
    return dt


def _datetime_skip_construct(
    year, month, day, hour, minute, second, microsecond, tzinfo
):
    dt = datetime.datetime(2020, 1, 1)
    dt._year = year
    dt._month = month
    dt._day = day
    dt._hour = hour
    dt._minute = minute
    dt._second = second
    dt._microsecond = microsecond
    dt._tzinfo = tzinfo
    return dt


def _symbolic_date_fields(varname: str) -> Tuple:
    return (
        make_bounded_int(varname + "_year", datetime.MINYEAR, datetime.MAXYEAR),
        make_bounded_int(varname + "_month", 1, 12),
        make_bounded_int(varname + "_day", 1, 31),
    )


def _symbolic_time_fields(varname: str) -> Tuple:
    return (
        make_bounded_int(varname + "_hour", 0, 23),
        make_bounded_int(varname + "_min", 0, 59),
        make_bounded_int(varname + "_sec", 0, 59),
        make_bounded_int(varname + "_usec", 0, 999999),
        make_bounded_int(varname + "_fold", 0, 1),
    )


def make_registrations():

    # WARNING: This is destructive for the datetime module.
    # It disables the C implementation for the entire interpreter.
    sys.modules["_datetime"] = None  # type: ignore
    importlib.reload(datetime)

    if sys.version_info >= (3, 10):
        # Prevent overeager realization.
        # TODO: This is operator.index; should we patch that instead?
        datetime._index = lambda x: x

    # Default pickling will realize the symbolic args; avoid this:
    datetime.date.__deepcopy__ = lambda s, _memo: _date_skip_construct(
        s.year, s.month, s.day
    )
    datetime.time.__deepcopy__ = lambda s, _memo: _time_skip_construct(
        s.hour, s.minute, s.second, s.microsecond, s.tzinfo
    )
    datetime.datetime.__deepcopy__ = lambda s, _memo: _datetime_skip_construct(
        s.year, s.month, s.day, s.hour, s.minute, s.second, s.microsecond, s.tzinfo
    )
    datetime.timedelta.__deepcopy__ = lambda d, _memo: _timedelta_skip_construct(
        d.days, d.seconds, d.microseconds
    )

    datetime.date.__ch_deep_realize__ = lambda s: _date_skip_construct(
        realize(s.year), realize(s.month), realize(s.day)
    )
    datetime.time.__ch_deep_realize__ = lambda s: _time_skip_construct(
        realize(s.hour),
        realize(s.minute),
        realize(s.second),
        realize(s.microsecond),
        deep_realize(s.tzinfo),
    )
    datetime.datetime.__ch_deep_realize__ = lambda s: _datetime_skip_construct(
        realize(s.year),
        realize(s.month),
        realize(s.day),
        realize(s.hour),
        realize(s.minute),
        realize(s.second),
        realize(s.microsecond),
        deep_realize(s.tzinfo),
    )
    datetime.timedelta.__ch_deep_realize__ = lambda d: _timedelta_skip_construct(
        realize(d.days), realize(d.seconds), realize(d.microseconds)
    )

    # TODO: `timezone` never makes a tzinfo with DST, so this is incomplete.
    # A complete solution would require generating a symbolc dst() member function.
    register_type(datetime.tzinfo, lambda p: p(datetime.timezone))

    # NOTE: these bounds have changed over python versions (e.g. [1]), so we pull the
    # following private constants from the runtime directly.
    # [1] https://github.com/python/cpython/commit/92c7e30adf5c81a54d6e5e555a6bdfaa60157a0d#diff-2a8962dcecb109859cedd81ddc5729bea57d156e0947cb8413f99781a0860fd1R2272
    _max_tz_offset = datetime.timezone._maxoffset
    _min_tz_offset = datetime.timezone._minoffset

    def make_timezone(p: Any) -> datetime.timezone:
        if p.space.smt_fork(desc="use explicit timezone"):
            delta = p(datetime.timedelta, "_offset")
            with ResumedTracing():
                if _min_tz_offset < delta < _max_tz_offset:
                    return datetime.timezone(delta, realize(p(str, "_name")))
                else:
                    raise IgnoreAttempt("Invalid timezone offset")
        else:
            return datetime.timezone.utc

    register_type(datetime.timezone, make_timezone)

    def make_date(p: Any) -> datetime.date:
        year, month, day = _symbolic_date_fields(p.varname)
        # We only approximate days-in-month upfront. Check for real after-the-fact:
        p.space.defer_assumption(
            "Invalid date",
            lambda: (
                not _raises_value_error(
                    datetime._check_date_fields, (year, month, day)  # type: ignore
                )
            ),
        )
        return _date_skip_construct(year, month, day)

    register_type(datetime.date, make_date)

    def make_time(p: Any) -> datetime.time:
        (hour, minute, sec, usec, fold) = _symbolic_time_fields(p.varname)
        tzinfo = p(Optional[datetime.timezone], "_tzinfo")
        return _time_skip_construct(hour, minute, sec, usec, tzinfo, fold)

    register_type(datetime.time, make_time)

    def make_datetime(p: Any) -> datetime.datetime:
        year, month, day = _symbolic_date_fields(p.varname)
        (hour, minute, sec, usec, fold) = _symbolic_time_fields(p.varname)
        tzinfo = p(Optional[datetime.timezone], "_tzinfo")
        # We only approximate days-in-month upfront. Check for real after-the-fact:
        p.space.defer_assumption(
            "Invalid datetime",
            lambda: not _raises_value_error(
                datetime._check_date_fields, (year, month, day)  # type: ignore
            ),
        )
        return _datetime_skip_construct(
            year, month, day, hour, minute, sec, usec, tzinfo
        )

    register_type(datetime.datetime, make_datetime)

    def make_timedelta(p: SymbolicFactory) -> datetime.timedelta:
        microseconds = make_bounded_int(p.varname + "_usec", 0, 999999)
        seconds = make_bounded_int(p.varname + "_sec", 0, 3600 * 24 - 1)
        days = make_bounded_int(p.varname + "_days", -999999999, 999999999)
        return _timedelta_skip_construct(days, seconds, microseconds)

    register_type(datetime.timedelta, make_timedelta)
