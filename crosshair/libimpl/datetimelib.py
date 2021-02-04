import datetime
import importlib
import sys
from crosshair import register_patch, register_type
from crosshair import realize, with_realized_args, IgnoreAttempt
from typing import Any, Callable


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


def make_registrations():

    # WARNING: This is destructive for the datetime module.
    # It disables the C implementation for the entire interpreter.
    sys.modules["_datetime"] = None  # type: ignore
    importlib.reload(datetime)

    # Default pickling will realize the symbolic args; avoid this:
    datetime.date.__deepcopy__ = lambda s, _memo: datetime.date(s.year, s.month, s.day)
    datetime.time.__deepcopy__ = lambda s, _memo: datetime.time(
        s.hour, s.minute, s.second, s.microsecond, s.tzinfo
    )
    datetime.datetime.__deepcopy__ = lambda s, _memo: datetime.datetime(
        s.year, s.month, s.day, s.hour, s.minute, s.second, s.microsecond, s.tzinfo
    )
    datetime.timedelta.__deepcopy__ = lambda d, _memo: _timedelta_skip_construct(
        d.days, d.seconds, d.microseconds
    )

    # Mokey patch validation so that we can defer it.
    orig_check_date_fields = datetime._check_date_fields
    orig_check_time_fields = datetime._check_time_fields
    datetime._check_date_fields = lambda y, m, d: (y, m, d)
    datetime._check_time_fields = lambda h, m, s, u, f: (h, m, s, u, f)

    # TODO: `timezone` never makes a tzinfo with DST, so this is incomplete.
    # A complete solution would require generating a symbolc dst() member function.
    register_type(datetime.tzinfo, lambda p: p(datetime.timezone))

    _min_tz_offset, _max_tz_offset = -datetime.timedelta(hours=24), datetime.timedelta(
        hours=24
    )

    def make_timezone(p: Any) -> datetime.timezone:
        if p.space.smt_fork():
            delta = p(datetime.timedelta)
            if _min_tz_offset < delta < _max_tz_offset:
                return datetime.timezone(delta, p(str))
            else:
                raise IgnoreAttempt("Invalid timezone offset")
        else:
            return datetime.timezone.utc

    register_type(datetime.timezone, make_timezone)

    def make_date(p: Any) -> datetime.date:
        year, month, day = p(int), p(int), p(int)
        try:
            p.space.defer_assumption(
                "Invalid date",
                lambda: (
                    not _raises_value_error(orig_check_date_fields, (year, month, day))
                ),
            )
            return datetime.date(year, month, day)
        except ValueError:
            raise IgnoreAttempt("Invalid date")

    register_type(datetime.date, make_date)

    def make_time(p: Any) -> datetime.time:
        hour, minute, sec, usec = p(int), p(int), p(int), p(int)
        tzinfo = p(datetime.timezone) if p.space.smt_fork() else None
        try:
            p.space.defer_assumption(
                "Invalid datetime",
                lambda: (
                    not _raises_value_error(
                        orig_check_time_fields, (hour, minute, sec, usec, 0)
                    )
                ),
            )
            return datetime.time(hour, minute, sec, usec, tzinfo)
        except ValueError:
            raise IgnoreAttempt("Invalid datetime")

    register_type(datetime.time, make_time)

    def make_datetime(p: Any) -> datetime.datetime:
        year, month, day, hour, minute, sec, usec = (
            p(int),
            p(int),
            p(int),
            p(int),
            p(int),
            p(int),
            p(int),
        )
        tzinfo = p(datetime.tzinfo) if p.space.smt_fork() else None
        try:
            p.space.defer_assumption(
                "Invalid datetime",
                lambda: (
                    not _raises_value_error(orig_check_date_fields, (year, month, day))
                    and not _raises_value_error(
                        orig_check_time_fields, (hour, minute, sec, usec, 0)
                    )
                ),
            )
            return datetime.datetime(year, month, day, hour, minute, sec, usec, tzinfo)
        except ValueError:
            raise IgnoreAttempt("Invalid datetime")

    register_type(datetime.datetime, make_datetime)

    def make_timedelta(p: Callable) -> datetime.timedelta:
        microseconds, seconds, days = p(int), p(int), p(int)
        # the normalized ranges, per the docs:
        if not (
            0 <= microseconds < 1000000
            and 0 <= seconds < 3600 * 24
            and -999999999 <= days <= 999999999
        ):
            raise IgnoreAttempt("Invalid timedelta")
        try:
            return _timedelta_skip_construct(days, seconds, microseconds)
        except OverflowError:
            raise IgnoreAttempt("Invalid timedelta")

    register_type(datetime.timedelta, make_timedelta)
