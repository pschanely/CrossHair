import datetime
import importlib
import sys
from crosshair import register_patch, register_type
from crosshair import realize, with_realized_args, IgnoreAttempt
from typing import Any, Callable


def make_registrations():

    # This is destructive for the datetime module:
    # It disables the C implementation for the entire interpreter.
    sys.modules['_datetime'] = None  # type: ignore
    importlib.reload(datetime)

    # Default pickling will realize the symbolic args; avoid this:
    datetime.date.__deepcopy__ = lambda s, _memo: datetime.date(s.year, s.month, s.day)
    datetime.timedelta.__deepcopy__ = lambda d, _memo: datetime.timedelta(
        days=d.days, seconds=d.seconds, microseconds=d.microseconds)

    # Mokey patch year/month/day validation to defer it.
    orig_check_date_fields = datetime._check_date_fields
    datetime._check_date_fields = lambda y, m, d: (y, m, d)
    def date_fields_ok(y, m, d):
        try:
            orig_check_date_fields(y, m, d)
            return True
        except ValueError:
            return False

    def make_date(p: Any) -> datetime.date:
        year, month, day = p(int), p(int), p(int)
        try:
            p.space.defer_assumption('Invalid date', lambda: date_fields_ok(year, month, day))
            return datetime.date(year, month, day)
        except ValueError:
            raise IgnoreAttempt('Invalid date')

    register_type(datetime.date, make_date)
    
    def make_timedelta(p: Callable) -> datetime.timedelta:
        microseconds, seconds, days = p(int), p(int), p(int)
        # the normalized ranges, per the docs:
        if not(0 <= microseconds < 1000000 and
               0 <= seconds < 3600*24 and
               -999999999 <= days <= 999999999):
            raise IgnoreAttempt('Invalid timedelta')
        try:
            return datetime.timedelta(
                days=days, seconds=seconds,
                microseconds=microseconds)
        except OverflowError:
            raise IgnoreAttempt('Invalid timedelta')

    register_type(datetime.timedelta, make_timedelta)
