#
# This file includes a modified version of CPython's pure python datetime
# implementation from:
# https://github.com/python/cpython/blob/v3.10.2/Lib/datetime.py
#
# The shared source code is licensed under the PSF license and is
# copyright © 2001-2022 Python Software Foundation; All Rights Reserved
#
# See the "LICENSE" file for complete license details on CrossHair.
#

# NOTE: At least some of this code could be rewritten to be more
# symbolic-friendly. Since much is fork-lifted from CPython, do not
# assume the coding decisions made here are very intentional or
# optimal.


import math as _math
import sys
import time as _time
from datetime import date as real_date
from datetime import datetime as real_datetime
from datetime import time as real_time
from datetime import timedelta as real_timedelta
from datetime import timezone as real_timezone
from datetime import tzinfo as real_tzinfo
from enum import Enum
from typing import Any, Optional, Tuple, Union

from crosshair import (
    IgnoreAttempt,
    ResumedTracing,
    realize,
    register_patch,
    register_type,
)
from crosshair.core import SymbolicFactory
from crosshair.libimpl.builtinslib import make_bounded_int, smt_or
from crosshair.statespace import context_statespace
from crosshair.tracers import NoTracing


def _cmp(x, y):
    return 0 if x == y else 1 if x > y else -1


MINYEAR = 1
MAXYEAR = 9999
_MAXORDINAL = 3652059  # date.max.toordinal()


# -1 is a placeholder for indexing purposes.
_DAYS_IN_MONTH = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

_DAYS_BEFORE_MONTH = [-1]  # -1 is a placeholder for indexing purposes.
dbm = 0
for dim in _DAYS_IN_MONTH[1:]:
    _DAYS_BEFORE_MONTH.append(dbm)
    dbm += dim
del dbm, dim


def _is_leap(year):
    """year -> 1 if leap year, else 0."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


import z3  # type: ignore


def _smt_is_leap(smt_year):
    """year -> 1 if leap year, else 0."""
    return context_statespace().smt_fork(
        z3.And(smt_year % 4 == 0, z3.Or(smt_year % 100 != 0, smt_year % 400 == 0)),
        "is leap year",
        probability_true=0.25,
    )


def _days_before_year(year):
    """year -> number of days before January 1st of year."""
    y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400


def _days_in_month(year, month):
    """year, month -> number of days in that month in that year."""
    # Avoid _DAYS_IN_MONTH so that we don't realize the month
    assert 1 <= month <= 12, month
    if month >= 8:
        return 31 if month % 2 == 0 else 30
    else:
        if month == 2:
            return 29 if _is_leap(year) else 28
        else:
            return 30 if month % 2 == 0 else 31


def _smt_days_in_month(smt_year, smt_month, smt_day):
    """constraint smt_day to match the year and month."""
    feb_days = 29 if _smt_is_leap(smt_year) else 28
    return z3.Or(
        z3.And(
            smt_day <= feb_days,
            smt_month == 2,
        ),
        z3.And(
            smt_day <= 30,
            z3.Or(
                smt_month == 4,
                smt_month == 6,
                smt_month == 9,
                smt_month == 11,
            ),
        ),
        z3.And(
            smt_day <= 31,
            z3.Or(
                smt_month == 1,
                smt_month == 3,
                smt_month == 5,
                smt_month == 7,
                smt_month == 8,
                smt_month == 10,
                smt_month == 12,
            ),
        ),
    )


def _days_before_month(year, month):
    """year, month -> number of days in year preceding first day of month."""
    assert 1 <= month <= 12, "month must be in 1..12"
    return _DAYS_BEFORE_MONTH[month] + (month > 2 and _is_leap(year))


def _ymd2ord(year, month, day):
    """year, month, day -> ordinal, considering 01-Jan-0001 as day 1."""
    assert 1 <= month <= 12, "month must be in 1..12"
    dim = _days_in_month(year, month)
    assert 1 <= day <= dim, "day must be in 1..%d" % dim
    return _days_before_year(year) + _days_before_month(year, month) + day


_DI400Y = _days_before_year(401)  # number of days in 400 years
_DI100Y = _days_before_year(101)  #    "    "   "   " 100   "
_DI4Y = _days_before_year(5)  #    "    "   "   "   4   "

# A 4-year cycle has an extra leap day over what we'd get from pasting
# together 4 single years.
assert _DI4Y == 4 * 365 + 1

# Similarly, a 400-year cycle has an extra leap day over what we'd get from
# pasting together 4 100-year cycles.
assert _DI400Y == 4 * _DI100Y + 1

# OTOH, a 100-year cycle has one fewer leap day than we'd get from
# pasting together 25 4-year cycles.
assert _DI100Y == 25 * _DI4Y - 1


def _ord2ymd(n):
    """ordinal -> (year, month, day), considering 01-Jan-0001 as day 1."""
    n -= 1
    n400, n = divmod(n, _DI400Y)
    year = n400 * 400 + 1  # ..., -399, 1, 401, ...
    n100, n = divmod(n, _DI100Y)
    n4, n = divmod(n, _DI4Y)
    n1, n = divmod(n, 365)
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        assert n == 0
        return year - 1, 12, 31
    leapyear = n1 == 3 and (n4 != 24 or n100 == 3)
    assert leapyear == _is_leap(year)
    month = (n + 50) >> 5
    preceding = _DAYS_BEFORE_MONTH[month] + (month > 2 and leapyear)
    if preceding > n:  # estimate is too large
        month -= 1
        preceding -= _DAYS_IN_MONTH[month] + (month == 2 and leapyear)
    n -= preceding
    assert 0 <= n < _days_in_month(year, month)
    return year, month, n + 1


# Month and day names.  For localized versions, see the calendar module.
_MONTHNAMES = [
    None,
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
_DAYNAMES = [None, "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _build_struct_time(y, m, d, hh, mm, ss, dstflag):
    wday = (_ymd2ord(y, m, d) + 6) % 7
    dnum = _days_before_month(y, m) + d
    return _time.struct_time((y, m, d, hh, mm, ss, wday, dnum, dstflag))


def _format_time(hh, mm, ss, us, timespec="auto"):
    specs = {
        "hours": "{:02d}",
        "minutes": "{:02d}:{:02d}",
        "seconds": "{:02d}:{:02d}:{:02d}",
        "milliseconds": "{:02d}:{:02d}:{:02d}.{:03d}",
        "microseconds": "{:02d}:{:02d}:{:02d}.{:06d}",
    }

    if timespec == "auto":
        # Skip trailing microseconds when us==0.
        timespec = "microseconds" if us else "seconds"
    elif timespec == "milliseconds":
        us //= 1000
    try:
        fmt = specs[timespec]
    except KeyError:
        raise ValueError("Unknown timespec value")
    else:
        return fmt.format(hh, mm, ss, us)


def _format_offset(off):
    s = ""
    if off is not None:
        if off.days < 0:
            sign = "-"
            off = -off
        else:
            sign = "+"
        hh, mm = divmod(off, timedelta(hours=1))
        mm, ss = divmod(mm, timedelta(minutes=1))
        s += "%s%02d:%02d" % (sign, hh, mm)
        if ss or ss.microseconds:
            s += ":%02d" % ss.seconds

            if ss.microseconds:
                s += ".%06d" % ss.microseconds
    return s


# Correctly substitute for %z and %Z escapes in strftime formats.
def _wrap_strftime(object, format, timetuple):
    format = realize(format)
    # Don't call utcoffset() or tzname() unless actually needed.
    freplace = None  # the string to use for %f
    zreplace = None  # the string to use for %z
    Zreplace = None  # the string to use for %Z

    # Scan format for %z and %Z escapes, replacing as needed.
    newformat = []
    push = newformat.append
    i, n = 0, len(format)
    while i < n:
        ch = format[i]
        i += 1
        if ch == "%":
            if i < n:
                ch = format[i]
                i += 1
                if ch == "f":
                    if freplace is None:
                        freplace = "%06d" % getattr(object, "microsecond", 0)
                    newformat.append(freplace)
                elif ch == "z":
                    if zreplace is None:
                        zreplace = ""
                        if hasattr(object, "utcoffset"):
                            offset = object.utcoffset()
                            if offset is not None:
                                sign = "+"
                                if offset.days < 0:
                                    offset = -offset
                                    sign = "-"
                                h, rest = divmod(offset, timedelta(hours=1))
                                m, rest = divmod(rest, timedelta(minutes=1))
                                s = rest.seconds
                                u = offset.microseconds
                                if u:
                                    zreplace = "%c%02d%02d%02d.%06d" % (
                                        sign,
                                        h,
                                        m,
                                        s,
                                        u,
                                    )
                                elif s:
                                    zreplace = "%c%02d%02d%02d" % (sign, h, m, s)
                                else:
                                    zreplace = "%c%02d%02d" % (sign, h, m)
                    assert "%" not in zreplace
                    newformat.append(zreplace)
                elif ch == "Z":
                    if Zreplace is None:
                        Zreplace = ""
                        if hasattr(object, "tzname"):
                            s = object.tzname()
                            if s is not None:
                                # strftime is going to have at this: escape %
                                Zreplace = s.replace("%", "%%")
                    newformat.append(Zreplace)
                else:
                    push("%")
                    push(ch)
            else:
                push("%")
        else:
            push(ch)
    newformat = "".join(newformat)
    return _time.strftime(newformat, timetuple)


# Helpers for parsing the result of isoformat()
def _parse_isoformat_date(dtstr):
    # It is assumed that this function will only be called with a
    # string of length exactly 10, and (though this is not used) ASCII-only
    year = int(dtstr[0:4])
    if dtstr[4] != "-":
        raise ValueError("Invalid date separator")

    month = int(dtstr[5:7])

    if dtstr[7] != "-":
        raise ValueError("Invalid date separator")

    day = int(dtstr[8:10])

    return [year, month, day]


def _parse_hh_mm_ss_ff(tstr):
    # Parses things of the form HH[:MM[:SS[.fff[fff]]]]
    len_str = len(tstr)

    time_comps = [0, 0, 0, 0]
    pos = 0
    for comp in range(0, 3):
        if (len_str - pos) < 2:
            raise ValueError("Incomplete time component")

        time_comps[comp] = int(tstr[pos : pos + 2])

        pos += 2
        next_char = tstr[pos : pos + 1]

        if not next_char or comp >= 2:
            break

        if next_char != ":":
            raise ValueError("Invalid time separator")

        pos += 1

    if pos < len_str:
        if tstr[pos] != ".":
            raise ValueError("Invalid microsecond component")
        else:
            pos += 1

            len_remainder = len_str - pos
            if len_remainder not in (3, 6):
                raise ValueError("Invalid microsecond component")

            time_comps[3] = int(tstr[pos:])
            if len_remainder == 3:
                time_comps[3] *= 1000

    return time_comps


def _parse_isoformat_time(tstr):
    # Format supported is HH[:MM[:SS[.fff[fff]]]][+HH:MM[:SS[.ffffff]]]
    len_str = len(tstr)
    if len_str < 2:
        raise ValueError("Isoformat time too short")

    # This is equivalent to re.search('[+-]', tstr), but faster
    tz_pos = tstr.find("-") + 1 or tstr.find("+") + 1
    timestr = tstr[: tz_pos - 1] if tz_pos > 0 else tstr

    time_comps = _parse_hh_mm_ss_ff(timestr)

    tzi = None
    if tz_pos > 0:
        tzstr = tstr[tz_pos:]

        # Valid time zone strings are:
        # HH:MM               len: 5
        # HH:MM:SS            len: 8
        # HH:MM:SS.ffffff     len: 15

        if len(tzstr) not in (5, 8, 15):
            raise ValueError("Malformed time zone string")

        tz_comps = _parse_hh_mm_ss_ff(tzstr)
        if all(x == 0 for x in tz_comps):
            tzi = timezone.utc
        else:
            tzsign = -1 if tstr[tz_pos - 1] == "-" else 1

            td = timedelta(
                hours=tz_comps[0],
                minutes=tz_comps[1],
                seconds=tz_comps[2],
                microseconds=tz_comps[3],
            )

            tzi = timezone(tzsign * td)

    time_comps.append(tzi)

    return time_comps


# Just raise TypeError if the arg isn't None or a string.
def _check_tzname(name):
    if name is not None and not isinstance(name, str):
        raise TypeError(
            "tzinfo.tzname() must return None or string, " "not '%s'" % type(name)
        )


# name is the offset-producing method, "utcoffset" or "dst".
# offset is what it returned.
# If offset isn't None or timedelta, raises TypeError.
# If offset is None, returns None.
# Else offset is checked for being in range.
# If it is, its integer value is returned.  Else ValueError is raised.
def _check_utc_offset(name, offset):
    assert name in ("utcoffset", "dst")
    if offset is None:
        return
    if not isinstance(offset, real_timedelta):
        raise TypeError(
            "tzinfo.%s() must return None "
            "or timedelta, not '%s'" % (name, type(offset))
        )
    if not -timedelta(1) < offset < timedelta(1):
        raise ValueError(
            "%s() must be strictly between "
            "-timedelta(hours=24) and timedelta(hours=24)" % (name)
        )


def _check_ints(values):
    for value in values:
        if not isinstance(value, int):
            raise TypeError


def _check_date_fields(year, month, day):
    _check_ints((year, month, day))
    if not MINYEAR <= year <= MAXYEAR:
        raise ValueError("year must be in %d..%d" % (MINYEAR, MAXYEAR))
    if not 1 <= month <= 12:
        raise ValueError("month must be in 1..12")
    dim = _days_in_month(year, month)
    if not 1 <= day <= dim:
        raise ValueError("day must be in 1..%d" % dim)
    return year, month, day


def _check_time_fields(hour, minute, second, microsecond, fold):
    _check_ints((hour, minute, second, microsecond, fold))
    if not 0 <= hour <= 23:
        raise ValueError("hour must be in 0..23")
    if not 0 <= minute <= 59:
        raise ValueError("minute must be in 0..59")
    if not 0 <= second <= 59:
        raise ValueError("second must be in 0..59")
    if not 0 <= microsecond <= 999999:
        raise ValueError("microsecond must be in 0..999999")
    if fold not in (0, 1):
        raise ValueError("fold must be either 0 or 1")
    return hour, minute, second, microsecond, fold


def _check_tzinfo_arg(tz):
    if tz is not None and not isinstance(tz, real_tzinfo):
        raise TypeError("tzinfo argument must be None or of a tzinfo subclass")


def _cmperror(x, y):
    raise TypeError("can't compare '%s' to '%s'" % (type(x).__name__, type(y).__name__))


def _divide_and_round(a, b):
    """
    divide a by b and round result to the nearest integer

    When the ratio is exactly half-way between two integers,
    the even integer is returned.
    """
    # Based on the reference implementation for divmod_near
    # in Objects/longobject.c.
    q, r = divmod(a, b)
    # round up if either r / b > 0.5, or r / b == 0.5 and q is odd.
    # The expression r / b > 0.5 is equivalent to 2 * r > b if b is
    # positive, 2 * r < b if b negative.
    r *= 2
    greater_than_half = r > b if b > 0 else r < b
    if greater_than_half or r == b and q % 2 == 1:
        q += 1

    return q


def _timedelta_to_microseconds(td):
    return (td.days * (24 * 3600) + td.seconds) * 1000000 + td.microseconds


def _timedelta_getstate(self):
    return (self.days, self.seconds, self.microseconds)


class timedelta:
    """
    Represent the difference between two datetime objects.

    Supported operators:

    - add, subtract timedelta
    - unary plus, minus, abs
    - compare to timedelta
    - multiply, divide by int

    In addition, datetime supports subtraction of two datetime objects
    returning a timedelta, and addition or subtraction of a datetime
    and a timedelta giving a datetime.

    Representation: (days, seconds, microseconds).  Why?  Because I
    felt like it.
    """

    def __new__(
        cls,
        days=0,
        seconds=0,
        microseconds=0,
        milliseconds=0,
        minutes=0,
        hours=0,
        weeks=0,
    ):
        # Doing this efficiently and accurately in C is going to be difficult
        # and error-prone, due to ubiquitous overflow possibilities, and that
        # C double doesn't have enough bits of precision to represent
        # microseconds over 10K years faithfully.  The code here tries to make
        # explicit where go-fast assumptions can be relied on, in order to
        # guide the C implementation; it's way more convoluted than speed-
        # ignoring auto-overflow-to-long idiomatic Python could be.

        # XXX Check that all inputs are ints or floats.

        # Normalize everything to days, seconds, microseconds.
        days += weeks * 7
        seconds += minutes * 60 + hours * 3600
        microseconds += milliseconds * 1000

        # roll fractional values down to lower tiers
        if isinstance(days, float):
            days, fractional_days = divmod(days, 1)
            seconds += fractional_days * (24 * 3600)
        if isinstance(seconds, float):
            seconds, fractional_seconds = divmod(seconds, 1)
            microseconds += fractional_seconds * (1_000_000)
        if isinstance(microseconds, float):
            microseconds = round(microseconds)

        # now everything is an integer; roll overflow back up into higher tiers:
        if not (0 <= microseconds < 1_000_000):
            addl_seconds, microseconds = divmod(microseconds, 1_000_000)
            seconds += addl_seconds
        if not (0 <= seconds < 24 * 3600):
            addl_days, seconds = divmod(seconds, 24 * 3600)
            days += addl_days

        if abs(days) > 999999999:
            raise OverflowError

        self = object.__new__(cls)
        self._days = days
        self._seconds = seconds
        self._microseconds = microseconds
        self._hashcode = -1
        return self

    def __repr__(self):
        args = []
        if self._days:
            args.append("days=%d" % self._days)
        if self._seconds:
            args.append("seconds=%d" % self._seconds)
        if self._microseconds:
            args.append("microseconds=%d" % self._microseconds)
        if not args:
            args.append("0")
        return "%s.%s(%s)" % (
            type(self).__module__,
            self.__class__.__qualname__,
            ", ".join(args),
        )

    def __str__(self):
        mm, ss = divmod(self._seconds, 60)
        hh, mm = divmod(mm, 60)
        s = "%d:%02d:%02d" % (hh, mm, ss)
        if self._days:

            def plural(n):
                return n, abs(n) != 1 and "s" or ""

            s = ("%d day%s, " % plural(self._days)) + s
        if self._microseconds:
            s = s + ".%06d" % self._microseconds
        return s

    def total_seconds(self):
        """Total seconds in the duration."""
        return (
            (self.days * 86400 + self.seconds) * 10**6 + self.microseconds
        ) / 10**6

    # Read-only field accessors
    @property
    def days(self):
        """days"""
        return self._days

    @property
    def seconds(self):
        """seconds"""
        return self._seconds

    @property
    def microseconds(self):
        """microseconds"""
        return self._microseconds

    def __add__(self, other):
        if isinstance(other, real_timedelta):
            # for CPython compatibility, we cannot use
            # our __class__ here, but need a real timedelta
            return timedelta(
                self._days + other.days,
                self._seconds + other.seconds,
                self._microseconds + other.microseconds,
            )
        elif isinstance(other, real_datetime):
            return datetime.fromdatetime(other).__add__(self)
        elif isinstance(other, real_date):
            return date.fromdate(other).__add__(self)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, real_timedelta):
            # for CPython compatibility, we cannot use
            # our __class__ here, but need a real timedelta
            return timedelta(
                self._days - other.days,
                self._seconds - other.seconds,
                self._microseconds - other.microseconds,
            )
        elif isinstance(other, real_date):
            return date.fromdate(other).__add__(self)
        elif isinstance(other, real_datetime):
            return datetime.fromdatetime(other).__add__(self)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, real_timedelta):
            return -self + other
        return NotImplemented

    def __neg__(self):
        # for CPython compatibility, we cannot use
        # our __class__ here, but need a real timedelta
        return timedelta(-self._days, -self._seconds, -self._microseconds)

    def __pos__(self):
        return self

    def __abs__(self):
        if self._days < 0:
            return -self
        else:
            return self

    def __mul__(self, other):
        if isinstance(other, int):
            # for CPython compatibility, we cannot use
            # our __class__ here, but need a real timedelta
            return timedelta(
                self._days * other, self._seconds * other, self._microseconds * other
            )
        if isinstance(other, float):
            usec = _timedelta_to_microseconds(self)
            a, b = other.as_integer_ratio()
            return timedelta(0, 0, _divide_and_round(usec * a, b))
        return NotImplemented

    __rmul__ = __mul__

    def __floordiv__(self, other):
        if not isinstance(other, (int, real_timedelta)):
            return NotImplemented
        usec = _timedelta_to_microseconds(self)
        if isinstance(other, real_timedelta):
            return usec // _timedelta_to_microseconds(other)
        if isinstance(other, int):
            return timedelta(0, 0, usec // other)

    def __truediv__(self, other):
        if not isinstance(other, (int, float, real_timedelta)):
            return NotImplemented
        usec = _timedelta_to_microseconds(self)
        if isinstance(other, real_timedelta):
            return usec / _timedelta_to_microseconds(other)
        if isinstance(other, int):
            return timedelta(0, 0, _divide_and_round(usec, other))
        if isinstance(other, float):
            a, b = other.as_integer_ratio()
            return timedelta(0, 0, _divide_and_round(b * usec, a))

    def __mod__(self, other):
        if isinstance(other, real_timedelta):
            r = _timedelta_to_microseconds(self) % _timedelta_to_microseconds(other)
            return timedelta(0, 0, r)
        return NotImplemented

    def __divmod__(self, other):
        if isinstance(other, real_timedelta):
            q, r = divmod(
                _timedelta_to_microseconds(self), _timedelta_to_microseconds(other)
            )
            return q, timedelta(0, 0, r)
        return NotImplemented

    # Comparisons of timedelta objects with other.

    def __eq__(self, other):
        if isinstance(other, real_timedelta):
            return self._cmp(other) == 0
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, real_timedelta):
            return self._cmp(other) <= 0
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, real_timedelta):
            return self._cmp(other) < 0
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, real_timedelta):
            return self._cmp(other) >= 0
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, real_timedelta):
            return self._cmp(other) > 0
        else:
            return NotImplemented

    def _cmp(self, other):
        assert isinstance(other, real_timedelta)
        return _cmp(_timedelta_getstate(self), _timedelta_getstate(other))

    def __hash__(self):
        if self._hashcode == -1:
            self._hashcode = hash(_timedelta_getstate(self))
        return self._hashcode

    def __bool__(self):
        return realize(
            smt_or(
                self._microseconds != 0,
                smt_or(
                    self._seconds != 0,
                    self._days != 0,
                ),
            )
        )

    def __ch_realize__(self):
        return real_timedelta(
            days=realize(self._days),
            seconds=realize(self._seconds),
            milliseconds=realize(self._microseconds) / 1000.0,
        )

    def __ch_pytype__(self):
        return real_timedelta


timedelta.min = timedelta(-999999999)  # type: ignore
timedelta.max = timedelta(  # type: ignore
    days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999
)
timedelta.resolution = timedelta(microseconds=1)  # type: ignore


def _date_getstate(self):
    yhi, ylo = divmod(self.year, 256)
    return bytes([yhi, ylo, self.month, self.day])


class date:
    """
    Concrete date type.

    Constructors:

    __new__()
    fromtimestamp()
    today()
    fromordinal()

    Operators
    ---------
    __repr__, __str__
    __eq__, __le__, __lt__, __ge__, __gt__, __hash__
    __add__, __radd__, __sub__ (add/radd only with timedelta arg)

    Methods
    -------
    timetuple()
    toordinal()
    weekday()
    isoweekday(), isocalendar(), isoformat()
    ctime()
    strftime()

    Properties (readonly):
    ---------------------
    year, month, day

    """

    def __init__(self, year, month, day):
        """
        Constructor.

        :param year:
        :param month: month, starting at 1
        :param day: day, starting at 1
        """
        year, month, day = _check_date_fields(year, month, day)
        self._year = year
        self._month = month
        self._day = day
        self._hashcode = -1

    # Additional constructors

    @classmethod
    def fromtimestamp(cls, t):
        """Construct a date from a POSIX timestamp (like time.time())."""
        y, m, d, hh, mm, ss, weekday, jday, dst = _time.localtime(t)
        return cls(y, m, d)

    @classmethod
    def today(cls):
        """Construct a date from time.time()."""
        t = _time.time()
        return cls.fromtimestamp(t)

    @classmethod
    def fromordinal(cls, n):
        """
        Construct a date from a proleptic Gregorian ordinal.

        January 1 of year 1 is day 1.  Only the year, month and day are
        non-zero in the result.
        """
        y, m, d = _ord2ymd(n)
        return cls(y, m, d)

    @classmethod
    def fromisoformat(cls, date_string):
        """Construct a date from the output of date.isoformat()."""
        if not isinstance(date_string, str):
            raise TypeError("fromisoformat: argument must be str")

        try:
            assert len(date_string) == 10
            return cls(*_parse_isoformat_date(date_string))
        except Exception:
            raise ValueError(f"Invalid isoformat string")

    @classmethod
    def fromisocalendar(cls, year, week, day):
        """
        Construct a date from the ISO year, week number and weekday.

        This is the inverse of the date.isocalendar() function
        """
        # Year is bounded this way because 9999-12-31 is (9999, 52, 5)
        if not MINYEAR <= year <= MAXYEAR:
            raise ValueError(f"Year is out of range")

        if not 0 < week < 53:
            out_of_range = True

            if week == 53:
                # ISO years have 53 weeks in them on years starting with a
                # Thursday and leap years starting on a Wednesday
                first_weekday = _ymd2ord(year, 1, 1) % 7
                if first_weekday == 4 or (first_weekday == 3 and _is_leap(year)):
                    out_of_range = False

            if out_of_range:
                raise ValueError(f"Invalid week: {week}")

        if not 0 < day < 8:
            raise ValueError(f"Invalid weekday (range is [1, 7])")

        # Now compute the offset from (Y, 1, 1) in days:
        day_offset = (week - 1) * 7 + (day - 1)

        # Calculate the ordinal day for monday, week 1
        day_1 = _isoweek1monday(year)
        ord_day = day_1 + day_offset

        return cls(*_ord2ymd(ord_day))

    @classmethod
    def fromdate(cls, d: real_date):
        return cls(d.year, d.month, d.day)

    # Conversions to string

    def __repr__(self):
        return "%s.%s(%d, %d, %d)" % (
            type(self).__module__,
            self.__class__.__qualname__,
            self._year,
            self._month,
            self._day,
        )

    # XXX These shouldn't depend on time.localtime(), because that
    # clips the usable dates to [1970 .. 2038).  At least ctime() is
    # easily done without using strftime() -- that's better too because
    # strftime("%c", ...) is locale specific.

    def ctime(self):
        """Return ctime() style string."""
        weekday = self.toordinal() % 7 or 7
        return "%s %s %2d 00:00:00 %04d" % (
            _DAYNAMES[weekday],
            _MONTHNAMES[self._month],
            self._day,
            self._year,
        )

    def strftime(self, fmt):
        """Format using strftime()."""
        return _wrap_strftime(self, fmt, self.timetuple())

    def __format__(self, fmt):
        if not isinstance(fmt, str):
            raise TypeError("must be str, not %s" % type(fmt).__name__)
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)

    def isoformat(self):
        """
        Return the date formatted according to ISO.

        This is 'YYYY-MM-DD'.

        References
        ----------
        - http://www.w3.org/TR/NOTE-datetime
        - http://www.cl.cam.ac.uk/~mgk25/iso-time.html

        """
        return "%04d-%02d-%02d" % (self._year, self._month, self._day)

    __str__ = isoformat

    # Read-only field accessors
    @property
    def year(self):
        """year (1-9999)"""
        return self._year

    @property
    def month(self):
        """month (1-12)"""
        return self._month

    @property
    def day(self):
        """day (1-31)"""
        return self._day

    # Standard conversions, __eq__, __le__, __lt__, __ge__, __gt__,
    # __hash__ (and helpers)

    def timetuple(self):
        """Return local time tuple compatible with time.localtime()."""
        return _build_struct_time(self._year, self._month, self._day, 0, 0, 0, -1)

    def toordinal(self):
        """
        Return proleptic Gregorian ordinal for the year, month and day.

        January 1 of year 1 is day 1.  Only the year, month and day values
        contribute to the result.
        """
        return _ymd2ord(self._year, self._month, self._day)

    def replace(self, year=None, month=None, day=None):
        """Return a new date with new values for the specified fields."""
        if year is None:
            year = self._year
        if month is None:
            month = self._month
        if day is None:
            day = self._day
        return date(year, month, day)

    # Comparisons of date objects with other.

    def __eq__(self, other):
        if isinstance(other, real_date):
            return self._cmp(other) == 0
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, real_date):
            return self._cmp(other) <= 0
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, real_date):
            return self._cmp(other) < 0
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, real_date):
            return self._cmp(other) >= 0
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, real_date):
            return self._cmp(other) > 0
        return NotImplemented

    def _cmp(self, other):
        assert isinstance(other, real_date)
        y, m, d = self._year, self._month, self._day
        y2, m2, d2 = other.year, other.month, other.day
        return _cmp((y, m, d), (y2, m2, d2))

    def __hash__(self):
        if self._hashcode == -1:
            self._hashcode = hash(_date_getstate(self))
        return self._hashcode

    # Computations

    def __add__(self, other):
        """Add a date to a timedelta."""
        if isinstance(other, real_timedelta):
            o = self.toordinal() + other.days
            if 0 < o <= _MAXORDINAL:
                return date.fromordinal(o)
            raise OverflowError("result out of range")
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        """Subtract two dates, or a date and a timedelta."""
        if isinstance(other, real_timedelta):
            return self + timedelta(-other.days)
        if isinstance(other, real_date):
            days1 = self.toordinal()
            days2 = other.toordinal()
            return timedelta(days1 - days2)
        return NotImplemented

    def weekday(self):
        """Return day of the week, where Monday == 0 ... Sunday == 6."""
        return (self.toordinal() + 6) % 7

    # Day-of-the-week and week-of-the-year, according to ISO

    def isoweekday(self):
        """Return day of the week, where Monday == 1 ... Sunday == 7."""
        # 1-Jan-0001 is a Monday
        return self.toordinal() % 7 or 7

    def isocalendar(self):
        """
        Return a named tuple containing ISO year, week number, and weekday.

        The first ISO week of the year is the (Mon-Sun) week
        containing the year's first Thursday; everything else derives
        from that.

        The first week is 1; Monday is 1 ... Sunday is 7.

        ISO calendar algorithm taken from
        http://www.phys.uu.nl/~vgent/calendar/isocalendar.htm
        (used with permission)
        """
        year = self._year
        week1monday = _isoweek1monday(year)
        today = _ymd2ord(self._year, self._month, self._day)
        # Internally, week and day have origin 0
        week, day = divmod(today - week1monday, 7)
        if week < 0:
            year -= 1
            week1monday = _isoweek1monday(year)
            week, day = divmod(today - week1monday, 7)
        elif week >= 52:
            if today >= _isoweek1monday(year + 1):
                year += 1
                week = 0
        return _IsoCalendarDate(year, week + 1, day + 1)

    def __ch_realize__(self):
        return real_date(realize(self._year), realize(self._month), realize(self._day))

    def __ch_pytype__(self):
        return real_date


_date_class = date  # so functions w/ args named "date" can get at the class

date.min = date(1, 1, 1)  # type: ignore
date.max = date(9999, 12, 31)  # type: ignore
date.resolution = timedelta(days=1)  # type: ignore


class tzinfo:
    """
    Abstract base class for time zone info classes.

    Subclasses must override the name(), utcoffset() and dst() methods.
    """

    def tzname(self, dt):
        """datetime -> string name of time zone."""
        raise NotImplementedError("tzinfo subclass must override tzname()")

    def utcoffset(self, dt):
        """datetime -> timedelta, positive for east of UTC, negative for west of UTC"""
        raise NotImplementedError("tzinfo subclass must override utcoffset()")

    def dst(self, dt):
        """
        datetime -> DST offset as timedelta, positive for east of UTC.

        Return 0 if DST not in effect.  utcoffset() must include the DST
        offset.
        """
        raise NotImplementedError("tzinfo subclass must override dst()")

    def fromutc(self, dt):
        """datetime in UTC -> datetime in local time."""
        if not isinstance(dt, real_datetime):
            raise TypeError("fromutc() requires a datetime argument")
        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")

        dtoff = dt.utcoffset()
        if dtoff is None:
            raise ValueError("fromutc() requires a non-None utcoffset() " "result")

        # See the long comment block at the end of this file for an
        # explanation of this algorithm.
        dtdst = dt.dst()
        if dtdst is None:
            raise ValueError("fromutc() requires a non-None dst() result")
        delta = dtoff - dtdst
        if delta:
            dt += delta
            dtdst = dt.dst()
            if dtdst is None:
                raise ValueError(
                    "fromutc(): dt.dst gave inconsistent " "results; cannot convert"
                )
        return dt + dtdst

    def __reduce__(self):
        getinitargs = getattr(self, "__getinitargs__", None)
        if getinitargs:
            args = getinitargs()
        else:
            args = ()
        getstate = getattr(self, "__getstate__", None)
        if getstate:
            state = getstate()
        else:
            state = getattr(self, "__dict__", None) or None
        if state is None:
            return (type(self), args)
        else:
            return (type(self), args, state)

    def __ch_pytype__(self):
        return real_tzinfo


class IsoCalendarDate(tuple):
    def __new__(cls, year, week, weekday):
        return super().__new__(cls, (year, week, weekday))

    @property
    def year(self):
        return self[0]

    @property
    def week(self):
        return self[1]

    @property
    def weekday(self):
        return self[2]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(year={self[0]}, week={self[1]}, weekday={self[2]})"
        )


_IsoCalendarDate = IsoCalendarDate
del IsoCalendarDate
_tzinfo_class = tzinfo


def _time_getstate(self):
    us2, us3 = divmod(self.microsecond, 256)
    us1, us2 = divmod(us2, 256)
    h = self.hour
    basestate = bytes([h, self.minute, self.second, us1, us2, us3])
    return (basestate,)


class time:
    """
    Time with time zone.

    Constructors
    ------------
    __new__()

    Operators
    ---------
    __repr__, __str__
    __eq__, __le__, __lt__, __ge__, __gt__, __hash__

    Methods
    -------
    strftime()
    isoformat()
    utcoffset()
    tzname()
    dst()

    Properties (readonly):
    ---------------------
    hour, minute, second, microsecond, tzinfo, fold

    """

    def __new__(cls, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, fold=0):
        hour, minute, second, microsecond, fold = _check_time_fields(
            hour, minute, second, microsecond, fold
        )
        _check_tzinfo_arg(tzinfo)
        self = object.__new__(cls)
        self._hour = hour
        self._minute = minute
        self._second = second
        self._microsecond = microsecond
        self._tzinfo = tzinfo
        self._hashcode = -1
        self._fold = fold
        return self

    # Read-only field accessors
    @property
    def hour(self):
        """hour (0-23)"""
        return self._hour

    @property
    def minute(self):
        """minute (0-59)"""
        return self._minute

    @property
    def second(self):
        """second (0-59)"""
        return self._second

    @property
    def microsecond(self):
        """microsecond (0-999999)"""
        return self._microsecond

    @property
    def tzinfo(self):
        """timezone info object"""
        return self._tzinfo

    @property
    def fold(self):
        return self._fold

    # Standard conversions, __hash__ (and helpers)

    # Comparisons of time objects with other.

    def __eq__(self, other):
        if isinstance(other, real_time):
            return self._cmp(other, allow_mixed=True) == 0
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, real_time):
            return self._cmp(other) <= 0
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, real_time):
            return self._cmp(other) < 0
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, real_time):
            return self._cmp(other) >= 0
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, real_time):
            return self._cmp(other) > 0
        else:
            return NotImplemented

    def _cmp(self, other, allow_mixed=False):
        assert isinstance(other, real_time)
        mytz = self._tzinfo
        ottz = other.tzinfo
        myoff = otoff = None

        if mytz is ottz:
            base_compare = True
        else:
            myoff = self.utcoffset()
            otoff = other.utcoffset()
            base_compare = myoff == otoff

        if base_compare:
            return _cmp(
                (self._hour, self._minute, self._second, self._microsecond),
                (other.hour, other.minute, other.second, other.microsecond),
            )
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2  # arbitrary non-zero value
            else:
                raise TypeError("cannot compare naive and aware times")
        myhhmm = self._hour * 60 + self._minute - myoff // timedelta(minutes=1)
        othhmm = other.hour * 60 + other.minute - otoff // timedelta(minutes=1)
        return _cmp(
            (myhhmm, self._second, self._microsecond),
            (othhmm, other.second, other.microsecond),
        )

    def __hash__(self):
        """Hash."""
        if self._hashcode == -1:
            if self.fold:
                t = self.replace(fold=0)
            else:
                t = self
            tzoff = t.utcoffset()
            if not tzoff:  # zero or None
                self._hashcode = hash(_time_getstate(t)[0])
            else:
                h, m = divmod(
                    timedelta(hours=self.hour, minutes=self.minute) - tzoff,
                    timedelta(hours=1),
                )
                assert not m % timedelta(minutes=1), "whole minute"
                m //= timedelta(minutes=1)
                if 0 <= h < 24:
                    self._hashcode = hash(time(h, m, self.second, self.microsecond))
                else:
                    self._hashcode = hash((h, m, self.second, self.microsecond))
        return self._hashcode

    # Conversion to string

    def _tzstr(self):
        """Return formatted timezone offset (+xx:xx) or an empty string."""
        off = self.utcoffset()
        return _format_offset(off)

    def __repr__(self):
        """Convert to formal string, for repr()."""
        if self._microsecond != 0:
            s = ", %d, %d" % (self._second, self._microsecond)
        elif self._second != 0:
            s = ", %d" % self._second
        else:
            s = ""
        s = "%s.%s(%d, %d%s)" % (
            type(self).__module__,
            self.__class__.__qualname__,
            self._hour,
            self._minute,
            s,
        )
        if self._tzinfo is not None:
            assert s[-1:] == ")"
            s = s[:-1] + ", tzinfo=%r" % self._tzinfo + ")"
        if self._fold:
            assert s[-1:] == ")"
            s = s[:-1] + ", fold=1)"
        return s

    def isoformat(self, timespec="auto"):
        """
        Return the time formatted according to ISO.

        The full format is 'HH:MM:SS.mmmmmm+zz:zz'. By default, the fractional
        part is omitted if self.microsecond == 0.

        The optional argument timespec specifies the number of additional
        terms of the time to include. Valid options are 'auto', 'hours',
        'minutes', 'seconds', 'milliseconds' and 'microseconds'.
        """
        s = _format_time(
            self._hour, self._minute, self._second, self._microsecond, timespec
        )
        tz = self._tzstr()
        if tz:
            s += tz
        return s

    __str__ = isoformat

    @classmethod
    def fromisoformat(cls, time_string):
        """Construct a time from the output of isoformat()."""
        if not isinstance(time_string, str):
            raise TypeError("fromisoformat: argument must be str")

        try:
            return cls(*_parse_isoformat_time(time_string))
        except Exception:
            raise ValueError(f"Invalid isoformat string")

    def strftime(self, fmt):
        """
        Format using strftime().  The date part of the timestamp passed
        to underlying strftime should not be used.
        """
        # The year must be >= 1000 else Python's strftime implementation
        # can raise a bogus exception.
        timetuple = (1900, 1, 1, self._hour, self._minute, self._second, 0, 1, -1)
        return _wrap_strftime(self, fmt, timetuple)

    def __format__(self, fmt):
        if not isinstance(fmt, str):
            raise TypeError("must be str, not %s" % type(fmt).__name__)
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)

    # Timezone functions

    def utcoffset(self):
        """
        Return the timezone offset as timedelta, positive east of UTC
        (negative west of UTC).
        """
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.utcoffset(None)
        _check_utc_offset("utcoffset", offset)
        return offset

    def tzname(self):
        """
        Return the timezone name.

        Note that the name is 100% informational -- there's no requirement that
        it mean anything in particular. For example, "GMT", "UTC", "-500",
        "-5:00", "EDT", "US/Eastern", "America/New York" are all valid replies.
        """
        if self._tzinfo is None:
            return None
        name = self._tzinfo.tzname(None)
        _check_tzname(name)
        return name

    def dst(self):
        """
        Return 0 if DST is not in effect, or the DST offset (as timedelta
        positive eastward) if DST is in effect.

        This is purely informational; the DST offset has already been added to
        the UTC offset returned by utcoffset() if applicable, so there's no
        need to consult dst() unless you're interested in displaying the DST
        info.
        """
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.dst(None)
        _check_utc_offset("dst", offset)
        return offset

    def replace(
        self,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        tzinfo=True,
        *,
        fold=None,
    ):
        """Return a new time with new values for the specified fields."""
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        if tzinfo is True:
            tzinfo = self.tzinfo
        if fold is None:
            fold = self._fold
        return time(hour, minute, second, microsecond, tzinfo, fold=fold)

    def __ch_realize__(self):
        return real_time(
            realize(self._hour),
            realize(self._minute),
            realize(self._second),
            realize(self._microsecond),
            realize(self._tzinfo),
            fold=realize(self._fold),
        )

    def __ch_pytype__(self):
        return real_time


_time_class = time  # so functions w/ args named "time" can get at the class

time.min = time(0, 0, 0)  # type: ignore
time.max = time(23, 59, 59, 999999)  # type: ignore
time.resolution = timedelta(microseconds=1)  # type: ignore


def _datetime_getstate(self):
    yhi, ylo = divmod(self.year, 256)
    us2, us3 = divmod(self.microsecond, 256)
    us1, us2 = divmod(us2, 256)
    m = self._month
    basestate = bytes(
        [yhi, ylo, m, self._day, self.hour, self.minute, self.second, us1, us2, us3]
    )
    return (basestate,)


class datetime(date):
    """
    datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])

    The year, month and day arguments are required. tzinfo may be None, or an
    instance of a tzinfo subclass. The remaining arguments may be ints.
    """

    def __init__(
        self,
        year,
        month,
        day=None,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=None,
        *,
        fold=0,
    ):
        year, month, day = _check_date_fields(year, month, day)
        hour, minute, second, microsecond, fold = _check_time_fields(
            hour, minute, second, microsecond, fold
        )
        _check_tzinfo_arg(tzinfo)
        date.__init__(self, year, month, day)
        self._hour = hour
        self._minute = minute
        self._second = second
        self._microsecond = microsecond
        self._tzinfo = tzinfo
        self._hashcode = -1
        self._fold = fold

    # Read-only field accessors
    @property
    def hour(self):
        """hour (0-23)"""
        return self._hour

    @property
    def minute(self):
        """minute (0-59)"""
        return self._minute

    @property
    def second(self):
        """second (0-59)"""
        return self._second

    @property
    def microsecond(self):
        """microsecond (0-999999)"""
        return self._microsecond

    @property
    def tzinfo(self):
        """timezone info object"""
        return self._tzinfo

    @property
    def fold(self):
        return self._fold

    @classmethod
    def _fromtimestamp(cls, t, utc, tz):
        """
        Construct a datetime from a POSIX timestamp (like time.time()).

        A timezone info object may be passed in as well.
        """
        frac, t = _math.modf(t)
        us = round(frac * 1e6)
        if us >= 1000000:
            t += 1
            us -= 1000000
        elif us < 0:
            t -= 1
            us += 1000000

        converter = _time.gmtime if utc else _time.localtime
        y, m, d, hh, mm, ss, weekday, jday, dst = converter(t)
        ss = min(ss, 59)  # clamp out leap seconds if the platform has them
        result = cls(y, m, d, hh, mm, ss, us, tz)
        if tz is None:
            # As of version 2015f max fold in IANA database is
            # 23 hours at 1969-09-30 13:00:00 in Kwajalein.
            # Let's probe 24 hours in the past to detect a transition:
            max_fold_seconds = 24 * 3600

            # On Windows localtime_s throws an OSError for negative values,
            # thus we can't perform fold detection for values of time less
            # than the max time fold. See comments in _datetimemodule's
            # version of this method for more details.
            if t < max_fold_seconds and sys.platform.startswith("win"):
                return result

            y, m, d, hh, mm, ss = converter(t - max_fold_seconds)[:6]
            probe1 = cls(y, m, d, hh, mm, ss, us, tz)
            trans = result - probe1 - timedelta(0, max_fold_seconds)
            if trans.days < 0:
                y, m, d, hh, mm, ss = converter(t + trans // timedelta(0, 1))[:6]
                probe2 = cls(y, m, d, hh, mm, ss, us, tz)
                if probe2 == result:
                    result._fold = 1
        else:
            result = tz.fromutc(result)
        return result

    @classmethod
    def fromtimestamp(cls, t, tz=None):
        """
        Construct a datetime from a POSIX timestamp (like time.time()).

        A timezone info object may be passed in as well.
        """
        _check_tzinfo_arg(tz)

        return cls._fromtimestamp(t, tz is not None, tz)

    @classmethod
    def utcfromtimestamp(cls, t):
        """Construct a naive UTC datetime from a POSIX timestamp."""
        return cls._fromtimestamp(t, True, None)

    @classmethod
    def now(cls, tz=None):
        """Construct a datetime from time.time() and optional time zone info."""
        t = _time.time()
        return cls.fromtimestamp(t, tz)

    @classmethod
    def utcnow(cls):
        """Construct a UTC datetime from time.time()."""
        t = _time.time()
        return cls.utcfromtimestamp(t)

    @classmethod
    def combine(cls, date, time, tzinfo=True):
        """Construct a datetime from a given date and a given time."""
        if not isinstance(date, real_date):
            raise TypeError("date argument must be a date instance")
        if not isinstance(time, real_time):
            raise TypeError("time argument must be a time instance")
        if tzinfo is True:
            tzinfo = time.tzinfo
        return cls(
            date.year,
            date.month,
            date.day,
            time.hour,
            time.minute,
            time.second,
            time.microsecond,
            tzinfo,
            fold=time.fold,
        )

    @classmethod
    def fromisoformat(cls, date_string):
        """Construct a datetime from the output of datetime.isoformat()."""
        if not isinstance(date_string, str):
            raise TypeError("fromisoformat: argument must be str")

        # Split this at the separator
        dstr = date_string[0:10]
        tstr = date_string[11:]

        try:
            date_components = _parse_isoformat_date(dstr)
        except ValueError:
            raise ValueError(f"Invalid isoformat string")

        if tstr:
            try:
                time_components = _parse_isoformat_time(tstr)
            except ValueError:
                raise ValueError(f"Invalid isoformat string")
        else:
            time_components = [0, 0, 0, 0, None]

        return cls(*(date_components + time_components))

    @classmethod
    def fromdatetime(cls, d: real_datetime):
        return cls(
            d.year,
            d.month,
            d.day,
            d.hour,
            d.minute,
            d.second,
            d.microsecond,
            d.tzinfo,
            fold=d.fold,
        )

    def timetuple(self):
        """Return local time tuple compatible with time.localtime()."""
        dst = self.dst()
        if dst is None:
            dst = -1
        elif dst:
            dst = 1
        else:
            dst = 0
        return _build_struct_time(
            self.year, self.month, self.day, self.hour, self.minute, self.second, dst
        )

    def _mktime(self):
        """Return integer POSIX timestamp."""
        epoch = datetime(1970, 1, 1)
        max_fold_seconds = 24 * 3600
        t = (self - epoch) // timedelta(0, 1)

        def local(u):
            y, m, d, hh, mm, ss = _time.localtime(u)[:6]
            return (datetime(y, m, d, hh, mm, ss) - epoch) // timedelta(0, 1)

        # Our goal is to solve t = local(u) for u.
        a = local(t) - t
        u1 = t - a
        t1 = local(u1)
        if t1 == t:
            # We found one solution, but it may not be the one we need.
            # Look for an earlier solution (if `fold` is 0), or a
            # later one (if `fold` is 1).
            u2 = u1 + (-max_fold_seconds, max_fold_seconds)[self.fold]
            b = local(u2) - u2
            if a == b:
                return u1
        else:
            b = t1 - u1
            assert a != b
        u2 = t - b
        t2 = local(u2)
        if t2 == t:
            return u2
        if t1 == t:
            return u1
        # We have found both offsets a and b, but neither t - a nor t - b is
        # a solution.  This means t is in the gap.
        return (max, min)[self.fold](u1, u2)

    def timestamp(self):
        """Return POSIX timestamp as float"""
        if self._tzinfo is None:
            s = self._mktime()
            return s + self.microsecond / 1e6
        else:
            return (self - _EPOCH).total_seconds()

    def utctimetuple(self):
        """Return UTC time tuple compatible with time.gmtime()."""
        offset = self.utcoffset()
        if offset:
            self -= offset
        y, m, d = self.year, self.month, self.day
        hh, mm, ss = self.hour, self.minute, self.second
        return _build_struct_time(y, m, d, hh, mm, ss, 0)

    def date(self):
        """Return the date part."""
        return date(self._year, self._month, self._day)

    def time(self):
        """Return the time part, with tzinfo None."""
        return time(
            self.hour, self.minute, self.second, self.microsecond, fold=self.fold
        )

    def timetz(self):
        """Return the time part, with same tzinfo."""
        return time(
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            self._tzinfo,
            fold=self.fold,
        )

    def replace(
        self,
        year=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        microsecond=None,
        tzinfo=True,
        *,
        fold=None,
    ):
        """Return a new datetime with new values for the specified fields."""
        if year is None:
            year = self.year
        if month is None:
            month = self.month
        if day is None:
            day = self.day
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        if tzinfo is True:
            tzinfo = self.tzinfo
        if fold is None:
            fold = self.fold
        return datetime(
            year, month, day, hour, minute, second, microsecond, tzinfo, fold=fold
        )

    def _local_timezone(self):
        if self.tzinfo is None:
            ts = self._mktime()
        else:
            ts = (self - _EPOCH) // timedelta(seconds=1)
        localtm = _time.localtime(ts)
        local = datetime(*localtm[:6])
        # Extract TZ data
        gmtoff = localtm.tm_gmtoff
        zone = localtm.tm_zone
        return timezone(timedelta(seconds=gmtoff), zone)

    def astimezone(self, tz=None):
        if tz is None:
            tz = self._local_timezone()
        elif not isinstance(tz, real_tzinfo):
            raise TypeError("tz argument must be an instance of tzinfo")

        mytz = self.tzinfo
        if mytz is None:
            mytz = self._local_timezone()
            myoffset = mytz.utcoffset(self)
        else:
            myoffset = mytz.utcoffset(self)
            if myoffset is None:
                mytz = self.replace(tzinfo=None)._local_timezone()
                myoffset = mytz.utcoffset(self)

        if tz is mytz:
            return self

        # Convert self to UTC, and attach the new time zone object.
        utc = (self - myoffset).replace(tzinfo=tz)

        # Convert from UTC to tz's local time.
        return tz.fromutc(utc)

    # Ways to produce a string.

    def ctime(self):
        """Return ctime() style string."""
        weekday = self.toordinal() % 7 or 7
        return "%s %s %2d %02d:%02d:%02d %04d" % (
            _DAYNAMES[weekday],
            _MONTHNAMES[self._month],
            self._day,
            self._hour,
            self._minute,
            self._second,
            self._year,
        )

    def isoformat(self, sep="T", timespec="auto"):
        """
        Return the time formatted according to ISO.

        The full format looks like 'YYYY-MM-DD HH:MM:SS.mmmmmm'.
        By default, the fractional part is omitted if self.microsecond == 0.

        If self.tzinfo is not None, the UTC offset is also attached, giving
        giving a full format of 'YYYY-MM-DD HH:MM:SS.mmmmmm+HH:MM'.

        Optional argument sep specifies the separator between date and
        time, default 'T'.

        The optional argument timespec specifies the number of additional
        terms of the time to include. Valid options are 'auto', 'hours',
        'minutes', 'seconds', 'milliseconds' and 'microseconds'.
        """
        s = "%04d-%02d-%02d%c" % (
            self._year,
            self._month,
            self._day,
            sep,
        ) + _format_time(
            self._hour, self._minute, self._second, self._microsecond, timespec
        )

        off = self.utcoffset()
        tz = _format_offset(off)
        if tz:
            s += tz

        return s

    def __repr__(self):
        """Convert to formal string, for repr()."""
        L = [
            self._year,
            self._month,
            self._day,  # These are never zero
            self._hour,
            self._minute,
            self._second,
            self._microsecond,
        ]
        if L[-1] == 0:
            del L[-1]
        if L[-1] == 0:
            del L[-1]
        s = "%s.%s(%s)" % (
            type(self).__module__,
            self.__class__.__qualname__,
            ", ".join(map(str, L)),
        )
        if self._tzinfo is not None:
            assert s[-1:] == ")"
            s = s[:-1] + ", tzinfo=%r" % self._tzinfo + ")"
        if self._fold:
            assert s[-1:] == ")"
            s = s[:-1] + ", fold=1)"
        return s

    def __str__(self):
        """Convert to string, for str()."""
        return self.isoformat(sep=" ")

    @classmethod
    def strptime(cls, date_string, format):
        """string, format -> new datetime parsed from a string (like time.strptime())."""
        import _strptime  # type: ignore

        return _strptime._strptime_datetime(cls, date_string, format)

    def _realized_if_concrete_tzinfo(self):
        with NoTracing():
            if isinstance(self._tzinfo, real_tzinfo):
                return realize(self)
            return self

    def utcoffset(self):
        """
        Return the timezone offset as timedelta positive east of UTC (negative west of
        UTC).
        """
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.utcoffset(self._realized_if_concrete_tzinfo())
        _check_utc_offset("utcoffset", offset)
        return offset

    def tzname(self):
        """
        Return the timezone name.

        Note that the name is 100% informational -- there's no requirement that
        it mean anything in particular. For example, "GMT", "UTC", "-500",
        "-5:00", "EDT", "US/Eastern", "America/New York" are all valid replies.
        """
        if self._tzinfo is None:
            return None
        name = self._tzinfo.tzname(self._realized_if_concrete_tzinfo())
        _check_tzname(name)
        return name

    def dst(self):
        """
        Return 0 if DST is not in effect, or the DST offset (as timedelta
        positive eastward) if DST is in effect.

        This is purely informational; the DST offset has already been added to
        the UTC offset returned by utcoffset() if applicable, so there's no
        need to consult dst() unless you're interested in displaying the DST
        info.
        """
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.dst(self._realized_if_concrete_tzinfo())
        _check_utc_offset("dst", offset)
        return offset

    # Comparisons of datetime objects with other.

    def __eq__(self, other):
        if isinstance(other, real_datetime):
            return self._cmp(other, allow_mixed=True) == 0
        elif not isinstance(other, real_date):
            return NotImplemented
        else:
            return False

    def __le__(self, other):
        if isinstance(other, real_datetime):
            return self._cmp(other) <= 0
        elif not isinstance(other, real_date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __lt__(self, other):
        if isinstance(other, real_datetime):
            return self._cmp(other) < 0
        elif not isinstance(other, real_date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __ge__(self, other):
        if isinstance(other, real_datetime):
            return self._cmp(other) >= 0
        elif not isinstance(other, real_date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __gt__(self, other):
        if isinstance(other, real_datetime):
            return self._cmp(other) > 0
        elif not isinstance(other, real_date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def _cmp(self, other, allow_mixed=False):
        assert isinstance(other, real_datetime)
        mytz = self._tzinfo
        ottz = other.tzinfo
        myoff = otoff = None

        if mytz is ottz:
            base_compare = True
        else:
            myoff = self.utcoffset()
            otoff = other.utcoffset()
            # Assume that allow_mixed means that we are called from __eq__
            if allow_mixed:
                if myoff != self.replace(fold=not self.fold).utcoffset():
                    return 2
                if otoff != other.replace(fold=not other.fold).utcoffset():
                    return 2
            base_compare = myoff == otoff

        if base_compare:
            return _cmp(
                (
                    self._year,
                    self._month,
                    self._day,
                    self._hour,
                    self._minute,
                    self._second,
                    self._microsecond,
                ),
                (
                    other.year,
                    other.month,
                    other.day,
                    other.hour,
                    other.minute,
                    other.second,
                    other.microsecond,
                ),
            )
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2  # arbitrary non-zero value
            else:
                raise TypeError("cannot compare naive and aware datetimes")
        # XXX What follows could be done more efficiently...
        diff = self - other  # this will take offsets into account
        if diff.days < 0:
            return -1
        return diff and 1 or 0

    def __add__(self, other):
        """Add a datetime and a timedelta."""
        if not isinstance(other, real_timedelta):
            return NotImplemented
        delta = timedelta(
            self.toordinal(),
            hours=self._hour,
            minutes=self._minute,
            seconds=self._second,
            microseconds=self._microsecond,
        )
        delta += other
        hour, rem = divmod(delta.seconds, 3600)
        minute, second = divmod(rem, 60)
        if 0 < delta.days <= _MAXORDINAL:
            return datetime.combine(
                date.fromordinal(delta.days),
                time(hour, minute, second, delta.microseconds, tzinfo=self._tzinfo),
            )
        raise OverflowError("result out of range")

    __radd__ = __add__

    def __sub__(self, other):
        """Subtract two datetimes, or a datetime and a timedelta."""
        if not isinstance(other, real_datetime):
            if isinstance(other, real_timedelta):
                return self + -other
            return NotImplemented

        days1 = self.toordinal()
        days2 = other.toordinal()
        secs1 = self._second + self._minute * 60 + self._hour * 3600
        secs2 = other.second + other.minute * 60 + other.hour * 3600
        base = timedelta(
            days1 - days2, secs1 - secs2, self._microsecond - other.microsecond
        )
        if self._tzinfo is other.tzinfo:
            return base
        myoff = self.utcoffset()
        otoff = other.utcoffset()
        if myoff == otoff:
            return base
        if myoff is None or otoff is None:
            raise TypeError("cannot mix naive and timezone-aware time")
        return base + otoff - myoff

    def __hash__(self):
        if self._hashcode == -1:
            if self.fold:
                t = self.replace(fold=0)
            else:
                t = self
            tzoff = t.utcoffset()
            if tzoff is None:
                self._hashcode = hash(_datetime_getstate(t)[0])
            else:
                days = _ymd2ord(self.year, self.month, self.day)
                seconds = self.hour * 3600 + self.minute * 60 + self.second
                self._hashcode = hash(
                    timedelta(days, seconds, self.microsecond) - tzoff
                )
        return self._hashcode

    def __ch_realize__(self):
        return real_datetime(
            realize(self._year),
            realize(self._month),
            realize(self._day),
            realize(self._hour),
            realize(self._minute),
            realize(self._second),
            realize(self.microsecond),
            realize(self._tzinfo),
            fold=realize(self._fold),
        )

    def __ch_pytype__(self):
        return real_datetime


datetime.min = datetime(1, 1, 1)  # type: ignore
datetime.max = datetime(9999, 12, 31, 23, 59, 59, 999999)  # type: ignore
datetime.resolution = timedelta(microseconds=1)  # type: ignore


def _isoweek1monday(year):
    # Helper to calculate the day number of the Monday starting week 1
    # XXX This could be done more efficiently
    THURSDAY = 3
    firstday = _ymd2ord(year, 1, 1)
    firstweekday = (firstday + 6) % 7  # See weekday() above
    week1monday = firstday - firstweekday
    if firstweekday > THURSDAY:
        week1monday += 7
    return week1monday


class timezone(tzinfo):
    class Omitted(Enum):
        value = 0

    _Omitted = Omitted.value

    def __new__(cls, offset: real_timedelta, name: Union[str, Omitted] = _Omitted):
        if not isinstance(offset, real_timedelta):
            raise TypeError("offset must be a timedelta")
        if name == cls._Omitted:
            if not offset:
                return cls.utc  # type: ignore
            name = None  # type: ignore
        elif not isinstance(name, str):
            raise TypeError("name must be a string")
        if not cls._minoffset <= offset <= cls._maxoffset:
            raise ValueError(
                "offset must be a timedelta "
                "strictly between -timedelta(hours=24) and "
                "timedelta(hours=24)."
            )
        return cls._create(offset, name)

    @classmethod
    def _create(cls, offset, name=None):
        self = tzinfo.__new__(cls)
        self._offset = offset
        self._name = name
        return self

    def __getinitargs__(self):
        """pickle support"""
        if self._name is None:
            return (self._offset,)
        return (self._offset, self._name)

    def __eq__(self, other):
        if isinstance(other, real_timezone):
            return self.utcoffset(None) == other.utcoffset(None)
        return NotImplemented

    def __hash__(self):
        return hash(self._offset)

    def __repr__(self):
        if self is self.utc:
            return "datetime.timezone.utc"
        if self._name is None:
            return "%s.%s(%r)" % (
                type(self).__module__,
                self.__class__.__qualname__,
                self._offset,
            )
        return "%s.%s(%r, %r)" % (
            type(self).__module__,
            self.__class__.__qualname__,
            self._offset,
            self._name,
        )

    def __str__(self):
        return self.tzname(None)

    def utcoffset(self, dt):
        if isinstance(dt, real_datetime) or dt is None:
            return self._offset
        raise TypeError("utcoffset() argument must be a datetime instance" " or None")

    def tzname(self, dt):
        if isinstance(dt, real_datetime) or dt is None:
            if self._name is None:
                return self._name_from_offset(self._offset)
            return self._name
        raise TypeError("tzname() argument must be a datetime instance" " or None")

    def dst(self, dt):
        if isinstance(dt, real_datetime) or dt is None:
            return None
        raise TypeError("dst() argument must be a datetime instance" " or None")

    def fromutc(self, dt):
        if isinstance(dt, real_datetime):
            if dt.tzinfo is not self:
                raise ValueError("fromutc: dt.tzinfo " "is not self")
            return dt + self._offset
        raise TypeError("fromutc() argument must be a datetime instance" " or None")

    _maxoffset = (
        timedelta(hours=24, microseconds=-1)
        if sys.version_info >= (3, 8)
        else timedelta(hours=23, minutes=59)
    )
    _minoffset = -_maxoffset

    @staticmethod
    def _name_from_offset(delta):
        if not delta:
            return "UTC"
        if delta < timedelta(0):
            sign = "-"
            delta = -delta
        else:
            sign = "+"
        hours, rest = divmod(delta, timedelta(hours=1))
        minutes, rest = divmod(rest, timedelta(minutes=1))
        seconds = rest.seconds
        microseconds = rest.microseconds
        if microseconds:
            return (
                f"UTC{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"
                f".{microseconds:06d}"
            )
        if seconds:
            return f"UTC{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"UTC{sign}{hours:02d}:{minutes:02d}"

    def __ch_realize__(self):
        offset = realize(self._offset)
        name = realize(self._name)
        return real_timezone(offset) if name is None else real_timezone(offset, name)

    def __ch_pytype__(self):
        return real_timezone


timezone.utc = timezone._create(timedelta(0))  # type: ignore
# bpo-37642: These attributes are rounded to the nearest minute for backwards
# compatibility, even though the constructor will accept a wider range of
# values. This may change in the future.
timezone.min = timezone._create(-timedelta(hours=23, minutes=59))  # type: ignore
timezone.max = timezone._create(timedelta(hours=23, minutes=59))  # type: ignore
_EPOCH = datetime(1970, 1, 1, tzinfo=real_timezone.utc)  # type: ignore


def _raises_value_error(fn, args):
    try:
        fn(*args)
        return False
    except ValueError:
        return True


def _timedelta_skip_construct(days, seconds, microseconds):
    # timedelta's constructor is convoluted to guide C implementations.
    # We use something simpler, and just ensure elsewhere that the inputs fall in the right ranges:
    delta = timedelta()
    delta._days = days  # type: ignore
    delta._seconds = seconds  # type: ignore
    delta._microseconds = microseconds  # type: ignore
    return delta


def _time_skip_construct(hour, minute, second, microsecond, tzinfo, fold):
    tm = time()
    tm._hour = hour
    tm._minute = minute
    tm._second = second
    tm._microsecond = microsecond
    tm._tzinfo = tzinfo
    tm._fold = fold
    return tm


def _date_skip_construct(year, month, day):
    dt = date(2020, 1, 1)
    dt._year = year
    dt._month = month
    dt._day = day
    return dt


def _datetime_skip_construct(
    year, month, day, hour, minute, second, microsecond, tzinfo
):
    dt = datetime(2020, 1, 1)
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
    year = make_bounded_int(varname + "_year", MINYEAR, MAXYEAR)
    month = make_bounded_int(varname + "_month", 1, 12)
    day = make_bounded_int(varname + "_day", 1, 31)
    context_statespace().add(_smt_days_in_month(year.var, month.var, day.var))
    return (year, month, day)


def _symbolic_time_fields(varname: str) -> Tuple:
    return (
        make_bounded_int(varname + "_hour", 0, 23),
        make_bounded_int(varname + "_min", 0, 59),
        make_bounded_int(varname + "_sec", 0, 59),
        make_bounded_int(varname + "_usec", 0, 999999),
        make_bounded_int(varname + "_fold", 0, 1),
    )


def make_registrations():

    # TODO: `timezone` never makes a tzinfo with DST, so this is incomplete.
    # A complete solution would require generating a symbolc dst() member function.
    register_type(real_tzinfo, lambda p: p(timezone))

    def make_timezone(p: Any) -> timezone:
        if p.space.smt_fork(desc="use explicit timezone"):
            delta = p(timedelta, "_offset")
            with ResumedTracing():
                if timezone._minoffset < delta < timezone._maxoffset:
                    return timezone(delta, realize(p(str, "_name")))
                else:
                    raise IgnoreAttempt("Invalid timezone offset")
        else:
            return timezone.utc  # type: ignore

    register_type(real_timezone, make_timezone)
    register_patch(real_timezone, lambda *a, **kw: timezone(*a, **kw))

    def make_date(p: Any) -> date:
        year, month, day = _symbolic_date_fields(p.varname)
        return _date_skip_construct(year, month, day)

    register_type(real_date, make_date)
    register_patch(real_date, lambda *a, **kw: date(*a, **kw))

    def make_time(p: Any) -> time:
        (hour, minute, sec, usec, fold) = _symbolic_time_fields(p.varname)
        tzinfo = p(Optional[timezone], "_tzinfo")
        return _time_skip_construct(hour, minute, sec, usec, tzinfo, fold)

    register_type(real_time, make_time)
    register_patch(real_time, lambda *a, **kw: time(*a, **kw))

    def make_datetime(p: Any) -> datetime:
        year, month, day = _symbolic_date_fields(p.varname)
        (hour, minute, sec, usec, fold) = _symbolic_time_fields(p.varname)
        tzinfo = p(Optional[timezone], "_tzinfo")
        return _datetime_skip_construct(
            year, month, day, hour, minute, sec, usec, tzinfo
        )

    register_type(real_datetime, make_datetime)
    register_patch(real_datetime, lambda *a, **kw: datetime(*a, **kw))

    def make_timedelta(p: SymbolicFactory) -> timedelta:
        microseconds = make_bounded_int(p.varname + "_usec", 0, 999999)
        seconds = make_bounded_int(p.varname + "_sec", 0, 3600 * 24 - 1)
        days = make_bounded_int(p.varname + "_days", -999999999, 999999999)
        return _timedelta_skip_construct(days, seconds, microseconds)

    register_type(real_timedelta, make_timedelta)
    register_patch(real_timedelta, lambda *a, **kw: timedelta(*a, **kw))
