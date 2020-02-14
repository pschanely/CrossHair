import random
import datetime
from crosshair import register_type, realize, IgnoreAttempt
from typing import Callable

def make_registrations():
    register_type(random.Random, lambda p: random.Random(p(int)))

    def make_date(p: Callable) -> datetime.date:
        year, month, day = p(int), p(int), p(int)
        # This condition isn't technically required, but it develops useful
        # symbolic inequalities before the realization below:
        if not (1 <= year <= 9999 and 1 <= month <= 12 and 1 <= day <= 31):
            raise IgnoreAttempt('Invalid date')
        try:
            return datetime.date(realize(year), realize(month), realize(day))
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
                days=realize(days), seconds=realize(seconds),
                microseconds=realize(microseconds))
        except OverflowError:
            raise IgnoreAttempt('Invalid timedelta')

    register_type(datetime.timedelta, make_timedelta)
