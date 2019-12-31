import random
import datetime
from crosshair import register_type, realize, IgnoreAttempt
from typing import Callable

def make_stdlib_registrations():
    register_type(random.Random, lambda p: random.Random(p(int)))

    def make_date(p: Callable) -> datetime.date:
        year, month, day = p(int), p(int), p(int)
        # Develop good symbolic constraints before realization:
        if not (1 <= year <= 9999 and 1 <= month <= 12 and 1 <= day <= 31):
            raise IgnoreAttempt('Invalid date')
        try:
            return datetime.date(realize(year), realize(month), realize(day))
        except ValueError:
            raise IgnoreAttempt('Invalid date')

    register_type(datetime.date, make_date)
