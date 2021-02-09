import datetime
import sys
import unittest
from typing import *

from crosshair.core_and_libs import *
from crosshair.test_util import check_ok
from crosshair.test_util import check_exec_err
from crosshair.test_util import check_post_err
from crosshair.test_util import check_fail
from crosshair.test_util import check_unknown
from crosshair.test_util import check_messages
from crosshair.util import set_debug


class DatetimeLibTests(unittest.TestCase):
    def test_symbolic_months_fail(self) -> None:
        def f(num_months: int) -> datetime.date:
            """
            pre: 0 <= num_months <= 100
            post: _.year != 2003
            """
            dt = datetime.date(2000, 1, 1)
            return dt + datetime.timedelta(days=30 * num_months)

        self.assertEqual(*check_fail(f, AnalysisOptions(per_path_timeout=10)))

    def test_date_fail(self) -> None:
        def f(dt: datetime.date) -> int:
            """
            post: _ != 2020
            """
            return dt.year

        self.assertEqual(*check_fail(f))

    def test_time_fail(self) -> None:
        def f(dt: datetime.time) -> int:
            """
            post: _ != 14
            """
            return dt.hour

        self.assertEqual(*check_fail(f))

    def test_datetime_fail(self) -> None:
        def f(dtime: datetime.datetime) -> int:
            """
            post: _ != 22
            """
            return dtime.second

        self.assertEqual(*check_fail(f, AnalysisOptions(max_iterations=60)))

    def test_timedelta_fail(self) -> None:
        def f(d: datetime.timedelta) -> int:
            """
            post: _ != 9
            """
            return d.seconds

        self.assertEqual(*check_fail(f))

    def test_date_plus_delta_unknown(self) -> None:
        def f(delta: datetime.timedelta) -> datetime.date:
            """
            post: _.year != -9999
            raises: OverflowError
            """
            return datetime.date(2000, 1, 1) + delta

        self.assertEqual(*check_unknown(f))

    def test_date_plus_delta_overflow_err(self) -> None:
        def f(delta: datetime.timedelta) -> datetime.date:
            """
            post: _.year != -9999
            """
            return datetime.date(2000, 1, 1) + delta

        self.assertEqual(*check_exec_err(f))

    def test_date_plus_delta_fail(self) -> None:
        def f(delta: datetime.timedelta) -> datetime.date:
            """
            post: _.year != 2001
            raises: OverflowError
            """
            return datetime.date(2000, 1, 1) + delta

        self.assertEqual(*check_fail(f))

    def TODO_test_leap_year(self) -> None:
        # The solver returns unknown when adding a delta to a symbolic date. (nonlinear I think)
        def f(start: datetime.date) -> datetime.date:
            """
            post: _.year == start.year + 1
            """
            return start + datetime.timedelta(days=365)

        self.assertEqual(*check_fail(f))


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
