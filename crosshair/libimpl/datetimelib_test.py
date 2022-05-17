import datetime
import sys
import unittest

from crosshair.options import AnalysisOptionSet
from crosshair.test_util import check_exec_err, check_fail, check_unknown
from crosshair.util import set_debug

_SLOW_TEST = AnalysisOptionSet(per_condition_timeout=10, per_path_timeout=5)


class DatetimeLibTests(unittest.TestCase):
    def test_symbolic_months_fail(self) -> None:
        def f(num_months: int) -> datetime.date:
            """
            pre: 0 <= num_months <= 100
            post: _.year != 2003
            """
            dt = datetime.date(2000, 1, 1)
            return dt + datetime.timedelta(days=30 * num_months)

        self.assertEqual(*check_fail(f, _SLOW_TEST))

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

        self.assertEqual(*check_fail(f, AnalysisOptionSet(max_iterations=60)))

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

        self.assertEqual(*check_fail(f, _SLOW_TEST))

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
