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
            '''
            pre: 0 <= num_months <= 100
            post: _.year != 2003
            '''
            dt = datetime.date(2000, 1, 1)
            return dt + datetime.timedelta(days=30 * num_months)
        self.assertEqual(*check_fail(f))

    def test_date_fail(self) -> None:
        def f(dt: datetime.date) -> int:
            '''
            post: _ != 2020
            '''
            return dt.year
        self.assertEqual(*check_fail(f))

    def TODO_test_date_plus_delta_fail(self) -> None:
        # TODO: getting unexpected isinstance(symbolic_int, int) == false
        def f(delta: datetime.timedelta) -> datetime.date:
            '''
            post: _.year != 2001
            '''
            return datetime.date(2000, 1, 1) + delta
        self.assertEqual(*check_fail(f))


if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
