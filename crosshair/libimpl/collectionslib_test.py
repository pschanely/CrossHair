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

class CollectionsLibTests(unittest.TestCase):
    # Remove these TODO prefixes to enable the tests:
    def TODO_test_deque_len_ok(self) -> None:
        def f(l: Deque[int]) -> Deque[int]:
            '''
            post: len(_) == len(__old__.l) + 1
            '''
            l.append(42)
            return l
        # Right now, this returns unknown rather than ok, because creation of
        # the deque forces the list length to become concrete.
        self.assertEqual(*check_ok(f))

    def TODO_test_deque_len_fail(self) -> None:
        def f(l: Deque[int]) -> Deque[int]:
            '''
            pre: len(l) > 0
            post: len(l) != 222
            '''
            return l
        # Right now, this returns unknown rather than fail, because creation of
        # the deque forces the list length to become concrete.
        self.assertEqual(*check_fail(f))

if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
