import sys
import unittest
from typing import *

from crosshair.core_and_libs import *
from crosshair.libimpl.collectionslib import ListBasedDeque
from crosshair.test_util import check_ok
from crosshair.test_util import check_exec_err
from crosshair.test_util import check_post_err
from crosshair.test_util import check_fail
from crosshair.test_util import check_unknown
from crosshair.test_util import check_messages
from crosshair.util import set_debug

class CollectionsLibTests(unittest.TestCase):

    def setUp(self):
        self.test_list = ListBasedDeque([1, 2, 3, 4, 5])

    def test_deque_appendleft(self) -> None:
        self.test_list.appendleft(0)
        self.assertEqual(self.test_list.popleft(), 0)

    def test_deque_appendleft_with_full_deque(self) -> None:
        temp_list = ListBasedDeque([1, 2, 3, 4, 5], maxlen=5)
        temp_list.appendleft(42)
        self.assertTrue(temp_list.popleft() == 42 and temp_list.pop() == 4)
    
    def test_deque_appendleft_doesnt_increase_size_with_maxlen(self) -> None:
        temp_list = ListBasedDeque([1, 2, 3, 4, 5], maxlen=5)
        temp_list.appendleft(42)
        self.assertTrue(len(temp_list) == temp_list.maxlen())

    def test_deque_append(self) -> None:
        self.test_list.append(0)
        self.assertEqual(self.test_list.pop(), 0)

    def test_deque_append_with_full_deque(self) -> None:
        temp_list = ListBasedDeque([1, 2, 3, 4, 5], maxlen=5)
        temp_list.append(42)
        self.assertTrue(temp_list.pop() == 42 and temp_list.popleft() == 2)
    
    def test_deque_append_doesnt_increase_size_with_maxlen(self) -> None:
        temp_list = ListBasedDeque([1, 2, 3, 4, 5], maxlen=5)
        temp_list.append(42)
        self.assertTrue(len(temp_list) == temp_list.maxlen())

    def test_deque_clear(self) -> None:
        self.test_list.clear()
        self.assertEqual(len(self.test_list), 0)

    def test_deque_count(self) -> None:
        self.test_list.append(1)
        count = self.test_list.count(1)
        self.assertEqual(count, 2)

    def test_deque_index(self) -> None:
        i = self.test_list.index(5)
        self.assertEqual(i, 4)

    def test_deque_index_with_start_index(self) -> None:
        i = self.test_list.index(5, start=1)
        self.assertEqual(i, 4)

    def test_deque_index_with_start_index_throws_correct_exception(self) -> None:
        with self.assertRaises(ValueError) as context:
            self.test_list.index(1, start=2)

        self.assertTrue('1 is not in list' in str(context.exception))
    
    def test_deque_index_with_start_and_end_index(self) -> None:
        i = self.test_list.index(2, start=0, end=3)
        self.assertEqual(i, 1)
    
    def test_deque_index_with_start_and_end_index_throws_correct_exception(self) -> None:
        with self.assertRaises(ValueError) as context:
            self.test_list.index(6, start=0, end=1)

        self.assertTrue('6 is not in list' in str(context.exception))

    def test_deque_insert(self) -> None:
        self.test_list.insert(index=1, item=42)
        self.assertEqual(self.test_list.index(42), 1)

    def test_deque_pop(self) -> None:
        self.assertEqual(self.test_list.pop(), 5)

    def test_deque_popleft(self) -> None:
        self.assertEqual(self.test_list.popleft(), 1)

    def test_deque_remove(self) -> None:
        original_length = len(self.test_list)
        self.test_list.remove(1)
        self.assertTrue(len(self.test_list) == original_length - 1)

    def test_deque_reverse(self) -> None:
        self.test_list.reverse()
        self.assertTrue(self.test_list.popleft() == 5)

    def test_deque_rotate(self) -> None:
        self.test_list.rotate(n=1)
        self.assertTrue(self.test_list.popleft() == 5)

    # def test_deque_rotate_left(self) -> None:
    #     self.test_list.rotate(-1)
    #     self.assertTrue(self.test_list.popleft() == 2)

    def test_deque_maxlen(self) -> None:
        l = ListBasedDeque([1, 2, 3], 5)
        self.assertTrue(l.maxlen() == 5)


if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    else:
        unittest.main()
