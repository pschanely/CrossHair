import collections
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


class CollectionsLibDequeTests(unittest.TestCase):
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

        self.assertTrue("1 is not in list" in str(context.exception))

    def test_deque_index_with_start_and_end_index(self) -> None:
        i = self.test_list.index(2, start=0, end=3)
        self.assertEqual(i, 1)

    def test_deque_index_with_start_and_end_index_throws_correct_exception(
        self,
    ) -> None:
        with self.assertRaises(ValueError) as context:
            self.test_list.index(6, start=0, end=1)

        self.assertTrue("6 is not in list" in str(context.exception))

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

    def test_deque_rotate_left(self) -> None:
        self.test_list.rotate(n=-1)
        self.assertTrue(self.test_list.popleft() == 2)

    def test_deque_maxlen(self) -> None:
        l = ListBasedDeque([1, 2, 3], 5)
        self.assertTrue(l.maxlen() == 5)

    def test_deque_len_ok(self) -> None:
        def f(l: Deque[int]) -> Deque[int]:
            """
            post: len(_) == len(__old__.l) + 1
            """
            l.append(42)
            return l

        self.assertEqual(*check_ok(f))

    def test_deque_len_fail(self) -> None:
        def f(l: Deque[int]) -> Deque[int]:
            """
            pre: len(l) > 0
            post: len(l) != 222
            """
            return l

        self.assertEqual(*check_fail(f))


class CollectionsLibDefaultDictTests(unittest.TestCase):
    def test_repr_equiv(self) -> None:
        def f(symbolic: DefaultDict[int, int]) -> Tuple[dict, dict]:
            """ post: _[0] == _[1] """
            concrete = collections.defaultdict(
                symbolic.default_factory, symbolic.items()
            )
            return (symbolic, concrete)

        self.assertEqual(*check_unknown(f))

    def test_basic_fail(self) -> None:
        def f(a: DefaultDict[int, int], k: int, v: int) -> None:
            """
            post[a]: a[42] != 42
            """
            a[k] = v

        self.assertEqual(*check_fail(f))

    def test_default_fail(self) -> None:
        def f(a: DefaultDict[int, int], k: int) -> None:
            """
            post: a[k] <= 100
            """
            if a[k] > 100:
                del a[k]

        self.assertEqual(*check_fail(f))

    def test_default_ok(self) -> None:
        def f(a: DefaultDict[int, int], k1: int, k2: int) -> DefaultDict[int, int]:
            """
            pre: len(a) == 0 and a.default_factory is not None
            post: _[k1] == _[k2]
            """
            return a

        self.assertEqual(*check_ok(f))


class BytesTest(unittest.TestCase):
    def test_specific_length(self) -> None:
        def f(b: bytes) -> int:
            """ post: _ != 5 """
            return len(b)

        self.assertEqual(*check_fail(f))

    def test_out_of_range_byte(self) -> None:
        def f(b: bytes) -> bytes:
            """
            pre: len(b) == 1
            post: _[0] != 256
            """
            return b

        self.assertEqual(*check_ok(f))


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
