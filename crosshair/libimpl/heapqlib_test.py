import heapq
from typing import List

from crosshair.test_util import check_ok


def test_heapify():
    def f(items: List[int]):
        """
        pre: len(items) == 3
        post: _[0] <= _[1]
        """
        heapq.heapify(items)
        return items

    actual, expected = check_ok(f)
    assert actual == expected
