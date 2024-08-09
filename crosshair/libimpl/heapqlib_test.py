import heapq
from typing import List

from crosshair.core import proxy_for_type
from crosshair.tracers import ResumedTracing


def test_heapify(space):
    items = proxy_for_type(List[int], "items")

    with ResumedTracing():
        space.add(len(items) == 3)
        heapq.heapify(items)
        assert not space.is_possible(items[0] > items[1])
