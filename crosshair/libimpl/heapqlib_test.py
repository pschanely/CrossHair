import heapq
from typing import List

from crosshair.options import AnalysisOptionSet
from crosshair.statespace import CONFIRMED, MessageType
from crosshair.test_util import check_states


def test_heapify():
    def f(items: List[int]):
        """
        pre: len(items) == 3
        post: _[0] <= _[1]
        """
        heapq.heapify(items)
        return items

    check_states(f, CONFIRMED)
