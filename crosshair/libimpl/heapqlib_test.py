import heapq
from typing import List

from crosshair.options import AnalysisOptionSet
from crosshair.statespace import CONFIRMED, MessageType
from crosshair.test_util import check_states

_SLOW_TEST = AnalysisOptionSet(per_condition_timeout=10)


def test_heapify():
    def f(items: List[int]):
        """
        pre: len(items) == 3
        post: _[0] <= _[1]
        """
        heapq.heapify(items)
        return items

    check_states(f, CONFIRMED, _SLOW_TEST)
