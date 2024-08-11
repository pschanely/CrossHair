import heapq
import sys
from typing import List

import pytest

from crosshair.core import proxy_for_type
from crosshair.tracers import ResumedTracing


# TODO https://github.com/pschanely/CrossHair/issues/298
@pytest.mark.skip(
    reason="heapq get reloaded somehow in parallel ci run, ruining the intercepts",
)
def test_heapify(space):
    items = proxy_for_type(List[int], "items")

    with ResumedTracing():
        space.add(len(items) == 3)
        heapq.heapify(items)
        assert not space.is_possible(items[0] > items[1])
