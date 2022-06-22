import heapq

import _heapq

from crosshair.core import register_patch
from crosshair.util import import_alternative


def make_registrations():
    pureheapq = import_alternative("heapq", ("_heapq",))
    native_funcs = [name for name in dir(_heapq) if not name.startswith("_")]
    assert native_funcs == [
        "heapify",
        "heappop",
        "heappush",
        "heappushpop",
        "heapreplace",
    ]
    for name in native_funcs:
        register_patch(getattr(heapq, name), getattr(pureheapq, name))
