import functools
import heapq

import _heapq

from crosshair.core import register_patch
from crosshair.util import imported_alternative, name_of_type


def _check_first_arg_is_list(fn):
    functools.wraps(fn)

    def wrapper(heap, *a, **kw):
        if not isinstance(heap, list):
            raise TypeError(
                f"{fn.__name__} argument must be list, not {name_of_type(heap)}"
            )
        return fn(heap, *a, **kw)

    return wrapper


def make_registrations():
    native_funcs = [name for name in dir(_heapq) if not name.startswith("_")]
    assert native_funcs == [
        "heapify",
        "heappop",
        "heappush",
        "heappushpop",
        "heapreplace",
    ]
    with imported_alternative("heapq", ("_heapq",)):

        # The pure python version doesn't always check argument types:
        heapq.heappush = _check_first_arg_is_list(heapq.heappush)
        heapq.heappop = _check_first_arg_is_list(heapq.heappop)
        heapq.heapify = _check_first_arg_is_list(heapq.heapify)

        for name in native_funcs:
            assert getattr(_heapq, name) != getattr(heapq, name)
            register_patch(getattr(_heapq, name), getattr(heapq, name))
