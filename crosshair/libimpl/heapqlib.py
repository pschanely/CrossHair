import functools
import heapq
import types

import _heapq

from crosshair.core import register_patch
from crosshair.util import debug, imported_alternative, name_of_type


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
    native_funcs = [
        "_heapify_max",
        "_heappop_max",
        "_heapreplace_max",
        "heapify",
        "heappop",
        "heappush",
        "heappushpop",
        "heapreplace",
    ]
    with imported_alternative("heapq", ("_heapq",)):

        # The pure python version doesn't always check argument types:
        heapq.heappush = heapq.heappush
        heapq.heappop = _check_first_arg_is_list(heapq.heappop)

        pure_fns = {name: getattr(heapq, name) for name in native_funcs}
    for name in native_funcs:
        native_fn = getattr(heapq, name)
        pure_fn = pure_fns[name]
        assert isinstance(native_fn, types.BuiltinFunctionType)
        assert isinstance(pure_fn, types.FunctionType)
        register_patch(native_fn, _check_first_arg_is_list(pure_fn))
