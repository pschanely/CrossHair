from copy import _deepcopy_atomic  # type: ignore
from copy import _deepcopy_dict  # type: ignore
from copy import _deepcopy_dispatch  # type: ignore
from copy import _deepcopy_list  # type: ignore
from copy import _deepcopy_tuple  # type: ignore
from copy import _keep_alive  # type: ignore
from copy import _reconstruct  # type: ignore
from copy import Error
from copyreg import dispatch_table  # type: ignore
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Tuple

from crosshair.tracers import ResumedTracing, is_tracing
from crosshair.util import IdKeyedDict, debug, test_stack

_MISSING = object


class CopyMode(int, Enum):
    REGULAR = 0
    BEST_EFFORT = 1
    REALIZE = 2


# We need to be able to realize some types that are not deep-copyable.
# Such realization overrides are defined here.
# TODO: This capability should probably be something that plugins can extend
_DEEP_REALIZATION_OVERRIDES = IdKeyedDict()
_DEEP_REALIZATION_OVERRIDES[MappingProxyType] = lambda p, m: MappingProxyType(
    deepcopyext(dict(p), CopyMode.REALIZE, m)
)


def deepcopyext(obj: object, mode: CopyMode, memo: Dict) -> Any:
    assert not is_tracing()
    objid = id(obj)
    cpy = memo.get(objid, _MISSING)
    if cpy is _MISSING:
        if mode == CopyMode.REALIZE:
            cls = type(obj)
            if hasattr(cls, "__ch_deep_realize__"):
                cpy = obj.__ch_deep_realize__(memo)  # type: ignore
            elif hasattr(cls, "__ch_realize__"):
                # Do shallow realization here, and then fall through to
                # _deepconstruct below.
                obj = obj.__ch_realize__()  # type: ignore
            realization_override = _DEEP_REALIZATION_OVERRIDES.get(cls)
            if realization_override:
                cpy = realization_override(obj, memo)
        if cpy is _MISSING:
            try:
                cpy = _deepconstruct(obj, mode, memo)
            except TypeError as exc:
                if mode == CopyMode.REGULAR:
                    raise
                debug(
                    "Cannot copy object of type",
                    type(obj),
                    "ignoring",
                    type(exc),
                    ":",
                    str(exc),
                    "at",
                    test_stack(exc.__traceback__),
                )
                cpy = obj
        memo[objid] = cpy
        _keep_alive(obj, memo)
    return cpy


def _deepconstruct(obj: object, mode: CopyMode, memo: Dict):
    cls = type(obj)

    def subdeepcopy(obj: object, memo: Dict):
        return deepcopyext(obj, mode, memo)

    if cls in _deepcopy_dispatch:
        creator = _deepcopy_dispatch[cls]
        if creator is _deepcopy_atomic:
            return obj
        elif creator in (_deepcopy_dict, _deepcopy_list, _deepcopy_tuple):
            return creator(obj, memo, deepcopy=subdeepcopy)
        else:
            return creator(obj, memo)
    if isinstance(obj, type):
        return obj
    if mode != CopyMode.REALIZE and hasattr(obj, "__deepcopy__"):
        return obj.__deepcopy__(memo)  # type: ignore
    if cls in dispatch_table:
        to_call = dispatch_table[cls]
        call_args: Tuple = (obj,)
    elif hasattr(cls, "__reduce_ex__"):
        to_call = getattr(cls, "__reduce_ex__")
        call_args = (obj, 4)
    elif hasattr(cls, "__reduce__"):
        to_call = getattr(cls, "__reduce__")
        call_args = (obj,)
    else:
        raise Error("un(deep)copyable object of type %s" % cls)
    if (
        getattr(cls, "__reduce__") is object.__reduce__
        and getattr(cls, "__reduce_ex__") is object.__reduce_ex__
    ):
        reduct = to_call(*call_args)
    else:
        with ResumedTracing():
            reduct = to_call(*call_args)
    if isinstance(reduct, str):
        return obj
    return _reconstruct(obj, memo, *reduct, deepcopy=subdeepcopy)
