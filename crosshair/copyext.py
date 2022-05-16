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
from typing import Any, Dict

from crosshair.tracers import is_tracing
from crosshair.util import debug

_MISSING = object


class CopyMode(int, Enum):
    REGULAR = 0
    BEST_EFFORT = 1
    REALIZE = 2


def deepcopyext(obj: object, mode: CopyMode, memo: Dict) -> Any:
    assert not is_tracing()
    objid = id(obj)
    cpy = memo.get(objid, _MISSING)
    if cpy is _MISSING:
        if mode == CopyMode.REALIZE:
            cls = type(obj)
            if hasattr(cls, "__ch_deep_realize__"):
                cpy = obj.__ch_deep_realize__()  # type: ignore
            elif hasattr(cls, "__ch_realize__"):
                # Do shallow realization here, and then fall through to
                # _deepconstruct below.
                obj = obj.__ch_realize__()  # type: ignore
        if cpy is _MISSING:
            try:
                cpy = _deepconstruct(obj, mode, memo)
            except TypeError as exc:
                if mode == CopyMode.REGULAR:
                    raise
                debug(f"Cannot copy object of type {type(obj)}, ignoring: {exc}")
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
        reduct = dispatch_table[cls](obj)
    elif hasattr(obj, "__reduce_ex__"):
        reduct = getattr(obj, "__reduce_ex__")(4)
    elif hasattr(obj, "__reduce__"):
        reduct = getattr(obj, "__reduce__")()
    else:
        raise Error("un(deep)copyable object of type %s" % cls)
    if isinstance(reduct, str):
        return obj
    return _reconstruct(obj, memo, *reduct, deepcopy=subdeepcopy)
