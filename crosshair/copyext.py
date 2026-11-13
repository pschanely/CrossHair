import sys

if sys.version_info >= (3, 14):
    from copy import _atomic_types
else:
    from copy import _deepcopy_atomic  # type: ignore
from copy import (  # type: ignore
    Error,
    _deepcopy_dict,
    _deepcopy_dispatch,
    _deepcopy_list,
    _deepcopy_tuple,
    _keep_alive,
    _reconstruct,
)
from copyreg import dispatch_table  # type: ignore
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

from crosshair.tracers import ResumedTracing
from crosshair.util import (
    CrossHairInternal,
    IdKeyedDict,
    assert_tracing,
    ch_stack,
    debug,
    name_of_type,
)

_MISSING = object


class CopyMode(int, Enum):
    REGULAR = 0
    BEST_EFFORT = 1  # Shares objects that cannot be copied; see copy_failures()
    REALIZE = 2


class CopyFailure(NamedTuple):
    typ: type
    exc: Exception

    def __str__(self) -> str:
        return f"{name_of_type(type(self.exc))}: {self.exc}"


class LazyCopySource:
    """
    An object whose copies create parts of themselves later.

    Failures to copy those parts accumulate in `copy_failures`.
    """

    copy_failures: List[CopyFailure]


_FailureSource = Union[CopyFailure, LazyCopySource]


class _FailureTracker:
    """
    Records the copy failures reachable from each object in a best-effort copy.
    """

    def __init__(self) -> None:
        self.frames: List[List[_FailureSource]] = []
        self.reach: Dict[int, Tuple[_FailureSource, ...]] = {}

    def add(self, source: _FailureSource) -> None:
        if self.frames:
            self.frames[-1].append(source)

    def add_reach_of(self, objid: int) -> None:
        if self.frames:
            self.frames[-1].extend(self.reach.get(objid, ()))

    def enter(self) -> None:
        self.frames.append([])

    def exit(self, objid: int) -> None:
        sources = tuple(dict.fromkeys(self.frames.pop()))
        if sources:
            self.reach[objid] = sources
            if self.frames:
                self.frames[-1].extend(sources)


_TRACKER_KEY = id(_FailureTracker)


def _tracker(memo: Dict) -> _FailureTracker:
    tracker = memo.get(_TRACKER_KEY)
    if tracker is None:
        tracker = memo[_TRACKER_KEY] = _FailureTracker()
    return tracker


def memoize_copy(obj: object, cpy: object, memo: Dict) -> None:
    """Record `cpy` as the copy of `obj` (for use in `__deepcopy__` implementations)."""
    memo[id(obj)] = cpy
    _keep_alive(obj, memo)


def note_lazy_copy_source(source: LazyCopySource, memo: Dict) -> None:
    """Make the failures of `source` count against objects whose copy reaches it."""
    _tracker(memo).add(source)


def copy_failures(obj: object, memo: Dict) -> List[CopyFailure]:
    """
    Get the failures of a best-effort copy of `obj` made with `memo`.

    Includes failures that lazy parts of the copy (or of `obj`) have recorded so far.
    """
    tracker: Optional[_FailureTracker] = memo.get(_TRACKER_KEY)
    if tracker is None:
        return []
    result: Dict[CopyFailure, None] = {}
    for source in tracker.reach.get(id(obj), ()):
        if isinstance(source, CopyFailure):
            result[source] = None
        else:
            result.update(dict.fromkeys(source.copy_failures))
    return list(result)


# We need to be able to realize some types that are not deep-copyable.
# Such realization overrides are defined here.
# TODO: This capability should probably be something that plugins can extend
_DEEP_REALIZATION_OVERRIDES = IdKeyedDict()
_DEEP_REALIZATION_OVERRIDES[MappingProxyType] = lambda p, m: MappingProxyType(
    deepcopyext(dict(p), CopyMode.REALIZE, m)
)
if sys.version_info >= (3, 10):
    _DEEP_REALIZATION_OVERRIDES[type({}.items())] = lambda p, m: deepcopyext(
        p.mapping, CopyMode.REALIZE, m
    ).items()
    _DEEP_REALIZATION_OVERRIDES[type({}.keys())] = lambda p, m: deepcopyext(
        p.mapping, CopyMode.REALIZE, m
    ).keys()
    _DEEP_REALIZATION_OVERRIDES[type({}.values())] = lambda p, m: deepcopyext(
        p.mapping, CopyMode.REALIZE, m
    ).values()


@assert_tracing(False)
def deepcopyext(obj: object, mode: CopyMode, memo: Dict) -> Any:
    tracker = _tracker(memo) if mode == CopyMode.BEST_EFFORT else None
    objid = id(obj)
    cpy = memo.get(objid, _MISSING)
    if cpy is not _MISSING:
        if objid not in map(id, memo.get(id(memo), ())):
            # we are trying to return some value that was not kept alive;
            # it may have been garbage collected and replaced.
            raise CrossHairInternal("Possibly transient value found in memo")
        if tracker:
            tracker.add_reach_of(objid)
        return cpy
    _keep_alive(obj, memo)
    if tracker:
        tracker.enter()
        try:
            cpy = _deepcopy_unmemoized(obj, mode, memo, tracker)
        finally:
            tracker.exit(objid)
    else:
        cpy = _deepcopy_unmemoized(obj, mode, memo, None)
    memo[objid] = cpy
    return cpy


def _deepcopy_unmemoized(
    obj: object, mode: CopyMode, memo: Dict, tracker: Optional[_FailureTracker]
) -> Any:
    deepconstruct_obj = obj
    if mode == CopyMode.REALIZE:
        cls = type(obj)
        if hasattr(cls, "__ch_deep_realize__"):
            return obj.__ch_deep_realize__(memo)  # type: ignore
        elif hasattr(cls, "__ch_realize__"):
            # Do shallow realization here, and then fall through to
            # _deepconstruct below.
            deepconstruct_obj = obj.__ch_realize__()  # type: ignore
            # this transient object may be inserted into memo below
            _keep_alive(deepconstruct_obj, memo)
        else:
            realization_override = _DEEP_REALIZATION_OVERRIDES.get(cls)
            if realization_override:
                return realization_override(obj, memo)
    try:
        return _deepconstruct(deepconstruct_obj, mode, memo)
    except Exception as exc:
        if mode == CopyMode.REGULAR or (
            mode == CopyMode.REALIZE and not isinstance(exc, TypeError)
        ):
            raise
        debug(
            "Cannot copy object of type",
            type(obj),
            "ignoring",
            type(exc),
            ":",
            str(exc),
            "at",
            ch_stack(currently_handling=exc),
        )
        if tracker:
            tracker.add(CopyFailure(type(obj), exc))
        return deepconstruct_obj


if sys.version_info >= (3, 14):

    def lookup_dispatch(cls: type) -> Callable:
        if cls in _atomic_types:
            return lambda obj, memo: obj
        return _deepcopy_dispatch.get(cls)

else:

    def lookup_dispatch(cls: type) -> Callable:
        return _deepcopy_dispatch.get(cls)


_OBJECT_GETSTATE = getattr(object, "__getstate__", None)


def _reduces_without_user_code(cls: type) -> bool:
    return (
        getattr(cls, "__reduce__") is object.__reduce__
        and getattr(cls, "__reduce_ex__") is object.__reduce_ex__
        and getattr(cls, "__getstate__", None) is _OBJECT_GETSTATE
        and not hasattr(cls, "__getnewargs_ex__")
        and not hasattr(cls, "__getnewargs__")
    )


def _deepconstruct(obj: object, mode: CopyMode, memo: Dict):
    cls = type(obj)

    def subdeepcopy(obj: object, memo: Dict):
        return deepcopyext(obj, mode, memo)

    creator = lookup_dispatch(cls)
    if creator is not None:
        if creator in (_deepcopy_dict, _deepcopy_list, _deepcopy_tuple):
            return creator(obj, memo, deepcopy=subdeepcopy)
        else:
            # TODO: We loose subdeepcopy in this case - won't
            # that make e.g. deep_realize be too shallow?
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
    if _reduces_without_user_code(cls):
        reduct = to_call(*call_args)
    else:
        with ResumedTracing():
            reduct = to_call(*call_args)
    if isinstance(reduct, str):
        return obj
    return _reconstruct(obj, memo, *reduct, deepcopy=subdeepcopy)
