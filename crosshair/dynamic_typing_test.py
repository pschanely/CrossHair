import collections
from inspect import Parameter, Signature, signature
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import pytest
from typing_extensions import TypedDict

from crosshair.dynamic_typing import (
    get_bindings_from_type_arguments,
    intersect_signatures,
    realize,
    unify,
)
from crosshair.options import AnalysisOptionSet
from crosshair.statespace import CANNOT_CONFIRM
from crosshair.test_util import check_states

_T = TypeVar("_T")
_U = TypeVar("_U")


def test_raw_tuple():
    bindings = collections.ChainMap()
    assert unify(tuple, Iterable[_T], bindings)


def test_typedicts():
    class A1(TypedDict):
        x: bool

    class A2(A1, total=False):
        y: bool  # can be omitted because total=False

    class B1(TypedDict):
        x: int

    class B2(B1, total=False):
        y: str

    bindings = collections.ChainMap()
    assert unify(A1, B1, bindings)
    assert unify(A2, B1, bindings)
    assert unify(A1, A2, bindings)
    assert unify(B1, B2, bindings)
    assert not unify(A2, B2, bindings)  # (because the types for "y" are incompatible)


def test_typevars():
    bindings = collections.ChainMap()
    assert unify(Tuple[int, str, List[int]], Tuple[int, _T, _U], bindings)
    assert realize(Mapping[_U, _T], bindings) == Mapping[List[int], str]


def test_bound_vtypears():
    assert unify(Dict[str, int], Dict[_T, _U])
    assert not (unify(Dict[str, int], Dict[_T, _T]))


def test_zero_type_args_ok():
    assert unify(map, Iterable[_T])
    assert not (unify(map, Iterable[int]))


def test_callable():
    bindings = collections.ChainMap()
    assert unify(Callable[[Iterable], bool], Callable[[List], bool], bindings)

    assert not unify(Callable[[List], bool], Callable[[Iterable], bool], bindings)
    assert unify(Callable[[int, _T], List[int]], Callable[[int, str], _U], bindings)
    assert realize(Callable[[_U], _T], bindings) == Callable[[List[int]], str]


def test_plain_callable():
    bindings = collections.ChainMap()
    assert unify(Callable[[int, str], List[int]], Callable, bindings)


def test_uniform_tuple():
    bindings = collections.ChainMap()
    assert unify(Tuple[int, int], Tuple[_T, ...], bindings)
    assert bindings[_T] == int
    assert not unify(Tuple[int, str], Tuple[_T, ...], bindings)


def test_tuple():
    bindings = collections.ChainMap()
    assert not unify(tuple, Tuple[int, str], bindings)


def test_union_fail():
    bindings = collections.ChainMap()
    assert not unify(Iterable[int], Union[int, Dict[str, _T]], bindings)


def test_union_ok():
    bindings = collections.ChainMap()
    assert unify(int, Union[str, int], bindings)
    assert unify(Tuple[int, ...], Union[int, Iterable[_T]], bindings)
    assert bindings[_T] == int


def test_union_into_union():
    bindings = collections.ChainMap()
    assert unify(Union[str, int], Union[str, int, float], bindings)
    assert not unify(Union[str, int, float], Union[str, int], bindings)


def test_nested_union():
    bindings = collections.ChainMap()
    assert unify(List[str], Sequence[Union[str, int]], bindings)


class Pair(Generic[_U, _T]):
    def __init__(self, u: _U, t: _T):
        self.u = u
        self.t = t


def test_bindings_from_type_arguments():
    var_mapping = get_bindings_from_type_arguments(Pair[int, str])
    assert var_mapping == {_U: int, _T: str}
    assert realize(List[_U], var_mapping) == List[int]


def test_intersect_signatures_basic():
    def f1(x: int, y: str, **kw) -> List[bool]:
        return []

    def f2(x: bool, *extra: str, **kw) -> List[int]:
        return []

    intersection = intersect_signatures(signature(f1), signature(f2))
    assert intersection is not None
    assert intersection.parameters == {
        "x": Parameter("x", kind=Parameter.KEYWORD_ONLY, annotation=bool),
        "y": Parameter("y", kind=Parameter.KEYWORD_ONLY, annotation=str),
        "kw": Parameter("kw", kind=Parameter.VAR_KEYWORD),
    }
    assert intersection.return_annotation == List[bool]


def test_intersect_signatures_typevars():
    _T = TypeVar("_T")

    def f1(cc, *args, **kwds):
        pass

    def f2(dd, left: Optional[_T], right: Optional[_T]):
        pass

    intersection = intersect_signatures(signature(f1), signature(f2))
    assert intersection is not None
    expected = {
        "dd": Parameter("dd", kind=Parameter.KEYWORD_ONLY),
        "left": Parameter("left", kind=Parameter.KEYWORD_ONLY, annotation=Optional[_T]),
        "right": Parameter(
            "right", kind=Parameter.KEYWORD_ONLY, annotation=Optional[_T]
        ),
    }
    assert intersection.parameters == expected


@pytest.mark.skip(
    reason="The inspect module doesn't expose runtime type information yet"
)
def test_intersect_signature_with_crosshair():
    def check_intersect_signatures(
        sig1: Signature, sig2: Signature, pos_args: List, kw_args: Mapping[str, object]
    ) -> None:
        """post: True"""

        def _sig_bindable(sig: Signature) -> bool:
            try:
                sig.bind(*pos_args, **kw_args)
                return True
            except TypeError:
                return False

        if _sig_bindable(sig1) or _sig_bindable(sig2):
            intersection = intersect_signatures(sig1, sig2)
            assert _sig_bindable(intersection)

    check_states(check_intersect_signatures, CANNOT_CONFIRM)
