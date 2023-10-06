import collections
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import TypedDict

from crosshair.dynamic_typing import get_bindings_from_type_arguments, realize, unify

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
