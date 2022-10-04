import collections
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from crosshair.dynamic_typing import realize, unify

_T = TypeVar("_T")
_U = TypeVar("_U")


def test_raw_tuple():
    bindings = collections.ChainMap()
    assert unify(tuple, Iterable[_T], bindings)


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
