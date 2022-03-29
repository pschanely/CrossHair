from random import Random
import sys
from crosshair.stubs_parser import (
    _rewrite_with_union,
    _rewrite_with_typing_types,
    signature_from_stubs,
)


def test_rewrite_with_union():
    test_str = "List[str | int] | Callable[int | str, int]"
    expect = "Union[List[Union[str , int]] , Callable[Union[int , str], int]]"
    assert expect == _rewrite_with_union(test_str)


def test_rewrite_with_typing_types():
    test_str = "list[dict[int, list]]"
    expect = "List[Dict[int, list]]"
    glo = dict()
    assert expect == _rewrite_with_typing_types(test_str, glo)
    assert "List" in glo


def test_signature_from_stubs():
    s = signature_from_stubs(Random.randint)
    if sys.version_info >= (3, 8):
        assert str(s) == "(self, a: int, b: int) -> int"
        s = signature_from_stubs(Random.sample)
        if sys.version_info >= (3, 10):
            expect = (
                "(self, population: collections.abc.Sequence[~_T] | "
                "collections.abc.Set[~_T], k: int, *, counts: "
                "collections.abc.Iterable[~_T] | None = Ellipsis) -> list[~_T]"
            )
        elif sys.version_info >= (3, 9):
            expect = (
                "(self, population: Union[collections.abc.Sequence[~_T], "
                "collections.abc.Set[~_T]], k: int, *, counts: "
                "Optional[collections.abc.Iterable[~_T]] = Ellipsis) -> list[~_T]"
            )
        else:
            expect = (
                "(self, population: Union[Sequence[~_T], Set[~_T]], k: int) -> "
                "List[~_T]"
            )
        assert str(s) == expect
    else:
        assert s is None