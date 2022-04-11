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
    if sys.version_info < (3, 9):
        test_str = "list[dict[int, list]]"
        expect = "typing.List[typing.Dict[int, list]]"
        glo = dict()
        assert expect == _rewrite_with_typing_types(test_str, glo)
        assert "typing" in glo


def test_signature_from_stubs():
    s, valid = signature_from_stubs(Random.randint)
    if sys.version_info >= (3, 8):
        assert valid and str(s[0]) == "(self, a: int, b: int) -> int"
        s, valid = signature_from_stubs(Random.sample)
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
        assert valid and str(s[0]) == expect
    else:
        assert not s
