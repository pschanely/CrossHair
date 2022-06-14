import re
import sys
from random import Random

from crosshair.stubs_parser import (
    _rewrite_with_typing_types,
    _rewrite_with_union,
    signature_from_stubs,
)


def test_rewrite_with_union():
    test_str = "List[str | int] | Callable[int | str, int]"
    expect = "Union[List[Union[str , int]] , Callable[Union[int , str], int]]"
    assert expect == _rewrite_with_union(test_str)


if sys.version_info < (3, 9):

    def test_rewrite_with_typing_types():
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
        expect_re = re.compile(
            r"""
            \( self .*
            population .* sequence .* _T .*
            \) \s \- \> .* _T
            """,
            re.VERBOSE | re.IGNORECASE,
        )
        assert valid and expect_re.match(str(s[0]))
    else:
        assert not s
