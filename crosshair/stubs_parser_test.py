import codecs
import json
import re
import sys
from random import Random

from crosshair.stubs_parser import _rewrite_with_union, signature_from_stubs


def test_rewrite_with_union():
    test_str = "List[str | int] | Callable[int | str, int]"
    expect = "Union[List[Union[str , int]] , Callable[Union[int , str], int]]"
    assert expect == _rewrite_with_union(test_str)


def test_inherited_method_resolves_from_a_base_class():
    """typeshed declares getstate on IncrementalDecoder, not on the subclass."""
    sigs, valid = signature_from_stubs(codecs.BufferedIncrementalDecoder.getstate)
    assert valid
    assert [str(s) for s in sigs] == ["(self) -> tuple[bytes, int]"]


def test_inherited_method_resolves_across_modules():
    """JSONDecodeError inherits __reduce__ from builtins.object."""
    sigs, valid = signature_from_stubs(json.JSONDecodeError.__reduce__)
    assert valid and len(sigs) == 1
    assert list(sigs[0].parameters) == ["self"]


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
