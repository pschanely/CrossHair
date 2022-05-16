from ast import literal_eval
from sys import maxunicode
from unicodedata import unidata_version

from crosshair.unicode_categories import (
    _PRECOMPUTED_CATEGORY_RANGES,
    CharMask,
    compute_categories,
)


def test_categories_cached_correctly():
    precomputed_str = _PRECOMPUTED_CATEGORY_RANGES[unidata_version]
    assert compute_categories() == literal_eval(precomputed_str)


def test_transformation_assumptions():
    for cp in range(maxunicode + 1):
        ch = chr(cp)
        assert len(ch.casefold()) <= 3
        assert len(ch.lower()) <= 2
        assert len(ch.upper()) <= 3
        assert len(ch.title()) <= 3


def test_union():
    assert CharMask([(10, 20)]).union(CharMask([(13, 18)])) == CharMask([(10, 20)])
