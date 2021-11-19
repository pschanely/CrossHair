from ast import literal_eval
from unicodedata import unidata_version
from sys import maxunicode

from crosshair.unicode_categories import compute_categories
from crosshair.unicode_categories import _PRECOMPUTED_CATEGORY_RANGES


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
