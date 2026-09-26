from sys import maxunicode
from unicodedata import unidata_version

from crosshair.unicode_categories import (
    CharMask,
    compute_unicode_data,
    read_unicode_data,
    unicode_data_path,
)


def test_stored_unicode_data_is_current():
    path = unicode_data_path(unidata_version)
    assert path.exists(), "Generate with `python -m crosshair.unicode_categories`"
    assert read_unicode_data(path) == compute_unicode_data()


def test_transformation_assumptions():
    for cp in range(maxunicode + 1):
        ch = chr(cp)
        assert len(ch.casefold()) <= 3
        assert len(ch.lower()) <= 2
        assert len(ch.upper()) <= 3
        assert len(ch.title()) <= 3


def test_union():
    assert CharMask([(10, 20)]).union(CharMask([(13, 18)])) == CharMask([(10, 20)])


def test_boundaries_roundtrip():
    mask = CharMask([3, (5, 9), maxunicode])
    assert mask.boundaries() == [3, 4, 5, 9, maxunicode, maxunicode + 1]
    assert CharMask.from_boundaries(mask.boundaries()) == mask
