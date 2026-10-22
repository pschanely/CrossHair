from sys import maxunicode
from unicodedata import unidata_version

import z3

from crosshair.statespace import make_default_solver
from crosshair.unicode_categories import (
    CharMask,
    IntTransform,
    casemap,
    compute_unicode_data,
    read_unicode_data,
    smt_lower_1st,
    smt_lower_changes,
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


def test_int_transform_segments():
    mapping = {65: 97, 66: 98, 68: 100, 200: 7, 205: 7, 300: 1}
    transform = IntTransform.from_mapping(mapping)
    assert transform.starts == (65, 200, 300)
    for key, value in mapping.items():
        assert z3.simplify(transform.smt_apply(z3.IntVal(key))).as_long() == value


def test_int_transform_matches_casemaps():
    for casefn in (str.upper, str.lower, str.title, str.casefold):
        mapping = {k: ord(v[0]) for k, v in casemap(casefn).items()}
        transform = IntTransform.from_mapping(mapping)
        assert all(transform.apply(k) == v for k, v in mapping.items())


def test_int_transform_offsets_are_decidable():
    codepoint = z3.Int("codepoint")
    solver = make_default_solver()
    solver.add(0 <= codepoint, codepoint <= maxunicode)
    solver.add(smt_lower_changes(codepoint))
    solver.add(smt_lower_1st(codepoint) == codepoint)
    assert solver.check() == z3.unsat
