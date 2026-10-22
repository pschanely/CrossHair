import json
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, wraps
from itertools import accumulate
from pathlib import Path
from sys import maxunicode
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from unicodedata import category, decimal, digit, numeric, unidata_version

import z3  # type: ignore

from crosshair.z3util import z3And, z3Eq, z3Gt, z3IntVal, z3Le, z3Lt, z3Or


@dataclass
class CharMask:
    """
    Represents a mask over unicode codepoints.

    Each range's minimum is less than it's maximum. Ranges cover valid codepoints.
    inv: all(0 <= minb < maxb <= maxunicode + 1 for (minb, maxb) in self.all_bounds())

    Ranges are listed in-order and are not overlapping or adjacent.
    inv: all(self.bounds_at(i)[1] < self.bounds_at(i + 1)[0] for i in range(len(self.parts) - 1))
    """

    parts: List[Union[int, Tuple[int, int]]]

    @classmethod
    def from_boundaries(cls, boundaries: Sequence[int]) -> "CharMask":
        """Build a mask from alternating range minimums and (exclusive) maximums."""
        mask = cls([])
        for i in range(0, len(boundaries), 2):
            mask.maybe_add_bounds(boundaries[i], boundaries[i + 1])
        return mask

    def maybe_add_bounds(self, minimum: int, maximum: int) -> None:
        """
        Append a new range to the bounds, if possible.

        `minimum` must be greater than or equal to the minimum of the highest existing
        range.
        pre: len(self.parts) == 0 or minimum >= self.bounds_at(len(self.parts) - 1)[0]

        New bounds must be valid.
        pre: 0 <= minimum <= maxunicode + 1
        pre: 0 <= maximum <= maxunicode + 1
        """
        if minimum < maximum:
            if self.parts:
                last_min, last_max = self.bounds_at(len(self.parts) - 1)
                if minimum <= last_max:
                    if maximum <= last_max:
                        return
                    assert minimum >= last_min
                    self.parts[-1] = (last_min, maximum)
                    return
            self.parts.append(minimum if minimum + 1 == maximum else (minimum, maximum))

    def smt_matches(self, smt_ch: z3.ExprRef) -> z3.ExprRef:
        constraints = []
        for part in self.parts:
            if isinstance(part, int):
                constraints.append(z3Eq(z3IntVal(part), smt_ch))
            else:
                constraints.append(
                    z3And(
                        z3Le(z3IntVal(part[0]), smt_ch), z3Gt(z3IntVal(part[1]), smt_ch)
                    )
                )
        if len(constraints) <= 1:
            return constraints[0] if constraints else z3.BoolVal(False)
        else:
            return z3Or(*constraints)

    def covers(self, codepoint: int) -> bool:
        for minimum, maximum in self.all_bounds():
            if minimum <= codepoint < maximum:
                return True
        return False

    def all_bounds(self) -> Iterable[Tuple[int, int]]:
        parts = self.parts
        for i in range(len(parts)):
            yield self.bounds_at(i)

    def boundaries(self) -> List[int]:
        """Return alternating range minimums and (exclusive) maximums."""
        return [b for bounds in self.all_bounds() for b in bounds]

    def codepoints(self) -> Iterator[int]:
        for minimum, maximum in self.all_bounds():
            yield from range(minimum, maximum)

    def bounds_at(self, idx: int) -> Tuple[int, int]:
        """
        Return bounds for an allowed range at the given index.

        pre: 0 <= idx < len(self.parts)
        """
        item = self.parts[idx]
        return (item, item + 1) if isinstance(item, int) else item

    def invert(self) -> "CharMask":
        numbers = [0] + self.boundaries() + [maxunicode + 1]
        inversion = CharMask([])
        for i in range(1, len(numbers), 2):
            inversion.maybe_add_bounds(numbers[i - 1], numbers[i])
        return inversion

    def union(self, other: "CharMask") -> "CharMask":
        ret = CharMask([])
        for minimum, maximum in sorted([*self.all_bounds(), *other.all_bounds()]):
            ret.maybe_add_bounds(minimum, maximum)
        return ret

    def intersect(self, other: "CharMask") -> "CharMask":
        myparts = self.parts
        otherparts = other.parts
        result = CharMask([])
        for myidx in range(len(myparts)):
            for otheridx in range(len(otherparts)):
                mymin, mymax = self.bounds_at(myidx)
                omin, omax = other.bounds_at(otheridx)
                result.maybe_add_bounds(max(mymin, omin), min(mymax, omax))
        return result

    def subtract(self, other: "CharMask") -> "CharMask":
        return self.intersect(other.invert())


def compute_categories() -> Dict[str, CharMask]:
    masks: Dict[str, CharMask] = defaultdict(lambda: CharMask([]))
    run_start = 0
    run_category = category(chr(0))
    for codepoint in range(1, maxunicode + 1):
        cur_category = category(chr(codepoint))
        if cur_category != run_category:
            masks[run_category].maybe_add_bounds(run_start, codepoint)
            run_start, run_category = codepoint, cur_category
    masks[run_category].maybe_add_bounds(run_start, maxunicode + 1)
    return dict(sorted(masks.items()))


CHAR_SET_PREDICATES: Dict[str, Callable[[str], object]] = {
    "changed_by_casefold": lambda ch: ch.casefold() != ch,
    "changed_by_lower": lambda ch: ch.lower() != ch,
    "changed_by_title": lambda ch: ch.title() != ch,
    "changed_by_upper": lambda ch: ch.upper() != ch,
    "has_decimal": lambda ch: decimal(ch, None) is not None,
    "has_digit": lambda ch: digit(ch, None) is not None,
    "has_numeric": lambda ch: numeric(ch, None) is not None,
    "islower": str.islower,
    "isspace": str.isspace,
    "istitle": str.istitle,
    "isupper": str.isupper,
    "word": re.compile(r"\w").fullmatch,
}


def compute_char_set(predicate: Callable[[str], object]) -> CharMask:
    mask = CharMask([])
    for codepoint in range(maxunicode + 1):
        if predicate(chr(codepoint)):
            mask.maybe_add_bounds(codepoint, codepoint + 1)
    return mask


def compute_unicode_data() -> Dict[str, Dict[str, CharMask]]:
    return {
        "categories": compute_categories(),
        "char_sets": {
            name: compute_char_set(predicate)
            for name, predicate in CHAR_SET_PREDICATES.items()
        },
    }


UNICODE_DATA_DIR = Path(__file__).parent / "unicode_data"


def unicode_data_path(version: str) -> Path:
    return UNICODE_DATA_DIR / f"{version}.json"


def write_unicode_data(path: Path, data: Dict[str, Dict[str, CharMask]]) -> None:
    """
    Write masks as JSON, one mask per line.

    Each mask is stored as the successive differences of its boundaries.
    """
    sections = []
    for section, masks in data.items():
        entries = []
        for name, mask in masks.items():
            boundaries = mask.boundaries()
            deltas = [b - a for a, b in zip([0] + boundaries, boundaries)]
            entries.append(f"{json.dumps(name)}:{json.dumps(deltas).replace(' ', '')}")
        sections.append(f"{json.dumps(section)}:{{\n" + ",\n".join(entries) + "\n}")
    path.write_text("{\n" + ",\n".join(sections) + "\n}\n")


def read_unicode_data(path: Path) -> Dict[str, Dict[str, CharMask]]:
    return {
        section: {
            name: CharMask.from_boundaries(list(accumulate(deltas)))
            for name, deltas in masks.items()
        }
        for section, masks in json.loads(path.read_text()).items()
    }


@lru_cache(maxsize=None)
def _stored_unicode_data() -> Optional[Dict[str, Dict[str, CharMask]]]:
    path = unicode_data_path(unidata_version)
    return read_unicode_data(path) if path.exists() else None


@lru_cache(maxsize=None)
def get_unicode_categories() -> Dict[str, CharMask]:
    stored = _stored_unicode_data()
    return stored["categories"] if stored else compute_categories()


@lru_cache(maxsize=None)
def get_char_set(name: str) -> CharMask:
    """Return the codepoints satisfying `CHAR_SET_PREDICATES[name]`."""
    stored = _stored_unicode_data()
    if stored:
        return stored["char_sets"][name]
    return compute_char_set(CHAR_SET_PREDICATES[name])


@lru_cache(maxsize=None)
def get_unicode_mask(*cat_names: str, invert: bool = False) -> CharMask:
    cats = get_unicode_categories()
    mask = cats[cat_names[0]]
    for cat_name in cat_names[1:]:
        mask = mask.union(cats[cat_name])
    if invert:
        mask = mask.invert()
    return mask


_T = TypeVar("_T")


@lru_cache(maxsize=None)
def casemap(casefn: Callable[[str], str]) -> Dict[int, str]:
    """Map each codepoint that `casefn` changes to its `casefn` result."""
    domain = get_char_set("changed_by_" + casefn.__name__)
    return {codepoint: casefn(chr(codepoint)) for codepoint in domain.codepoints()}


@lru_cache(maxsize=None)
def nummap(numfn: Callable[[str], _T]) -> Dict[int, _T]:
    """Map each codepoint that has a `numfn` value to that value."""
    domain = get_char_set("has_" + numfn.__name__)
    return {codepoint: numfn(chr(codepoint)) for codepoint in domain.codepoints()}


@lru_cache(maxsize=None)
def fractionmap() -> Dict[int, Tuple[int, int]]:
    ret = {}
    for codepoint, value in nummap(numeric).items():
        frac = Fraction(value).limit_denominator()
        ret[codepoint] = (frac.numerator, frac.denominator)
    return ret


def make_mask(vals: Iterable[int]) -> CharMask:
    mask = CharMask([])
    for val in vals:
        mask.maybe_add_bounds(val, val + 1)
    return mask


@dataclass(frozen=True)
class IntTransform:
    """
    Represents a partial map over integers, as sorted, non-overlapping segments.

    Each segment begins at a key in `starts` and runs to the next one. Every key in
    a segment maps to the key plus the segment's offset, or to the segment's
    constant (whichever is not None).
    Inputs that were not in the original map produce an arbitrary value.
    """

    starts: Tuple[int, ...]
    offsets: Tuple[Optional[int], ...]
    constants: Tuple[Optional[int], ...]

    @classmethod
    def from_mapping(cls, mapping: Dict[int, int]) -> "IntTransform":
        starts: List[int] = []
        offsets: List[Optional[int]] = []
        constants: List[Optional[int]] = []
        for key, value in sorted(mapping.items()):
            if starts:
                offset_matches = offsets[-1] == value - key
                constant_matches = constants[-1] == value
                if offset_matches or constant_matches:
                    if not offset_matches:
                        offsets[-1] = None
                    if not constant_matches:
                        constants[-1] = None
                    continue
            starts.append(key)
            offsets.append(value - key)
            constants.append(value)
        return cls(tuple(starts), tuple(offsets), tuple(constants))

    def apply(self, key: int) -> int:
        idx = bisect_right(self.starts, key) - 1
        offset = self.offsets[idx]
        return key + offset if offset is not None else self.constants[idx]  # type: ignore

    def smt_apply(self, smt_int: z3.ExprRef) -> z3.ExprRef:
        if not self.starts:
            return z3IntVal(0)
        constants = [c for c in self.constants if c is not None]
        if len(constants) == len(self.starts):
            return self._smt_segments(
                smt_int, [z3IntVal(c) for c in constants], 0, len(self.starts)
            )
        # Offsets are applied outside the tree; otherwise z3 may misjudge the
        # problem as difference logic and fail to solve it.
        leaves = [
            z3IntVal(o) if o is not None else z3IntVal(c) - smt_int  # type: ignore
            for o, c in zip(self.offsets, self.constants)
        ]
        return smt_int + self._smt_segments(smt_int, leaves, 0, len(self.starts))

    def _smt_segments(
        self, smt_int: z3.ExprRef, leaves: List[z3.ExprRef], lo: int, hi: int
    ) -> z3.ExprRef:
        if hi - lo == 1:
            return leaves[lo]
        mid = (lo + hi) // 2
        return z3.If(
            z3Lt(smt_int, z3IntVal(self.starts[mid])),
            self._smt_segments(smt_int, leaves, lo, mid),
            self._smt_segments(smt_int, leaves, mid, hi),
        )


SmtFn = Callable[[z3.ExprRef], z3.ExprRef]


def smt_mask(mask_getter: Callable[[], CharMask]) -> SmtFn:
    """Turn a mask getter into an SMT predicate over codepoints."""
    get_mask = lru_cache(maxsize=None)(mask_getter)

    @wraps(mask_getter)
    def matches(smt_codepoint: z3.ExprRef) -> z3.ExprRef:
        return get_mask().smt_matches(smt_codepoint)

    return matches


def smt_transform(mapping_getter: Callable[[], Dict[int, int]]) -> SmtFn:
    """Turn a mapping getter into an SMT function over the mapping's keys."""

    @lru_cache(maxsize=None)
    def get_transform() -> IntTransform:
        return IntTransform.from_mapping(mapping_getter())

    @wraps(mapping_getter)
    def apply(smt_key: z3.ExprRef) -> z3.ExprRef:
        return get_transform().smt_apply(smt_key)

    return apply


def _nth_char_map(casefn: Callable[[str], str], idx: int) -> Dict[int, int]:
    return {k: ord(v[idx]) for k, v in casemap(casefn).items() if len(v) > idx}


def _has_nth_char(casefn: Callable[[str], str], idx: int) -> CharMask:
    return make_mask(k for k, v in casemap(casefn).items() if len(v) > idx)


@smt_mask
def smt_is_ascii() -> CharMask:
    return CharMask([(0, 128)])


@smt_mask
def smt_is_alpha() -> CharMask:
    return get_unicode_mask("Lm", "Lt", "Lu", "Ll", "Lo")


@smt_mask
def smt_is_alnum() -> CharMask:
    alpha = get_unicode_mask("Lm", "Lt", "Lu", "Ll", "Lo")
    return alpha.union(get_char_set("has_numeric"))


@smt_mask
def smt_is_decimal() -> CharMask:
    return get_char_set("has_decimal")


@smt_mask
def smt_is_digit() -> CharMask:
    return get_char_set("has_digit")


@smt_mask
def smt_is_numeric() -> CharMask:
    return get_char_set("has_numeric")


@smt_mask
def smt_is_lower() -> CharMask:
    return get_char_set("islower")


@smt_mask
def smt_is_upper() -> CharMask:
    return get_char_set("isupper")


@smt_mask
def smt_is_title() -> CharMask:
    return get_char_set("istitle")


@smt_mask
def smt_is_printable() -> CharMask:
    printable = get_unicode_mask(
        "Cc", "Co", "Cn", "Cf", "Cs", "Zs", "Zl", "Zp", "Zs"
    ).invert()
    # The ascii space char is printable too:
    return printable.union(CharMask([32]))


@smt_mask
def smt_is_space() -> CharMask:
    return get_char_set("isspace")


@smt_mask
def smt_is_newline() -> CharMask:
    nls = "\n\x0b\x0c\r\x1c\x1d\x1e\x85\u2028\u2029"
    return CharMask(list(map(ord, nls)))


@smt_mask
def smt_casefold_changes() -> CharMask:
    return get_char_set("changed_by_casefold")


@smt_mask
def smt_casefold_has_2nd() -> CharMask:
    return _has_nth_char(str.casefold, 1)


@smt_mask
def smt_casefold_has_3rd() -> CharMask:
    return _has_nth_char(str.casefold, 2)


@smt_transform
def smt_casefold_1st() -> Dict[int, int]:
    return _nth_char_map(str.casefold, 0)


@smt_transform
def smt_casefold_2nd() -> Dict[int, int]:
    return _nth_char_map(str.casefold, 1)


@smt_transform
def smt_casefold_3rd() -> Dict[int, int]:
    return _nth_char_map(str.casefold, 2)


@smt_mask
def smt_lower_changes() -> CharMask:
    return get_char_set("changed_by_lower")


@smt_mask
def smt_lower_has_2nd() -> CharMask:
    return _has_nth_char(str.lower, 1)


@smt_transform
def smt_lower_1st() -> Dict[int, int]:
    return _nth_char_map(str.lower, 0)


@smt_transform
def smt_lower_2nd() -> Dict[int, int]:
    return _nth_char_map(str.lower, 1)


@smt_mask
def smt_title_changes() -> CharMask:
    return get_char_set("changed_by_title")


@smt_mask
def smt_title_has_2nd() -> CharMask:
    return _has_nth_char(str.title, 1)


@smt_mask
def smt_title_has_3rd() -> CharMask:
    return _has_nth_char(str.title, 2)


@smt_transform
def smt_title_1st() -> Dict[int, int]:
    return _nth_char_map(str.title, 0)


@smt_transform
def smt_title_2nd() -> Dict[int, int]:
    return _nth_char_map(str.title, 1)


@smt_transform
def smt_title_3rd() -> Dict[int, int]:
    return _nth_char_map(str.title, 2)


@smt_mask
def smt_upper_changes() -> CharMask:
    return get_char_set("changed_by_upper")


@smt_mask
def smt_upper_has_2nd() -> CharMask:
    return _has_nth_char(str.upper, 1)


@smt_mask
def smt_upper_has_3rd() -> CharMask:
    return _has_nth_char(str.upper, 2)


@smt_transform
def smt_upper_1st() -> Dict[int, int]:
    return _nth_char_map(str.upper, 0)


@smt_transform
def smt_upper_2nd() -> Dict[int, int]:
    return _nth_char_map(str.upper, 1)


@smt_transform
def smt_upper_3rd() -> Dict[int, int]:
    return _nth_char_map(str.upper, 2)


@smt_transform
def smt_decimal_value() -> Dict[int, int]:
    return nummap(decimal)


@smt_transform
def smt_digit_value() -> Dict[int, int]:
    return nummap(digit)


@smt_transform
def smt_numeric_numerator() -> Dict[int, int]:
    return {k: v[0] for k, v in fractionmap().items()}


@smt_transform
def smt_numeric_denominator() -> Dict[int, int]:
    return {k: v[1] for k, v in fractionmap().items()}


def _test_invert_symmetry(m: CharMask):
    """
    Check that double inversion is the same as the original.

    post: list(_.all_bounds()) == list(m.all_bounds())
    """
    return m.invert().invert()


def _test_intersection(left: CharMask, right: CharMask):
    """
    Check intersection behavior.

    post: _.covers(0) == (left.covers(0) and right.covers(0))
    post: _.covers(9) == (left.covers(9) and right.covers(9))
    """
    return left.intersect(right)


def _test_union(left: CharMask, right: CharMask):
    """
    Check union behavior.

    post: _.covers(0) == (left.covers(0) or right.covers(0))
    post: _.covers(9) == (left.covers(9) or right.covers(9))
    """
    return left.union(right)


def _test_set_operations(left: CharMask, right: CharMask):
    """
    Check: compliment of the intersection is the same as the union of both compliments.

    post: _[0] == _[1]
    """
    return (
        left.intersect(right).invert(),
        left.invert().union(right.invert()),
    )


if __name__ == "__main__":
    path = unicode_data_path(unidata_version)
    write_unicode_data(path, compute_unicode_data())
    print(f"Wrote {path}")
