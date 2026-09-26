import json
import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, partial, wraps
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

from crosshair.z3util import z3And, z3App, z3Eq, z3Gt, z3IntVal, z3Le, z3Or


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

    def interpret_smt_function(self, smt_fn: z3.ExprRef) -> z3.ExprRef:
        """Return an interpretation for an smt (Int->Bool) function for this mask."""
        codepoint = z3.Int("c")
        return z3.ForAll(
            [codepoint],
            smt_fn(codepoint) == self.smt_matches(codepoint),
            patterns=[smt_fn(codepoint)],  # (always expand)
        )

    def smt_matches(self, smt_ch: z3.ExprRef):
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


_INTERPRETATION_CACHE: Dict[str, z3.ExprRef] = {}


def _cached_int_transform(name: str, transforms: Dict[int, int]) -> z3.ExprRef:
    if name in _INTERPRETATION_CACHE:
        return _INTERPRETATION_CACHE[name]
    else:
        smt_fn = z3.Function(name, z3.IntSort(), z3.IntSort())
        val_to_key = defaultdict(list)
        for k, v in transforms.items():
            val_to_key[v].append(k)
        interpretation = z3And(
            *[
                z3Eq(z3App(smt_fn, z3IntVal(k)), z3IntVal(val))
                for (val, keys) in val_to_key.items()
                for k in keys
            ]
        )
        _INTERPRETATION_CACHE[name] = interpretation
        return interpretation


def _cached_mask_interpretation(
    name: str, mask_getter: Callable[[], CharMask]
) -> Tuple[z3.ExprRef, Callable[[z3.ExprRef], z3.ExprRef]]:
    if name in _INTERPRETATION_CACHE:
        return _INTERPRETATION_CACHE[name]
    else:
        mask = mask_getter()
        smt_fn = z3.Function("is_" + name, z3.IntSort(), z3.BoolSort())
        interpretation = mask.interpret_smt_function(smt_fn)
        ret = (interpretation, smt_fn)
        _INTERPRETATION_CACHE[name] = ret
        return ret


def transform_fn(transformer):
    name = transformer.__name__

    @wraps(transformer)
    def wrapper(self) -> z3.ExprRef:
        if name in self._cached_smt_fns:
            return self._cached_smt_fns[name]
        self.solver.add(_cached_int_transform(name, transformer(self)))
        smt_fn = z3.Function(name, z3.IntSort(), z3.IntSort())
        self._cached_smt_fns[name] = smt_fn
        return smt_fn

    return wrapper


def mask_fn(mask_getter):
    name = mask_getter.__name__

    @wraps(mask_getter)
    def wrapper(self, *a) -> z3.ExprRef:
        if name in self._cached_smt_fns:
            return self._cached_smt_fns[name]
        constr, checker = _cached_mask_interpretation(name, partial(mask_getter, self))
        self.solver.add(constr)
        self._cached_smt_fns[name] = checker
        return checker

    return wrapper


class UnicodeMaskCache:
    def __init__(self, solver: z3.Solver):
        self.solver = solver
        self._cached_smt_fns: Dict[str, z3.FuncDeclRef] = {}

    @mask_fn
    def ascii(self):
        return CharMask([(0, 128)])

    @mask_fn
    def alnum(self):
        alpha = get_unicode_mask("Lm", "Lt", "Lu", "Ll", "Lo")
        return alpha.union(get_char_set("has_numeric"))

    @mask_fn
    def alpha(self):
        return get_unicode_mask("Lm", "Lt", "Lu", "Ll", "Lo")

    @mask_fn
    def decimal(self):
        return get_unicode_mask("Nd")

    @mask_fn
    def digit(self):
        return get_char_set("has_digit")

    @mask_fn
    def numeric(self):
        return get_char_set("has_numeric")

    @mask_fn
    def lower(self):
        return get_char_set("islower")

    @mask_fn
    def printable(self):
        printable = get_unicode_mask(
            "Cc", "Co", "Cn", "Cf", "Cs", "Zs", "Zl", "Zp", "Zs"
        ).invert()
        # The ascii space char is printable too:
        printable = printable.union(CharMask([32]))
        return printable

    @mask_fn
    def space(self):
        return get_char_set("isspace")

    @mask_fn
    def newline(self):
        nls = (
            "\n",
            "\x0b",
            "\x0c",
            "\r",
            "\x1c",
            "\x1d",
            "\x1e",
            "\x85",
            "\u2028",
            "\u2029",
        )
        return CharMask(list(map(ord, nls)))

    @mask_fn
    def upper(self):
        return get_char_set("isupper")

    @mask_fn
    def title(self):
        return get_char_set("istitle")

    @mask_fn
    def word(self):
        return get_char_set("word")

    @mask_fn
    def casefold_exists(self):
        return get_char_set("changed_by_casefold")

    @transform_fn
    def casefold_1st(self):
        return {k: ord(v[0]) for k, v in casemap(str.casefold).items()}

    @mask_fn
    def casefold_2nd_exists(self):
        return make_mask(k for k, v in casemap(str.casefold).items() if len(v) >= 2)

    @transform_fn
    def casefold_2nd(self):
        return {k: ord(v[1]) for k, v in casemap(str.casefold).items() if len(v) >= 2}

    @mask_fn
    def casefold_3rd_exists(self):
        return make_mask(k for k, v in casemap(str.casefold).items() if len(v) >= 3)

    @transform_fn
    def casefold_3rd(self):
        return {k: ord(v[2]) for k, v in casemap(str.casefold).items() if len(v) >= 3}

    @mask_fn
    def tolower_exists(self):
        return get_char_set("changed_by_lower")

    @transform_fn
    def tolower_1st(self):
        return {k: ord(v[0]) for k, v in casemap(str.lower).items()}

    @mask_fn
    def tolower_2nd_exists(self):
        return make_mask(k for k, v in casemap(str.lower).items() if len(v) >= 2)

    @transform_fn
    def tolower_2nd(self):
        return {k: ord(v[1]) for k, v in casemap(str.lower).items() if len(v) >= 2}

    @mask_fn
    def totitle_exists(self):
        return get_char_set("changed_by_title")

    @transform_fn
    def totitle_1st(self):
        return {k: ord(v[0]) for k, v in casemap(str.title).items()}

    @mask_fn
    def totitle_2nd_exists(self):
        return make_mask(k for k, v in casemap(str.title).items() if len(v) >= 2)

    @transform_fn
    def totitle_2nd(self):
        return {k: ord(v[1]) for k, v in casemap(str.title).items() if len(v) >= 2}

    @mask_fn
    def totitle_3rd_exists(self):
        return make_mask(k for k, v in casemap(str.title).items() if len(v) >= 3)

    @transform_fn
    def totitle_3rd(self):
        return {k: ord(v[2]) for k, v in casemap(str.title).items() if len(v) >= 3}

    @mask_fn
    def toupper_exists(self):
        return get_char_set("changed_by_upper")

    @transform_fn
    def toupper_1st(self):
        return {k: ord(v[0]) for k, v in casemap(str.upper).items()}

    @mask_fn
    def toupper_2nd_exists(self):
        return make_mask(k for k, v in casemap(str.upper).items() if len(v) >= 2)

    @transform_fn
    def toupper_2nd(self):
        return {k: ord(v[1]) for k, v in casemap(str.upper).items() if len(v) >= 2}

    @mask_fn
    def toupper_3rd_exists(self):
        return make_mask(k for k, v in casemap(str.upper).items() if len(v) >= 3)

    @transform_fn
    def toupper_3rd(self):
        return {k: ord(v[2]) for k, v in casemap(str.upper).items() if len(v) >= 3}

    @mask_fn
    def digit_exists(self):
        return get_char_set("has_digit")

    @transform_fn
    def decimal_int(self):
        return nummap(decimal)

    @mask_fn
    def decimal_exists(self):
        return get_char_set("has_decimal")

    @transform_fn
    def digit_int(self):
        return nummap(digit)

    @mask_fn
    def numeric_exists(self):
        return get_char_set("has_numeric")

    @transform_fn
    def numeric_numerator(self):
        return {k: v[0] for k, v in fractionmap().items()}

    @transform_fn
    def numeric_denominator(self):
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
