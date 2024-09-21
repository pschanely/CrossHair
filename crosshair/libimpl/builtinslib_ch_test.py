import operator
import sys
from math import isnan
from numbers import Integral
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pytest  # type: ignore

from crosshair.core import realize
from crosshair.core_and_libs import MessageType, analyze_function, run_checkables
from crosshair.test_util import ResultComparison, compare_results, compare_returns

_TRICKY_UNICODE = (
    "A\u01f2",  # upper followed by title cased character
    "\ua770",  # Lm, lower (superscript "9")
    "\u01f2",  # Lt, title-cased but not upper (compound "Dz" like char)
    "\u2165",  # Nl (roman numeral "VI")
)


def check_abs(x: Union[int, bool, float]) -> ResultComparison:
    """post: _"""
    return compare_returns(abs, x)


def check_ascii(x: object) -> ResultComparison:
    """post: _"""
    return compare_results(ascii, x)


def check_bin(x: Integral) -> ResultComparison:
    """post: _"""
    return compare_results(bin, x)


def check_callable(x: object) -> ResultComparison:
    """post: _"""
    return compare_results(callable, x)


def check_chr(x: int) -> ResultComparison:
    """post: _"""
    return compare_results(chr, x)


# NOTE: dir() is not expected to be compatible.

_NAN_SENTINAL: Tuple = tuple()
# (anything that is copyable and compares equal with itself)


def check_divmod(
    numerator: Union[int, float], denominator: Union[int, float]
) -> ResultComparison:
    """post: _"""
    if numerator == 0 or denominator == 0:
        pass
    return compare_returns(
        lambda n, d: [_NAN_SENTINAL if isnan(x) else x for x in divmod(n, d)],
        numerator,
        denominator,
    )


def check_eval(expr: str):
    """
    pre: len(expr) == 1
    post: _
    """
    return compare_results(lambda e: eval(e, {}, {}), expr)


# NOTE: not patching exit()

# TODO: this fails; symbolic callables do not have correct behavior for
# inputs outside their expected domain.
# def check_filter(f: Callable[[int], bool], l: List[str]):
#    ''' post: _ '''
#    return compare_results(filter, f, l)


def check_float(o: Union[str, int, float]) -> ResultComparison:
    """post: _"""
    # TODO this isn't hitting most of the branches we care about right now.
    return compare_returns(float, o)


def check_float_sub(float_a: float, float_b: float) -> ResultComparison:
    """post: _"""
    return compare_returns(lambda a, b: a - b, float_a, float_b)


def check_format(x: object, f: str) -> ResultComparison:
    """post: _"""
    return compare_results(format, x, f)


# TODO: Add str and other types to check here
def check_format_dunder(obj: Union[int, float], fmt: str) -> ResultComparison:
    """post: _"""
    if fmt == "03d":
        pass
    elif fmt == "+.3f":
        pass
    # TODO: do not realize `fmt` here- instead we should intercept the native
    # __format__ calls to check for a symbolic format string.
    return compare_returns(lambda o, f: o.__format__(f), obj, realize(fmt))


# CrossHair proxies don't have the same attributes as native:
# def check_getattr(o: object, n: str, d: object) -> ResultComparison:

# NOTE: not patching globals()

# CrossHair proxies don't have the same attributes as native:
# def check_hasattr(o: str, n: str) -> ResultComparison:


def check_hash(o: Union[int, str, bytes, tuple]) -> ResultComparison:
    """post: _"""
    return compare_results(hash, o)


# We test frozenset explicitly because it's trickier:
def check_frozenset_hash(o: frozenset) -> ResultComparison:
    """post: _"""
    return compare_results(hash, o)


# NOTE: not patching help()


def check_hex(o: int) -> ResultComparison:
    """post: _"""
    return compare_results(hex, o)


# NOTE: not testing id()
# NOTE: not testing input()


def check_int(o: Union[str, int, float]) -> ResultComparison:
    """post: _"""
    return compare_returns(int, o)


def check_int_with_base(o: str, b: int) -> ResultComparison:
    """post: _"""
    return compare_results(int, o, b)


def check_isinstance(o: object, t: type) -> ResultComparison:
    """post: _"""
    return compare_results(isinstance, o, t)


def check_issubclass(o: object, t: type) -> ResultComparison:
    """post: _"""
    return compare_results(issubclass, o, t)


def check_iter(obj: Union[str, List[int], Dict[int, int]]) -> ResultComparison:
    """post: _"""
    # Note that we don't check Set[int] because of unstable ordering.
    return compare_results(lambda o: list(iter(o)), obj)


def check_len(
    s: Union[Dict[int, int], Tuple[int, ...], str, List[int], Set[int]]
) -> ResultComparison:
    """post: _"""
    return compare_results(len, s)


# NOTE: not testing license()
# NOTE: not testing locals()

# TODO: this fails; right now because symbolic callables have a bug that
# let's them realize inconsistently.
# def check_map(f: Callable[[int], str], l: List[int]):
#    ''' post: _ '''
#    return compare_results(map, f, l)


def check_max(
    x: Sequence, k: Optional[Callable[[Any], Any]], d: object
) -> ResultComparison:
    """post: _"""
    kw = {"default": d}
    if k is not None:
        kw["key"] = k
    return compare_results(max, x, **kw)


def check_min(x: Sequence) -> ResultComparison:
    """post: _"""
    return compare_results(min, x)


# NOTE: not testing next()


def check_oct(x: int) -> ResultComparison:
    """post: _"""
    return compare_results(oct, x)


# NOTE: not testing open()


def check_ord(x: str) -> ResultComparison:
    """post: _"""
    return compare_results(ord, x)


def check_print(o: object) -> ResultComparison:
    """post: _"""
    return compare_results(print, o)


def check_pow(
    b: Union[int, float], e: Union[int, float], m: Optional[int]
) -> ResultComparison:
    """post: _"""
    return compare_returns(pow, b, e, m)


# NOTE: not testing quit()


def check_range_len(start: int, stop: int, step: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda a, o, e: len(range(a, o, e)), start, stop, step)


def check_range_reversed(start: int, stop: int, step: int) -> ResultComparison:
    """post: _"""
    return compare_results(
        lambda a, o, e: list(reversed(range(a, o, e))), start, stop, step
    )


def check_reversed(obj: Union[List[int], Tuple[int]]) -> ResultComparison:
    """post: _"""
    return compare_results(lambda o: list(reversed(o)), obj)


def check_repr(o: object) -> ResultComparison:
    """post: _"""
    return compare_results(repr, o)


def check_round_float(o: float, d: Optional[int]) -> ResultComparison:
    """post: _"""
    return compare_returns(round, o, d)


def check_round_int(o: int, d: Optional[int]) -> ResultComparison:
    """post: _"""
    return compare_results(round, o, d)


# CrossHair proxies don't have the same attributes as native:
# def check_setattr(o: object, n: str, v: object) -> ResultComparison:


def check_sorted(s: Sequence) -> ResultComparison:
    """post: _"""
    return compare_results(sorted, s)


def check_sum(
    s: Union[Sequence[int], Sequence[float]],
    # i: Union[None, int, float]
) -> ResultComparison:
    """post: _"""
    return compare_returns(sum, s)


# NOTE: not testing vars()


def check_zip(s: Sequence[Sequence[int]]) -> ResultComparison:
    """post: _"""
    return compare_results(lambda args: zip(*args), s)


# Check list methods


def check_list_index(
    lst: List[int], item: int, start: int, stop: int
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda ln, *a: ln.index(*a), lst, item, start, stop)


def check_list_extend_and_slice(container: Union[List[int], bytearray]):
    """
    pre: len(container) <= 4
    post: _
    """

    def f(c):
        addition = [1, 2] if isinstance(container, list) else bytearray([1, 2])
        c += addition
        c = c[:3]
        del addition[0]  # This mutation hopefully doesn't change the result!
        return (c, type(c))

    return compare_results(f, container)


def check_list_setitem_slice(container: List[int], key: slice, replacement: List[int]):
    """
    pre: len(container) <= 3 and len(replacement) <= 3
    post: _
    """
    # crosshair: max_iterations=100

    def f(c, k, r):
        c[k] = r
        r.append(42)
        return c

    return compare_results(f, container, key, replacement)


# Check dict methods


def check_dict___init__(
    pos_args: List[Tuple[int, int]], kw_args: Dict[str, int]
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda a, kw: dict(*a, **kw), pos_args, kw_args)


def check_dict_get(dictionary: Dict[int, int], key: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda d, k: d.get(k), dictionary, key)


def check_dict_iter(dictionary: Dict[int, int]) -> ResultComparison:
    """post: _"""
    return compare_results(lambda d: list(d), dictionary)


def check_dict_clear(dictionary: Dict[int, int]) -> ResultComparison:
    """post: _"""

    def checker(d):
        d.clear()
        return d

    return compare_results(checker, dictionary)


def check_dict_pop(dictionary: Dict[int, int], key: int) -> ResultComparison:
    """post: _"""

    def checker(d, k):
        x = d.pop(k)
        return (x, d)

    return compare_results(checker, dictionary, key)


def check_dict_popitem(dictionary: Dict[int, int]) -> ResultComparison:
    """post: _"""

    def checker(d):
        x = d.popitem()
        return (x, d)

    return compare_results(checker, dictionary)


def check_dict_update(left: Dict[int, int], right: Dict[int, int]) -> ResultComparison:
    """post: _"""

    def checker(d1, d2):
        d1.update(d2)
        return d1

    return compare_results(checker, left, right)


def check_dict_values(dictionary: Dict[int, int]) -> ResultComparison:
    """post: _"""
    # TODO: value views compare false even with new views from the same dict.
    # Ensure we match this behavior.
    return compare_results(lambda d: list(d.values()), dictionary)


# check set/frozenset methods


def check_set_eq(setobj: Set[int]) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s, setobj)


def check_set_clear(setobj: Set[int]) -> ResultComparison:
    """post: _"""

    def checker(s):
        s.clear()
        return s

    return compare_results(checker, setobj)


def check_set_remove(setobj: Set[int], item: int) -> ResultComparison:
    """post: _"""

    def checker(s, i):
        s.remove(i)
        return realize(s)

    return compare_results(checker, setobj, item)


def check_set_add(setobj: Set[int], item: int) -> ResultComparison:
    """post: _"""

    def checker(s, i):
        s.add(i)
        return realize(s)

    return compare_results(checker, setobj, item)


def check_set_symmetric_difference_update(
    left: Set[int], right: Set[int]
) -> ResultComparison:
    """post: _"""

    def checker(left, right):
        left ^= right
        return sorted(left)

    return compare_results(checker, left, right)


def check_set_union_sorted(
    left: Union[Set[int], FrozenSet[int]], right: Union[Set[int], FrozenSet[int]]
) -> ResultComparison:
    """post: _"""
    # We check union-sorted, because realizing the set contents would suppress duplicates
    return compare_results(lambda lt, r: sorted(lt | r), left, right)


def check_set_difference(
    left: Union[Set[int], FrozenSet[int]], right: Union[Set[int], FrozenSet[int]]
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda lt, r: lt - r, left, right)


def check_set_intersection(
    left: Union[Set[int], FrozenSet[int]],
    # unlike operators, named methods (set.intersection) can take iterables:
    right: List[int],
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda lt, r: lt.intersection(r), left, right)


def check_set_compare(
    left: Union[Set[int], FrozenSet[int]], right: Union[Set[int], FrozenSet[int]]
) -> ResultComparison:
    """
    pre: len(left) + len(right) <= 4
    post: _
    """
    # crosshair: max_uninteresting_iterations=75
    # (running this a little longer - it's been able to detect deepcopy memo
    # keepalive issues in the past)
    return compare_results(lambda lt, r: lt <= r, left, right)


def check_set_bool(setobj: Union[Set[int], FrozenSet[int]]) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s, bool(setobj))


def check_seq_concat_and_slice(seq1: List[int], seq2: List[int], slc: slice):
    """post: _"""
    return compare_results(lambda s1, s2, c: (s1 + s2)[c], seq1, seq2, slc)


# Check int methods


if sys.version_info >= (3, 10):

    def check_int_bit_count(val: int):
        """post: _"""
        realize(val in range(-3, 3))
        return compare_results(lambda v: v.bit_count(), val)


def check_int_bit_length(val: int):
    """post: _"""
    realize(val in range(-3, 3))
    return compare_results(lambda v: v.bit_length(), val)


def check_int_to_bytes(val: int, big: bool, signed: bool):
    """post: _"""
    realize(val == 2**16)
    return compare_results(
        lambda v, b, s: v.to_bytes(2, "big" if b else "little", signed=s),
        val,
        big,
        signed,
    )


# Check string methods


def check_str_getitem_index(string: str, idx: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s[idx], string)


def check_str_getitem_slice(string: str, start: int, end: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s[start:end], string)


def check_str_capitalize(string: str) -> ResultComparison:
    """post: _"""
    if string in _TRICKY_UNICODE:
        pass
    return compare_results(lambda s: s.capitalize(), string)


def check_str_casefold(string: str) -> ResultComparison:
    """
    pre: len(string) <= 2
    post: _
    """
    return compare_results(lambda s: s.casefold(), string)


def check_str_center(string: str, size: int, fill: str) -> ResultComparison:
    """post: _"""
    if not string:
        pass
    if len(string) % 2 == 0:
        pass
    if size % 2 == 0:
        pass
    if fill == " ":
        pass
    return compare_results(lambda s, *a: s.center(*a), string, size, fill)


def check_str_contains(needle: str, haystack: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda n, h: n in h, needle, haystack)


def check_str_contains_against_literal(needle: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda n: n in "abc", needle)


def check_str_count(
    string: str, sub: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.count(*a), string, sub, start, end)


def check_str_encode_wild(string: str, encoding: str, errors: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.encode(*a), string, encoding, errors)


def check_str_endswith(
    string: str, suffix: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """post: _"""
    return compare_results(
        lambda s, *a: s.endswith(*a),
        string,
    )


def check_str_expandtabs(string: str, tabsize: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.expandtabs(*a), string, tabsize)


def check_str_find(
    big: str, little: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.find(*a), big, little, start, end)


def check_str_find_empty(big: str, start: int, end: int):
    """post: _"""
    # Lots of tricky edge cases when searching for an empty string.
    # Target these cases more narrowly.
    if big != "":
        return True
    return compare_results(lambda s, *a: s.find("", *a), big, start, end)


def check_str_fstring(string: str, num: int, lst: List[int]) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, n, ls: f"{n:02d}{s}{ls!r}", string, num, lst)


def check_str_format(string: str, *args: object, **kwargs: object) -> ResultComparison:
    """post: _"""
    return compare_results(
        lambda s, *a, **kw: s.format(*a, **kw), string, *args, **kwargs
    )


def check_str_format_map(string: str, mapping: Mapping) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.format_map(*a), string, mapping)


def check_str_index(
    string: str, sub: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.index(*a), string, sub, start, end)


def check_str_isalpha(s: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s.isalpha(), s)


def check_str_isalnum(string: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s.isalnum(), string)


def check_str_isascii(string: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s.isascii(), string)


def check_str_isdecimal(string: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s.isdecimal(), string)


def check_str_isdigit(string: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s.isdigit(), string)


def check_str_isidentifier(string: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s.isidentifier(), string)


def check_str_islower(string: str) -> ResultComparison:
    """post: _"""
    if string in _TRICKY_UNICODE:
        pass
    return compare_results(lambda s: s.islower(), string)


def check_str_isnumeric(string: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s.isnumeric(), string)


def check_str_isprintable(string: str) -> ResultComparison:
    """post: _"""
    if string in (" ", "\n", "\x01"):
        pass
    return compare_results(lambda s: s.isprintable(), string)


def check_str_isspace(string: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s: s.isspace(), string)


def check_str_istitle(string: str) -> ResultComparison:
    """
    pre: len(string) <= 3
    post: _
    """
    if string in _TRICKY_UNICODE:
        pass
    return compare_results(lambda s: s.istitle(), string)


def check_str_isupper(string: str) -> ResultComparison:
    """post: _"""
    if string in _TRICKY_UNICODE:
        pass
    return compare_results(lambda s: s.isupper(), string)


def check_str_join(string: str, seq: Sequence[str]) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.join(*a), string, seq)


def check_str_ljust(string: str, width: int, fill: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.ljust(*a), string, width, fill)


def check_str_lower(string: str) -> ResultComparison:
    """
    pre: len(string) <= 2
    post: _
    """
    return compare_results(lambda s, *a: s.lower(*a), string)


def check_str_lstrip(string: str, chars: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.lstrip(*a), string, chars)


def check_str_partition(string: str, sep: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.partition(*a), string, sep)


def check_str_replace(
    string: str, old: str, new: str, maxsplit: int
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.replace(*a), string, old, new, maxsplit)


def check_str_rfind(
    string: str, substr: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.rfind(*a), string, substr, start, end)


def check_str_rfind_empty(big: str, start: int, end: int):
    """post: _"""
    # Lots of tricky edge cases when searching for an empty string.
    # Target these cases more narrowly.
    if big != "":
        return True
    return compare_results(lambda s, *a: s.rfind("", *a), big, start, end)


def check_str_rindex(
    string: str, sub: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.rindex(*a), string, sub, start, end)


def check_str_rjust(string: str, width: int, fill: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.rjust(*a), string, width, fill)


def check_str_rpartition(string: str, sep: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.rpartition(*a), string, sep)


def check_str_rsplit(string: str, sep: str, maxsplit: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.rsplit(*a), string, sep, maxsplit)


def check_str_rstrip(string: str, chars: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.rstrip(*a), string, chars)


def check_str_split(string: str, sep: str, maxsplit: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.split(*a), string, sep, maxsplit)


def check_str_splitlines(string: str, keepends: bool) -> ResultComparison:
    """post: _"""
    if "\r" in string:
        pass
    return compare_results(lambda s, *a: s.splitlines(*a), string, keepends)


def check_str_startswith(
    string: str,
    prefix: Union[str, Tuple[str, ...]],
    start: Optional[int],
    end: Optional[int],
) -> ResultComparison:
    """post: _"""
    return compare_results(
        lambda s, *a, **kw: s.startswith(*a, **kw), string, prefix, start, end
    )


def check_str_strip(string: str, chars: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.strip(*a), string, chars)


def check_str_swapcase(string: str):
    """post: _"""
    if string not in ("Ab", "\u01f2"):
        return True
    return compare_results(lambda s: s.swapcase(), string)


def check_str_title(string: str):
    """post: _"""
    if string not in ("A\u01f2", "aA"):
        return True
    return compare_results(lambda s: s.title(), string)


def check_str_translate(
    string: str, tbl: Union[Mapping[int, int], List[str]]
) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.translate(*a), string, tbl)


def check_str_upper(string: str) -> ResultComparison:
    """
    pre: len(string) <= 2
    post: _
    """
    return compare_results(lambda s: s.upper(), string)


def check_str_zfill(string: str, width: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda s, *a: s.zfill(*a), string, width)


if sys.version_info >= (3, 9):

    def check_str_removeprefix(s: str, prefix: str):
        """post: _"""
        return compare_results(lambda s, *a: s.removeprefix(*a), s, prefix)

    def check_str_removesuffix(s: str, suffix: str):
        """post: _"""
        return compare_results(lambda s, *a: s.removesuffix(*a), s, suffix)


# Check bytes, bytearray, memoryview methods


def check_buffer_getitem_return_type(container: Union[bytes, bytearray, memoryview]):
    """post: _"""
    return compare_results(lambda c: type(c[:1]), container)


def check_buffer_hex_noargs(container: Union[bytes, bytearray, memoryview]):
    """post: _"""
    return compare_results(lambda c: c.hex(), container)


def check_buffer_hex(
    container: Union[bytes, bytearray, memoryview], sep: str, bytes_per_sep: int
):
    """post: _"""
    return compare_results(lambda c, s, b: c.hex(s, b), container, sep, bytes_per_sep)


def check_buffer_fromhex(hexstr: str):
    """post: _"""
    return compare_results(lambda s: bytes.fromhex(s), hexstr)


def check_buffer_setitem_splice(container: bytearray):
    """post: _"""

    def setter(c):
        c[0:0] = [42]
        return c

    return compare_results(setter, container)


def check_buffer_setitem_add_self(container: memoryview):
    """post: _"""

    def setter(c):
        c[0:0] = c
        return c

    return compare_results(setter, container)


def check_buffer_setitem_replace(
    container: Union[memoryview, bytearray],
    replacement: Union[memoryview, bytearray, bytes],
    realize_at: int,
):
    """post: _"""

    def setter(c, r):
        if r == 0:
            c = realize(c)
        elif r == 1:
            r = realize(r)
        c[0:1] = r
        return c

    return compare_results(setter, container, replacement)


def check_buffer_crosstype_addition(
    buffer1: Union[bytes, bytearray, memoryview],
    buffer2: Union[bytes, bytearray, memoryview],
    realize_at: int,
):
    """post: _"""

    def adder(b1, b2, r):
        if r == 0:
            b1 = realize(b1)
        elif r == 1:
            b2 = realize(b2)
        return b1 + b2

    return compare_results(adder, buffer1, buffer2, realize_at)


def check_buffer_add_return_type(container: Union[bytearray, memoryview]):
    """post: _"""
    return compare_results(lambda c: type(c + b"abc"), container)


def check_bytes___init__(source: Union[int, List[int], bytes, bytearray, memoryview]):
    """
    post: _
    raises: KeyError
    """
    return compare_results(lambda c, s: c(s), bytes, source)


def check_bytearray___init__(
    source: Union[int, List[int], bytes, bytearray, memoryview]
):
    """
    post: _
    raises: KeyError
    """
    return compare_results(lambda c, s: c(s), bytearray, source)


def check_memoryview___init__(
    source: Union[int, List[int], bytes, bytearray, memoryview]
):
    """
    post: _
    raises: KeyError
    """
    return compare_results(lambda c, s: c(s), memoryview, source)


def check_buffer_iter(container: Union[bytes, bytearray, memoryview]):
    """post: _"""
    return compare_results(list, container)


def check_buffer_equal(
    buffer1: Union[bytes, bytearray, memoryview],
    buffer2: Union[bytes, bytearray, memoryview],
    realize_at: int,
):
    """post: _"""

    def compare(b1, b2, r):
        if r == 0:
            b1 = realize(b1)
        elif r == 1:
            b2 = realize(b2)
        return b1 == b2

    return compare_results(compare, buffer1, buffer2, realize_at)


def check_buffer_compare(
    buffer1: Union[bytes, bytearray, memoryview],
    buffer2: Union[bytes, bytearray, memoryview],
    realize_at: int,
):
    """post: _"""

    def compare(b1, b2, r):
        if r == 0:
            b1 = realize(b1)
        elif r == 1:
            b2 = realize(b2)
        # A lot of esotric Python behaviors in (<); see comments in BytesLike._cmp_op.
        return b1 < b2

    return compare_results(compare, buffer1, buffer2, realize_at)


def check_buffer_percent_format(buffer: Union[bytes, bytearray, memoryview]):
    """post: _"""
    return compare_results(lambda b: b"%04b" % b, buffer)


def check_memoryview_conversions(view: memoryview):
    """post: _"""
    if len(view) == 1:
        pass
    return compare_results(
        lambda mv: (mv.tobytes(), mv.tolist(), mv.hex(), mv.cast("b")), view
    )


# Check operators


def check_add_seqs(seq: Union[str, bytes, bytearray, List[int], Tuple[int, ...]]):
    """
    pre: len(seq) == 1
    post: _
    """

    def f(s):
        combined = s + s
        return (combined, type(combined))

    return compare_results(f, seq)


def check_and(left: int):
    """post: _"""
    return compare_results(lambda lt: (lt & 3, 4 & lt), left)


def check_truediv(left: Union[int, float], right: Union[int, float]):
    """post: _"""
    return compare_returns(operator.truediv, left, right)


def check_lt_strings(left: str, right: str):
    """post: _"""
    return compare_results(operator.lt, left, right)


def check_ge_numeric(
    left: Union[int, bool, float, complex], right: Union[int, bool, float, complex]
):
    """post: _"""
    return compare_returns(operator.ge, left, right)


def check_mod(
    left: Union[int, bool, float, complex], right: Union[int, bool, float, complex]
):
    """post: _"""
    if left == 0 or right == 0:
        pass
    return compare_returns(operator.mod, left, right)


def check_floordiv(x: Union[int, float], y: Union[int, float]) -> ResultComparison:
    """post: _"""
    if y == 0 or x == 0:
        pass
    return compare_returns(operator.floordiv, x, y)


def check_getitem(
    container: Union[Dict[int, int], List[int], Tuple[int, ...]], key: int
):
    """post: _"""
    return compare_results(lambda d, k: d[k], container, key)


def check_getitem_slice(
    container: Union[List[int], Tuple[int, ...], str, bytes, bytearray], key: slice
):
    """post: _"""

    def f(d, s):
        ret = d[s]
        return (ret, type(ret))

    return compare_results(f, container, key)


def check_delitem_int(container: Union[Dict[int, int], List[int]], key: int):
    """post: _"""

    def checker(d, k):
        del d[k]
        return d

    return compare_results(checker, container, key)


def check_delitem_slice(container: List[int], key: slice):
    """post: _"""

    def checker(d, k):
        del d[k]
        return d

    return compare_results(checker, container, key)


def check_inplace_mutation(container: Union[bytearray, List[int], Dict[int, int]]):
    """post: _"""

    def setter(c):
        if c:
            c[0] &= 42
        return c

    return compare_results(setter, container)


def check_eq_atomic(
    left: Union[bool, int, float, str], right: Union[bool, int, float, str]
):
    """post: _"""
    return compare_returns(lambda a, b: a == b, left, right)


def check_trunc(num: Union[bool, int, float]):
    """post: _"""
    if num >= 0:
        pass
    return compare_returns(lambda n: n.__trunc__(), num)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    this_module = sys.modules[__name__]
    fn = getattr(this_module, fn_name)
    messages = run_checkables(analyze_function(fn))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
