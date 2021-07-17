from numbers import Integral
from typing import *
import sys

import pytest  # type: ignore

from crosshair.core import realize
from crosshair.core_and_libs import analyze_function
from crosshair.core_and_libs import run_checkables
from crosshair.core_and_libs import MessageType
from crosshair.options import AnalysisOptionSet
from crosshair.options import DEFAULT_OPTIONS
from crosshair.test_util import compare_results
from crosshair.test_util import ResultComparison


def check_abs(x: float) -> ResultComparison:
    """ post: _ """
    return compare_results(abs, x)


def check_ascii(x: object) -> ResultComparison:
    """ post: _ """
    return compare_results(ascii, x)


def check_bin(x: Integral) -> ResultComparison:
    """ post: _ """
    return compare_results(bin, x)


def check_callable(x: object) -> ResultComparison:
    """ post: _ """
    return compare_results(callable, x)


def check_chr(x: int) -> ResultComparison:
    """ post: _ """
    return compare_results(chr, x)


# NOTE: dir() is not expected to be compatible.


def check_divmod(x: Union[int, float]) -> ResultComparison:
    """ post: _ """
    return compare_results(divmod, x)


def check_eval(e: str, g: Optional[Dict[str, Any]], l: Optional[Dict[str, Any]]):
    """
    pre: len(e) == 1
    post: _
    """
    return compare_results(eval, e, {}, {})


# NOTE: not patching exit()

# TODO: this fails; symbolic callables do not have correct behavior for
# inputs outside their expected domain.
# def check_filter(f: Callable[[int], bool], l: List[str]):
#    ''' post: _ '''
#    return compare_results(filter, f, l)


def check_format(x: object, f: str) -> ResultComparison:
    """ post: _ """
    return compare_results(format, x, f)


# CrossHair proxies don't have the same attributes as native:
# def check_getattr(o: object, n: str, d: object) -> ResultComparison:

# NOTE: not patching globals()

# CrossHair proxies don't have the same attributes as native:
# def check_hasattr(o: str, n: str) -> ResultComparison:


def check_hash(o: object) -> ResultComparison:
    """ post: _ """
    return compare_results(hash, o)


# NOTE: not patching help()


def check_hex(o: int) -> ResultComparison:
    """ post: _ """
    return compare_results(hex, o)


# NOTE: not testing id()
# NOTE: not testing input()


def check_int(o: Union[str, int, float]) -> ResultComparison:
    """ post: _ """
    return compare_results(int, o)


def check_int_with_base(o: str, b: int) -> ResultComparison:
    """ post: _ """
    return compare_results(int, o, b)


def check_isinstance(o: object, t: type) -> ResultComparison:
    """ post: _ """
    return compare_results(isinstance, o, t)


def check_issubclass(o: object, t: type) -> ResultComparison:
    """ post: _ """
    return compare_results(issubclass, o, t)


def check_iter(i: Union[List[int], Set[int], Dict[int, int]]) -> ResultComparison:
    """ post: _ """
    return compare_results(iter, i)


def check_len(s: Sized) -> ResultComparison:
    """ post: _ """
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
    """ post: _ """
    kw = {"default": d}
    if k is not None:
        kw["key"] = k
    return compare_results(max, x, **kw)


def check_min(x: Sequence) -> ResultComparison:
    """ post: _ """
    return compare_results(min, x)


# NOTE: not testing next()


def check_oct(x: int) -> ResultComparison:
    """ post: _ """
    return compare_results(oct, x)


# NOTE: not testing open()


def check_ord(x: str) -> ResultComparison:
    """ post: _ """
    return compare_results(ord, x)


def check_print(o: object) -> ResultComparison:
    """ post: _ """
    return compare_results(print, o)


def check_pow(
    b: Union[int, float], e: Union[int, float], m: Optional[int]
) -> ResultComparison:
    """ post: _ """
    return compare_results(pow, b, e, m)


# NOTE: not testing quit()


def check_reversed(o: Union[List[int], Tuple[int]]) -> ResultComparison:
    """ post: _ """
    return compare_results(reversed, o)


def check_repr(o: object) -> ResultComparison:
    """ post: _ """
    return compare_results(repr, o)


def check_round(o: Union[float, int], d: Optional[int]) -> ResultComparison:
    """ post: _ """
    return compare_results(round, o, d)


# CrossHair proxies don't have the same attributes as native:
# def check_setattr(o: object, n: str, v: object) -> ResultComparison:


def check_sorted(s: Sequence) -> ResultComparison:
    """ post: _ """
    return compare_results(sorted, s)


def check_sum(
    s: Union[Sequence[int], Sequence[float]],
    # i: Union[None, int, float]
) -> ResultComparison:
    """ post: _ """
    return compare_results(sum, s)


# NOTE: not testing vars()


def check_zip(s: Sequence[Sequence[int]]) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda args: zip(*args), s)


# Check string methods


def check_str_getitem_index(string: str, idx: int) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s[idx], string)


def check_str_getitem_slice(string: str, start: int, end: int) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s[start:end], string)


def check_str_capitalize(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.capitalize(), string)


def check_str_casefold(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.casefold(), string)


def check_str_center(string: str, fill: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.center(*a), string, fill)


def check_str_contains(needle: str, haystack: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda n, h: n in h, needle, haystack)


def check_str_count(
    string: str, sub: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.count(*a), string, sub, start, end)


def check_str_encode(string: str, encoding: str, errors: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.encode(*a), string, encoding, errors)


def check_str_endswith(
    string: str, suffix: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """ post: _ """
    return compare_results(
        lambda s, *a: s.endswith(*a),
        string,
    )


def check_str_expandtabs(string: str, tabsize: int) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.expandtabs(*a), string, tabsize)


def check_str_find(
    s: str, sub: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.find(*a), sub, start, end)


def check_str_format(string: str, *args: object, **kwargs: object) -> ResultComparison:
    """ post: _ """
    return compare_results(
        lambda s, *a, **kw: s.format(*a, **kw), string, *args, **kwargs
    )


def check_str_format_map(string: str, mapping: Mapping) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.format_map(*a), string, mapping)


def check_str_index(
    string: str, sub: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.index(*a), string, sub, start, end)


def check_str_isalpha(s: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.isalpha(), s)


def check_str_isalnum(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.isalnum(), string)


def check_str_isascii(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.isascii(), string)


def check_str_isdecimal(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.isdecimal(), string)


def check_str_isdigit(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.isdigit(), string)


def check_str_isidentifier(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.isidentifier(), string)


def check_str_islower(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.islower(), string)


def check_str_isnumeric(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.isnumeric(), string)


def check_str_isprintable(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.isprintable(), string)


def check_str_isspace(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.isspace(), string)


def check_str_istitle(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.istitle(), string)


def check_str_isupper(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.isupper(), string)


def check_str_join(string: str, seq: Sequence[str]) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.join(*a), string, seq)


def check_str_ljust(string: str, width: int, fill: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.ljust(*a), string, width, fill)


def check_str_lower(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.lower(*a), string)


def check_str_lstrip(string: str, chars: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.lstrip(*a), string, chars)


def check_str_partition(string: str, sep: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.partition(*a), string, sep)


def check_str_replace(
    string: str, old: str, new: str, maxsplit: int
) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.replace(*a), string, old, new, maxsplit)


def check_str_rfind(
    string: str, substr: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.rfind(*a), string, substr, start, end)


def check_str_rindex(
    string: str, sub: str, start: Optional[int], end: Optional[int]
) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.rindex(*a), string, sub, start, end)


def check_str_rjust(string: str, width: int, fill: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.rjust(*a), string, width, fill)


def check_str_rpartition(string: str, sep: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.rpartition(*a), string, sep)


def check_str_rsplit(string: str, sep: str, maxsplit: int) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.rsplit(*a), string, sep, maxsplit)


def check_str_rstrip(string: str, chars: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.rstrip(*a), string, chars)


def check_str_split(string: str, sep: str, maxsplit: int) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.split(*a), string, sep, maxsplit)


def check_str_splitlines(string: str, keepends: bool) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.splitlines(*a), string, keepends)


def check_str_startswith(
    string: str,
    prefix: Union[str, Tuple[str, ...]],
    start: Optional[int],
    end: Optional[int],
) -> ResultComparison:
    """ post: _ """
    return compare_results(
        lambda s, *a, **kw: s.startswith(*a, **kw), string, prefix, start, end
    )


def check_str_strip(string: str, chars: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.strip(*a), string, chars)


def check_str_swapcase(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.swapcase(), string)


def check_str_title(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.title(), string)


def check_str_translate(string: str, tbl: Mapping[int, int]) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.translate(*a), string, tbl)


def check_str_upper(string: str) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s: s.upper(), string)


def check_str_zfill(string: str, width: int) -> ResultComparison:
    """ post: _ """
    return compare_results(lambda s, *a: s.zfill(*a), string, width)


def check_str_removeprefix(s: str, prefix: str):
    """ post: _ """
    return compare_results(lambda s, *a: s.removeprefix(*a), s, prefix)


def check_str_removesuffix(s: str, suffix: str):
    """ post: _ """
    return compare_results(lambda s, *a: s.removesuffix(*a), s, suffix)


# Check operators


def check_getitem(
    container: Union[Dict[int, int], List[int], Tuple[int, ...]], key: int
):
    """ post: _ """
    return compare_results(lambda d, k: d[k], container, key)


# TODO: check dictionary and other container!

# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    opts = AnalysisOptionSet(
        max_iterations=20, per_condition_timeout=10, per_path_timeout=10
    )
    this_module = sys.modules[__name__]
    fn = getattr(this_module, fn_name)
    messages = run_checkables(analyze_function(fn, opts))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
