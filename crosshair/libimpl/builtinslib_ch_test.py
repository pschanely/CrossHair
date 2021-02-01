from numbers import Integral
from typing import *
import sys

from crosshair.core_and_libs import analyze_module
from crosshair.core_and_libs import AnalysisOptions
from crosshair.core_and_libs import MessageType
from crosshair.test_util import compare_results
from crosshair.test_util import ResultComparison


# This file only has one "test"; it runs crosshair on itself.
# To debug, you can just run crosshair on individual functions; i.e.:
#
# $ crosshair check crosshair.libimpl.builtinslib.builtinslib_chtest.check_<something>
#
def test_builtins():
    opts = AnalysisOptions(
        max_iterations=5,
        per_condition_timeout=10)
    messages = analyze_module(sys.modules[__name__], opts)
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []


def check_abs(x: float) -> ResultComparison:
    ''' post: _ '''
    return compare_results(abs, x)

def check_ascii(x: object) -> ResultComparison:
    ''' post: _ '''
    return compare_results(ascii, x)

def check_bin(x: Integral) -> ResultComparison:
    ''' post: _ '''
    return compare_results(bin, x)

def check_callable(x: object) -> ResultComparison:
    ''' post: _ '''
    return compare_results(callable, x)

def check_chr(x: int) -> ResultComparison:
    ''' post: _ '''
    return compare_results(chr, x)

# NOTE: dir() is not expected to be compatible.

def check_divmod(x: Union[int, float]) -> ResultComparison:
    ''' post: _ '''
    return compare_results(divmod, x)

def check_eval(e: str, g: Optional[Dict[str, Any]], l: Optional[Dict[str, Any]]):
    '''
    pre: len(e) == 1
    post: _
    '''
    return compare_results(eval, e, {}, {})

# NOTE: not patching exit()

# TODO: this fails; symbolic callables do not have correct behavior for
# inputs outside their expected domain.
#def check_filter(f: Callable[[int], bool], l: List[str]):
#    ''' post: _ '''
#    return compare_results(filter, f, l)

def check_format(x: object, f: str) -> ResultComparison:
    ''' post: _ '''
    return compare_results(format, x, f)

# CrossHair proxies don't have the same attributes as native:
#def check_getattr(o: object, n: str, d: object) -> ResultComparison:

# NOTE: not patching globals()

# CrossHair proxies don't have the same attributes as native:
#def check_hasattr(o: str, n: str) -> ResultComparison:

def check_hash(o: object) -> ResultComparison:
    ''' post: _ '''
    return compare_results(hash, o)

# NOTE: not patching help()

def check_hex(o: int) -> ResultComparison:
    ''' post: _ '''
    return compare_results(hex, o)

# NOTE: not testing id()
# NOTE: not testing input()

def check_isinstance(o: object, t: type) -> ResultComparison:
    ''' post: _ '''
    return compare_results(isinstance, o, t)

def check_issubclass(o: object, t: type) -> ResultComparison:
    ''' post: _ '''
    return compare_results(issubclass, o, t)

def check_iter(i: Union[List, Set, Dict]) -> ResultComparison:
    ''' post: _ '''
    return compare_results(iter, i)

def check_len(s: Sized) -> ResultComparison:
    ''' post: _ '''
    return compare_results(len, s)

# NOTE: not testing license()
# NOTE: not testing locals()

# TODO: this fails; right now because symbolic callables have a bug that
# let's them realize inconsistently.
#def check_map(f: Callable[[int], str], l: List[int]):
#    ''' post: _ '''
#    return compare_results(map, f, l)

def check_max(x: Sequence, k: Optional[Callable[[Any], Any]], d: object) -> ResultComparison:
    ''' post: _ '''
    return compare_results(max, x, k, d)

def check_min(x: Sequence) -> ResultComparison:
    ''' post: _ '''
    return compare_results(min, x)

# NOTE: not testing next()

def check_oct(x: int) -> ResultComparison:
    ''' post: _ '''
    return compare_results(oct, x)

# NOTE: not testing open()

def check_ord(x: str) -> ResultComparison:
    ''' post: _ '''
    return compare_results(ord, x)

def check_print(o: object) -> ResultComparison:
    ''' post: _ '''
    return compare_results(print, o)

def check_pow(b: Union[int, float], e: Union[int, float], m: Optional[int]) -> ResultComparison:
    ''' post: _ '''
    return compare_results(pow, b, e, m)

# NOTE: not testing quit()

def check_reversed(o: Union[List, Tuple]) -> ResultComparison:
    ''' post: _ '''
    return compare_results(reversed, o)

def check_repr(o: object) -> ResultComparison:
    ''' post: _ '''
    return compare_results(repr, o)

def check_round(o: Union[float, int], d: Optional[int]) -> ResultComparison:
    ''' post: _ '''
    return compare_results(round, o, d)

# CrossHair proxies don't have the same attributes as native:
#def check_setattr(o: object, n: str, v: object) -> ResultComparison:

def check_sorted(s: Sequence) -> ResultComparison:
    ''' post: _ '''
    return compare_results(sorted, s)

def check_sum(s: Union[Sequence[int], Sequence[float]],
             #i: Union[None, int, float]
) -> ResultComparison:
    ''' post: _ '''
    return compare_results(sum, s)

# NOTE: not testing vars()

def check_zip(s: Sequence[Sequence[int]]) -> ResultComparison:
    ''' post: _ '''
    return compare_results(lambda args: zip(*args), s)

