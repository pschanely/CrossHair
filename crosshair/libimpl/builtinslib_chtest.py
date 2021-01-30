from copy import deepcopy
from dataclasses import dataclass
from traceback import extract_tb
from typing import *

from crosshair.core import realize
from crosshair.test_util import compare_results
from crosshair.test_util import ResultComparison
from crosshair.util import name_of_type
from crosshair.util import test_stack
from crosshair.util import IgnoreAttempt
from crosshair.util import UnexploredPath
from crosshair.util import debug


def test_bin(x: int) -> ResultComparison:
    ''' post: _ '''
    return compare_results(bin, x)

def test_format(x: object, f: str) -> ResultComparison:
    ''' post: _ '''
    return compare_results(format, x, f)

# CrossHair proxies don't have the same attributes as native:
#def test_getattr(o: object, n: str, d: object) -> ResultComparison:

# CrossHair proxies don't have the same attributes as native:
#def test_hasattr(o: str, n: str) -> ResultComparison:

def test_hash(o: object) -> ResultComparison:
    ''' post: _ '''
    return compare_results(hash, o)

def test_isinstance(o: object, t: type) -> ResultComparison:
    ''' post: _ '''
    return compare_results(isinstance, o, t)

def test_issubclass(o: object, t: type) -> ResultComparison:
    ''' post: _ '''
    return compare_results(issubclass, o, t)

def test_len(s: Sized) -> ResultComparison:
    ''' post: _ '''
    return compare_results(len, s)

def test_max(x: Sequence, k: Optional[Callable[[Any], Any]], d: object) -> ResultComparison:
    ''' post: _ '''
    return compare_results(max, x, k, d)

def test_min(x: Sequence) -> ResultComparison:
    ''' post: _ '''
    return compare_results(min, x)

def test_ord(x: str) -> ResultComparison:
    ''' post: _ '''
    return compare_results(ord, x)

def test_pow(b: Union[int, float], e: Union[int, float], m: Optional[int]) -> ResultComparison:
    ''' post: _ '''
    return compare_results(pow, b, e, m)

def test_repr(o: object) -> ResultComparison:
    ''' post: _ '''
    return compare_results(repr, o)

# CrossHair proxies don't have the same attributes as native:
#def test_setattr(o: object, n: str, v: object) -> ResultComparison:

def test_sorted(s: Sequence) -> ResultComparison:
    ''' post: _ '''
    return compare_results(sorted, s)

