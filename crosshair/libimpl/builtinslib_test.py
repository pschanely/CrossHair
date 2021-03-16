import collections
import enum
import math
import sys
import unittest
from typing import *

from crosshair.libimpl.builtinslib import SmtFloat
from crosshair.libimpl.builtinslib import SmtInt
from crosshair.libimpl.builtinslib import SmtList
from crosshair.libimpl.builtinslib import crosshair_types_for_python_type
from crosshair.libimpl.builtinslib import _isinstance
from crosshair.libimpl.builtinslib import _max
from crosshair.core import analyze_function
from crosshair.core import realize
from crosshair.core_and_libs import *
from crosshair.options import AnalysisOptionSet
from crosshair.options import DEFAULT_OPTIONS
from crosshair.test_util import check_ok
from crosshair.test_util import check_exec_err
from crosshair.test_util import check_post_err
from crosshair.test_util import check_fail
from crosshair.test_util import check_unknown
from crosshair.util import set_debug
from crosshair.statespace import SimpleStateSpace
from crosshair.statespace import StateSpaceContext


class Cat:
    def size(self) -> int:
        return 1


class BiggerCat(Cat):
    def size(self) -> int:
        return 2


class Color(enum.Enum):
    RED = 0
    BLUE = 1
    GREEN = 2


class SmokeDetector:
    """ inv: not (self._is_plugged_in and self._in_original_packaging) """

    _in_original_packaging: bool
    _is_plugged_in: bool

    def signaling_alarm(self, air_samples: List[str]) -> bool:
        """
        pre: self._is_plugged_in
        post: implies('smoke' in air_samples, _ == True)
        """
        return "smoke" in air_samples


if sys.version_info >= (3, 8):

    class Movie(TypedDict):
        name: str
        year: int


class UnitTests(unittest.TestCase):
    def test_crosshair_types_for_python_type(self) -> None:
        self.assertEqual(crosshair_types_for_python_type(int), (SmtInt,))
        self.assertEqual(crosshair_types_for_python_type(SmokeDetector), ())

    def test_isinstance(self):
        with StateSpaceContext(SimpleStateSpace()):
            f = SmtFloat("f")
            self.assertFalse(isinstance(f, float))
            self.assertFalse(isinstance(f, int))
            self.assertTrue(_isinstance(f, float))
            self.assertFalse(_isinstance(f, int))

    def test_smtfloat_like_a_float(self):
        with StateSpaceContext(SimpleStateSpace()):
            self.assertEqual(type(SmtFloat(12)), float)
            self.assertEqual(SmtFloat(12), 12.0)


class BooleanTest(unittest.TestCase):
    def test_simple_bool_with_fail(self) -> None:
        def f(a: bool, b: bool) -> bool:
            """ post: _ == a """
            return True if a else b

        self.assertEqual(*check_fail(f))

    def test_simple_bool_ok(self) -> None:
        def f(a: bool, b: bool) -> bool:
            """ post: _ == a or b """
            return True if a else b

        self.assertEqual(*check_ok(f))

    def test_bool_ors_fail(self) -> None:
        def f(a: bool, b: bool, c: bool, d: bool) -> bool:
            """ post: _ == (a ^ b) or (c ^ d) """
            return a or b or c or d

        self.assertEqual(*check_fail(f))

    def test_bool_ors(self) -> None:
        def f(a: bool, b: bool, c: bool, d: bool) -> bool:
            """
            pre: (not a) and (not d)
            post: _ == (a ^ b) or (c ^ d)
            """
            return a or b or c or d

        self.assertEqual(*check_ok(f))

    def test_bool_as_numbers(self) -> None:
        def f(a: bool, b: bool) -> int:
            """ post: _ in (1, 2) """
            return (a * b) + True

        self.assertEqual(*check_ok(f))


class NumbersTest(unittest.TestCase):
    def test_floordiv(self) -> None:
        def f(n: int, d: int) -> Tuple[int, int]:
            """
            pre: n in (5, -5)
            pre: d in (5, 3, -3, -5)
            post: _[0] == _[1]
            """
            return ((n // d), (int(n) // int(d)))

        self.assertEqual(*check_ok(f))

    def test_mod(self) -> None:
        def f(n: int, d: int) -> Tuple[int, int]:
            """
            pre: n in (5, -5)
            pre: d in (5, 3, -3, -5)
            post: _[0] == _[1]
            """
            return ((n % d), (int(n) % int(d)))

        self.assertEqual(*check_ok(f))

    def test_simple_compare_ok(self) -> None:
        def f(i: List[int]) -> bool:
            """
            pre: 10 < len(i)
            post: _
            """
            return 9 < len(i[1:])

        self.assertEqual(*check_ok(f))

    def test_promotion_compare_ok(self) -> None:
        def f(i: int, f: float) -> bool:
            """
            pre: i == 7
            pre: f == 7.0
            post: _
            """
            return i == f and f >= i and i >= f

        self.assertEqual(*check_ok(f))

    def test_numeric_promotions(self) -> None:
        def f(b: bool, i: int) -> Tuple[int, float, float]:
            """
            #post: 100 <= _[0] <= 101
            #post: 3.14 <= _[1] <= 4.14
            post: isinstance(_[2], float)
            """
            return ((b + 100), (b + 3.14), (i + 3.14))

        self.assertEqual(*check_ok(f))

    def test_numbers_as_bool(self) -> None:
        def f(x: float, y: float):
            """
            pre: math.isfinite(x) and math.isfinite(y)
            post: _ == x or _ == y
            """
            return x or y

        self.assertEqual(*check_ok(f))

    def test_int_reverse_operators(self) -> None:
        def f(i: int) -> float:
            """
            pre: i != 0
            post: _ > 0
            """
            return (1 + i) + (1 - i) + (1 / i)

        self.assertEqual(*check_ok(f))

    def test_int_minus_symbolic_fail(self) -> None:
        def f(i: int) -> float:
            """
            post: _ != 42
            """
            return 1 - i

        self.assertEqual(*check_fail(f))

    def test_int_div_fail(self) -> None:
        def f(a: int, b: int) -> int:
            """ post: a <= _ <= b """
            return (a + b) // 2

        self.assertEqual(*check_fail(f))

    def test_int_div_ok(self) -> None:
        def f(a: int, b: int) -> int:
            """
            pre: a < b
            post: a <= _ <= b
            """
            return (a + b) // 2

        self.assertEqual(*check_ok(f))

    def test_int_bitwise_fail(self) -> None:
        def f(a: int, b: int) -> int:
            """
            pre: 0 <= a <= 3
            pre: 0 <= b <= 3
            post: _ < 7
            """
            return (a << 1) ^ b

        self.assertEqual(*check_fail(f))

    def test_int_bitwise_ok(self) -> None:
        def f(a: int, b: int) -> int:
            """
            pre: 0 <= a <= 3
            pre: 0 <= b <= 3
            post: _ <= 7
            """
            return (a << 1) ^ b

        self.assertEqual(*check_ok(f))

    def test_true_div_fail(self) -> None:
        def f(a: int, b: int) -> float:
            """
            pre: a != 0 and b != 0
            post: _ >= 1.0
            """
            return (a + b) / b

        self.assertEqual(*check_fail(f))

    def test_true_div_ok(self) -> None:
        def f(a: int, b: int) -> float:
            """
            pre: a >= 0 and b > 0
            post: _ >= 1.0
            """
            return (a + b) / b

        self.assertEqual(*check_ok(f))

    def test_trunc_fail(self) -> None:
        def f(n: float) -> int:
            """
            pre: n > 100
            post: _ < n
            """
            return math.trunc(n)

        self.assertEqual(*check_fail(f))

    def test_trunc_ok(self) -> None:
        def f(n: float) -> int:
            """ post: abs(_) <= abs(n) """
            return math.trunc(n)

        self.assertEqual(*check_ok(f))

    def test_round_fail(self) -> None:
        def f(n1: int, n2: int) -> Tuple[int, int]:
            """
            pre: n1 < n2
            post: _[0] < _[1] # because we round towards even
            """
            return (round(n1 + 0.5), round(n2 + 0.5))

        self.assertEqual(*check_fail(f))

    def test_round_unknown(self) -> None:
        def f(num: float, ndigits: Optional[int]) -> float:
            """
            post: isinstance(_, int) == (ndigits is None)
            """
            return round(num, ndigits)

        # TODO: this is unknown (rounding reals is hard)
        self.assertEqual(*check_unknown(f))

    def test_number_isinstance(self) -> None:
        def f(x: float) -> float:
            """ post: isinstance(_, float) """
            return x

        self.assertEqual(*check_ok(f))

    def test_mismatched_types(self) -> None:
        def f(x: float, y: list) -> float:
            """
            pre: x == 1.0 and y == []
            post: _ == 1
            """
            return x + y  # type: ignore

        self.assertEqual(*check_exec_err(f, "TypeError: unsupported operand type"))

    def test_surprisingly_valid_types(self) -> None:
        def f(x: bool) -> float:
            """
            pre: x == True
            post: _ == -2
            """
            return ~x

        self.assertEqual(*check_ok(f))

    def test_float_from_hex(self) -> None:
        def f(s: str) -> float:
            """
            pre: s == '0x3.a7p10'
            post: _ == 3740.0
            """
            return float.fromhex(s)

        self.assertEqual(*check_ok(f))

    def test_int_from_bytes(self) -> None:
        def f(byt: bytes) -> int:
            r"""
            pre: byt == b'\x00\x05'
            post: _ == 5
            """
            return int.from_bytes(byt, byteorder="big")

        self.assertEqual(*check_ok(f))

    def TODO_test_int_repr(self) -> None:
        def f(x: int) -> str:
            """ post: len(_) != 3 """
            return repr(x)

        self.assertEqual(*check_fail(f))

    def test_nonlinear(self) -> None:
        def make_bigger(x: int, e: int) -> float:
            """
            pre: e > 1
            post: __return__ !=  592704
            """
            # Expenentation is not SMT-solvable. (z3 gives unsat for this)
            # But CrossHair gracefully falls back to realized values, yielding
            # the counterexample of: 84 ** 3
            return x ** e

        self.assertEqual(*check_fail(make_bigger))


class StringsTest(unittest.TestCase):
    def test_cast_to_bool_fail(self) -> None:
        def f(a: str) -> str:
            """ post: a """
            return a

        self.assertEqual(*check_fail(f))

    def test_multiply_fail(self) -> None:
        def f(a: str) -> str:
            """ post: len(_) == len(a) * 3 """
            return 3 * a

        self.assertEqual(*check_ok(f))

    def test_multiply_ok(self) -> None:
        def f(a: str) -> str:
            """ post: len(_) == len(a) * 5 """
            return a * 3 + 2 * a

        self.assertEqual(*check_ok(f))

    def test_multiply_by_symbolic_ok(self) -> None:
        def f(i: int) -> str:
            """
            pre: i > 0
            post: len(_) == 3 * i
            post: _[2] == 'b'
            """
            return "a\x00b" * i

        self.assertEqual(*check_ok(f))

    def test_full_symbolic_multiply_unknown(self) -> None:
        def f(s: str, i: int) -> str:
            """
            pre: s and i > 0
            post: _[0] == s[0]
            """
            return s * i

        self.assertEqual(*check_unknown(f))

    def test_prefixing_fail(self) -> None:
        def f(a: str, indent: bool) -> str:
            """ post: len(_) == len(a) + indent """
            return ("  " if indent else "") + a

        self.assertEqual(*check_fail(f))

    def test_prefixing_ok(self) -> None:
        def f(a: str, indent: bool) -> str:
            """ post: len(_) == len(a) + (2 if indent else 0) """
            return ("  " if indent else "") + a

        self.assertEqual(*check_ok(f))

    def test_find_with_limits_ok(self) -> None:
        def f(a: str) -> int:
            """ post: _ == -1 """
            return a.find("abc", 1, 3)

        self.assertEqual(*check_ok(f))

    def test_find_with_negative_limits_ok(self) -> None:
        def f(a: str) -> int:
            """ post: _ == -1 """
            return a.find("abc", -2, 3)

        self.assertEqual(*check_ok(f))

    def test_rfind_with_limits_ok(self) -> None:
        def f(a: str) -> int:
            """ post: _ == -1 """
            return a.rfind("abc", 1, 3)

        self.assertEqual(*check_ok(f))

    def test_rfind_with_negative_limits_ok(self) -> None:
        def f(a: str) -> int:
            """ post: _ == -1 """
            return a.rfind("abc", -2, 3)

        self.assertEqual(*check_ok(f))

    def test_index_err(self) -> None:
        def f(s1: str, s2: str) -> int:
            """
            pre: s1 == 'aba'
            pre: 'ab' in s2
            post: True
            """
            return s1.index(s2)

        # index() raises ValueError when a match isn't found:
        self.assertEqual(*check_exec_err(f, "ValueError"))

    def test_negative_index_slicing(self) -> None:
        def f(s: str) -> Tuple[str, str]:
            """ post: sum(map(len, _)) == len(s) - 1 """
            idx = s.find(":")
            return (s[:idx], s[idx + 1 :])

        self.assertEqual(*check_fail(f))  # (fails when idx == -1)

    def test_starts_and_ends_ok(self) -> None:
        def f(s: str) -> str:
            """
            pre: s == 'aba'
            post: s.startswith('ab')
            post: s.endswith('ba')
            """
            return s

        self.assertEqual(*check_ok(f))

    def test_split_ok(self) -> None:
        def f(s: str) -> list:
            """ post: len(_) in (1, 2) """
            return s.split(":", 1)

        self.assertEqual(*check_ok(f))

    def test_split_fail(self) -> None:
        def f(s: str) -> list:
            """ post: _ != ['ab', 'cd'] """
            return s.split(",")

        self.assertEqual(*check_fail(f))

    def test_rsplit_ok(self) -> None:
        def f(s: str) -> list:
            """ post: len(_) in (1, 2) """
            return s.rsplit(":", 1)

        self.assertEqual(*check_ok(f))

    def test_rsplit_fail(self) -> None:
        def f(s: str) -> list:
            """ post: __return__ != ['a', 'b'] """
            return s.rsplit(":", 1)

        self.assertEqual(*check_fail(f))

    def test_str_comparison_fail(self) -> None:
        def f(s1: str, s2: str) -> bool:
            """ post: _ """
            return s1 >= s2

        self.assertEqual(*check_fail(f))

    def test_compare_ok(self) -> None:
        def f(a: str, b: str) -> bool:
            """
            pre: a and b
            post: implies(__return__, a[0] <= b[0])
            """
            return a < b

        self.assertEqual(*check_ok(f, AnalysisOptionSet(per_path_timeout=5)))

    def test_realized_compare(self) -> None:
        def f(a: str, b: str) -> bool:
            """
            post: implies(_, a == b)
            """
            return realize(a) == b

        self.assertEqual(*check_unknown(f))

    def test_int_str_comparison_fail(self) -> None:
        def f(a: int, b: str) -> Tuple[bool, bool]:
            """ post: (not _[0]) or (not _[1]) """
            return (a != b, b != a)

        self.assertEqual(*check_fail(f))

    def test_int_str_comparison_ok(self) -> None:
        def f(a: int, b: str) -> bool:
            """ post: _ == False """
            return a == b or b == a

        self.assertEqual(*check_ok(f))

    def test_string_formatting_literal(self) -> None:
        def f(o: object) -> str:
            """ post: True """
            return "object of type {typ} with repr {zzzzz}".format(  # type: ignore
                typ=type(o), rep=repr(o)
            )

        self.assertEqual(*check_exec_err(f))

    def test_string_formatting_varfmt(self) -> None:
        def f(fmt: str) -> str:
            """
            # NOTE: with a iteration-base, pure python implementation of format, we wouldn't need this precondition:
            pre: '{}' in fmt
            post: True
            """
            return fmt.format(ver=sys.version, platform=sys.platform)

        self.assertEqual(*check_exec_err(f))

    def test_percent_format(self) -> None:
        def f(fmt: str) -> str:
            """
            pre: '%' not in fmt
            post: True
            """
            return fmt % ()

        self.assertEqual(*check_unknown(f))

    def test_join_ok(self) -> None:
        def f(items: List[str]) -> str:
            """
            pre: 2 <= len(items) <= 3
            post: len(_) >= 3
            post: 'and' in _
            """
            return "and".join(items)

        self.assertEqual(*check_ok(f))

    def test_join_fail(self) -> None:
        def f(items: List[str]) -> str:
            """
            pre: len(items) > 0
            post: len(_) > 0
            """
            return ", ".join(items)

        self.assertEqual(*check_fail(f))

    def test_upper_unknown(self) -> None:
        def f(s: str) -> str:
            """ post: __return__ != "FOOBAR" """
            return s.upper()

        self.assertEqual(
            *check_unknown(f)
        )  # Ideally we'd find the counterexample input, "foobar"

    def test_csv_example(self) -> None:
        def f(lines: List[str]) -> List[str]:
            """
            pre: all(',' in line for line in lines)
            post: __return__ == [line.split(',')[0] for line in lines]
            """
            return [line[: line.index(",")] for line in lines]

        # TODO: the model generation doesn't work right here (getting a lot of empty strings):
        options = AnalysisOptionSet(per_path_timeout=0.5, per_condition_timeout=5)
        self.assertEqual(*check_unknown(f, options))


class TuplesTest(unittest.TestCase):
    def test_tuple_range_intersection_fail(self) -> None:
        def f(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            """
            pre: a[0] < a[1] and b[0] < b[1]
            post: _[0] <= _[1]
            """
            return (max(a[0], b[0]), min(a[1], b[1]))

        self.assertEqual(*check_fail(f))

    def test_tuple_range_intersection_ok(self) -> None:
        def f(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            """
            pre: a[0] < a[1] and b[0] < b[1]
            post: _ is None or _[0] <= _[1]
            """
            if a[1] > b[0] and a[0] < b[1]:  # (if the ranges overlap)
                return (max(a[0], b[0]), min(a[1], b[1]))
            else:
                return None

        self.assertEqual(*check_ok(f))

    def test_tuple_with_uniform_values_fail(self) -> None:
        def f(a: Tuple[int, ...]) -> float:
            """
            post: True
            """
            return sum(a) / len(a)

        self.assertEqual(*check_exec_err(f))

    def test_tuple_with_uniform_values_ok(self) -> None:
        def f(a: Tuple[int, ...]) -> Tuple[int, ...]:
            """
            pre: len(a) < 4
            post: 0 not in _
            """
            return tuple(x for x in a if x)

        self.assertEqual(*check_ok(f))

    def test_runtime_type(self) -> None:
        def f(t: Tuple) -> Tuple:
            """ post: t != (1, 2) """
            return t

        self.assertEqual(*check_fail(f))

    def test_isinstance_check(self) -> None:
        def f(uniform_tuple: Tuple[List, ...], basic_tuple: tuple) -> Tuple[bool, bool]:
            """ post: _ == (True, True)"""
            return (isinstance(uniform_tuple, tuple), isinstance(basic_tuple, tuple))

        self.assertEqual(*check_ok(f))


class ListsTest(unittest.TestCase):
    def test_range_can_be_called(self) -> None:
        def f(a: int) -> Iterable[int]:
            """ post: len(_) == a or a < 0 """
            return range(a)

        self.assertEqual(*check_unknown(f))

    def test_containment_fail(self) -> None:
        def f(a: int, b: List[int]) -> bool:
            """
            post: _ == (a in b[:3])
            """
            return a in b

        self.assertEqual(*check_fail(f))

    def test_containment_ok(self) -> None:
        def f(a: int, b: List[int]) -> bool:
            """
            pre: 1 == len(b)
            post: _ == (a == b[0])
            """
            return a in b

        self.assertEqual(*check_ok(f))

    def test_doubling_fail(self) -> None:
        def f(a: List[int]) -> List[int]:
            """
            post: len(_) > len(a)
            """
            return a + a

        self.assertEqual(*check_fail(f))

    def test_doubling_ok(self) -> None:
        def f(a: List[int]) -> List[int]:
            """
            post: len(_) > len(a) or not a
            """
            return a + a

        self.assertEqual(*check_ok(f))

    def test_multiply_ok(self) -> None:
        def f(a: List[int]) -> List[int]:
            """ post: len(_) == len(a) * 5 """
            return a * 3 + 2 * a

        self.assertEqual(*check_ok(f))

    def test_average(self) -> None:
        def average(numbers: List[float]) -> float:
            """
            pre: len(numbers) > 0
            post: min(numbers) <= _ <= max(numbers)
            """
            return sum(numbers) / len(numbers)

        self.assertEqual(*check_unknown(average))

    def test_mixed_symbolic_and_literal_concat_ok(self) -> None:
        def f(l: List[int], i: int) -> List[int]:
            """
            pre: i >= 0
            post: len(_) == len(l) + 1
            """
            return (
                l[:i]
                + [
                    42,
                ]
                + l[i:]
            )

        self.assertEqual(*check_ok(f))

    def test_range_fail(self) -> None:
        def f(l: List[int]) -> List[int]:
            """
            pre: len(l) == 3
            post: len(_) > len(l)
            """
            n: List[int] = []
            for i in range(len(l)):
                n.append(l[i] + 1)
            return n

        self.assertEqual(*check_fail(f))

    def test_range_ok(self) -> None:
        def f(l: List[int]) -> List[int]:
            """
            pre: l and len(l) < 10  # (max is to cap runtime)
            post: _[0] == l[0] + 1
            """
            n: List[int] = []
            for i in range(len(l)):
                n.append(l[i] + 1)
            return n

        self.assertEqual(*check_ok(f))

    def test_equality(self) -> None:
        def f(l: List[int]) -> List[int]:
            """
            pre: len(l) > 0
            post: _ != l
            """
            # extra check for positive equality:
            assert l == [x for x in l], "list does not equal itself"
            nl = l[:]
            nl[0] = 42
            return nl

        self.assertEqual(*check_fail(f))

    def test_extend_literal_unknown(self) -> None:
        def f(l: List[int]) -> List[int]:
            """
            post: _[:2] == [1, 2]
            """
            r = [1, 2, 3]
            r.extend(l)
            return r

        self.assertEqual(*check_unknown(f))

    def test_index_error(self) -> None:
        def f(l: List[int], idx: int) -> int:
            """
            pre: idx >= 0 and len(l) > 2
            post: True
            """
            return l[idx]

        self.assertEqual(*check_exec_err(f, "IndexError"))

    def test_index_type_error(self) -> None:
        def f(l: List[int]) -> int:
            """ post: True """
            return l[0.0:]  # type: ignore

        self.assertEqual(*check_exec_err(f, "TypeError"))

    def test_index_ok(self) -> None:
        def f(l: List[int]) -> bool:
            """
            pre: len(l) <= 3
            post: _ == (7 in l)
            """
            try:
                return l[l.index(7)] == 7
                return True
            except ValueError:
                return False

        self.assertEqual(*check_ok(f))

    def test_nested_lists_fail(self) -> None:
        def f(l: List[List[int]]) -> int:
            """
            post: _ > 0
            """
            total = 0
            for i in l:
                total += len(i)
            return total

        self.assertEqual(*check_fail(f))

    def test_nested_lists_ok(self) -> None:
        def f(l: List[List[int]]) -> int:
            """
            pre: len(l) < 4
            post: _ >= 0
            """
            total = 0
            for i in l:
                total += len(i)
            return total

        self.assertEqual(*check_ok(f))

    def test_iterable(self) -> None:
        def f(a: Iterable[str]) -> str:
            """
            pre: a
            post: _ in a
            """
            return next(iter(a))

        self.assertEqual(*check_ok(f))

    def test_isinstance_check(self) -> None:
        def f(l: List) -> bool:
            """ post: _ """
            return isinstance(l, list)

        self.assertEqual(*check_ok(f))

    def test_slice_outside_range_ok(self) -> None:
        def f(l: List[int], i: int) -> List[int]:
            """
            pre: i >= len(l)
            post: _ == l
            """
            return l[:i]

        self.assertEqual(*check_unknown(f))

    def test_slice_amount(self) -> None:
        def f(l: List[int]) -> List[int]:
            """
            pre: len(l) >= 3
            post: len(_) == 1
            """
            return l[2:3]

        self.assertEqual(*check_ok(f))

    def test_slice_assignment_ok(self) -> None:
        def f(l: List[int]) -> None:
            """
            pre: len(l) >= 2
            post[l]:
                l[1] == 42
                l[2] == 43
                len(l) == 4
            """
            l[1:-1] = [42, 43]

        self.assertEqual(*check_ok(f))

    def test_slice_assignment_out_of_bounds(self) -> None:
        def f(l: List[int], i: int) -> None:
            """
            pre: i != -1
            post: l == __old__.l[:i] + __old__.l[i+1:]
            """
            l[i : i + 1] = []

        self.assertEqual(*check_unknown(f))

    def test_insert_ok(self) -> None:
        def f(l: List[int]) -> None:
            """
            pre: len(l) == 4
            post[l]:
                len(l) == 5
                l[2] == 42
            """
            l.insert(-2, 42)

        self.assertEqual(*check_ok(f))

    def test_insert_with_conversions(self) -> None:
        def f(l: List[Set[int]], a: bool, b: int) -> None:
            """
            # self.insert(a,b) with {'a': True, 'b': 10, 'self': [{0}]}
            post: True
            """
            l.insert(a, b)  # type: ignore

        self.assertEqual(*check_ok(f))

    def test_pop_ok(self) -> None:
        def f(l: List[int]) -> None:
            """
            pre: l == [4, 5]
            post: l == [4]
            """
            l.pop()

        self.assertEqual(*check_ok(f))

    def test_count_ok(self) -> None:
        def f(l: List[Dict[int, Dict[int, int]]]) -> int:
            """
            pre: l == [{1: {2: 3}}]
            post: _ == 1
            """
            return l.count({1: {2: 3}})

        self.assertEqual(*check_ok(f))

    def test_assignment_ok(self) -> None:
        def f(l: List[int]) -> None:
            """
            pre: len(l) >= 4
            post[l]: l[3] == 42
            """
            l[3] = 42

        self.assertEqual(*check_ok(f))

    def test_slice_delete_fail(self) -> None:
        def f(l: List[int]) -> None:
            """
            pre: len(l) >= 2
            post[l]: len(l) > 0
            """
            del l[-2:]

        self.assertEqual(*check_fail(f))

    def test_item_delete_ok(self) -> None:
        def f(l: List[int]) -> None:
            """
            pre: len(l) == 5
            post[l]: len(l) == 4
            """
            del l[2]

        self.assertEqual(*check_ok(f))

    def test_item_delete_type_error(self) -> None:
        def f(l: List[float]) -> None:
            """
            pre: len(l) == 0
            post: True
            """
            del l[1.0]  # type: ignore

        self.assertEqual(*check_exec_err(f, "TypeError"))

    def test_item_delete_oob(self) -> None:
        def f(l: List[float]) -> None:
            """ post: True """
            del l[1]

        self.assertEqual(*check_exec_err(f, "IndexError"))

    def test_sort_ok(self) -> None:
        def f(l: List[int]) -> None:
            """
            pre: len(l) == 3
            post[l]: l[0] == min(l)
            """
            l.sort()

        self.assertEqual(*check_ok(f))

    def test_reverse_ok(self) -> None:
        def f(l: List[int]) -> None:
            """
            pre: len(l) == 2
            post[l]: l[0] == 42
            """
            l.append(42)
            l.reverse()

        self.assertEqual(*check_ok(f))

    def test_comparison_type_error(self) -> None:
        def f(a: List[Set], b: str):
            """ post: True """
            return a <= b  # type: ignore

        self.assertEqual(*check_exec_err(f, "TypeError"))


class DictionariesTest(unittest.TestCase):
    def test_dict_basic_fail(self) -> None:
        def f(a: Dict[int, int], k: int, v: int) -> None:
            """
            post[a]: a[k] == 42
            """
            a[k] = v

        self.assertEqual(*check_fail(f))

    def test_dict_basic_ok(self) -> None:
        def f(a: Dict[int, str], k: int, v: str) -> None:
            """
            post[a]: a[k] == v
            """
            a[k] = v

        self.assertEqual(*check_ok(f))

    def test_dict_get_with_defaults_ok(self) -> None:
        def f(a: Dict[float, float]) -> float:
            """ post: (_ == 1.2) or (_ == a[42.42]) """
            return a.get(42.42, 1.2)

        self.assertEqual(*check_ok(f))

    def test_dict_empty_bool(self) -> None:
        def f(a: Dict[int, str]) -> bool:
            """
            post[a]: _ == True
            """
            a[0] = "zero"
            return bool(a)

        self.assertEqual(*check_ok(f))

    def TODO_test_dict_deep_equality(
        self,
    ) -> None:  # This is too challenging right now.
        def f(a: Dict[bool, set], b: Dict[str, List[Set[float]]]) -> object:
            """
            pre: a == {True: set()}
            pre: b == {'': [set(), {1.0}]}
            post: _
            """
            if a == {True: set()}:
                if b == {"": [set(), {1.0}]}:
                    return False
            return True

        self.assertEqual(*check_fail(f))

    def test_dict_over_objects(self) -> None:
        def f(a: Dict[object, object]) -> int:
            """
            post: _ >= 0
            """
            return len(a)

        self.assertEqual(*check_ok(f))

    def test_dict_iter_fail(self) -> None:
        def f(a: Dict[int, str]) -> List[int]:
            """
            post[a]: 5 in _
            """
            a[10] = "ten"
            return list(a.__iter__())

        self.assertEqual(*check_fail(f))

    def test_dict_iter_ok(self) -> None:
        def f(a: Dict[int, str]) -> List[int]:
            """
            pre: len(a) < 3
            post[a]: 10 in _
            """
            a[10] = "ten"
            return list(a.__iter__())

        self.assertEqual(*check_ok(f))

    def test_dict_to_string_ok(self) -> None:
        def f(a: Dict[int, str]) -> str:
            """
            pre: len(a) == 0
            post: _ == '{}'
            """
            return str(a)

        self.assertEqual(*check_ok(f))

    def test_dict_items_ok(self) -> None:
        def f(a: Dict[int, str]) -> Iterable[Tuple[int, str]]:
            """
            pre: len(a) < 5
            post[a]: (10,'ten') in _
            """
            a[10] = "ten"
            return a.items()

        self.assertEqual(*check_ok(f))

    def test_dict_del_fail(self) -> None:
        def f(a: Dict[str, int]) -> None:
            """
            post[a]: True
            """
            del a["42"]

        self.assertEqual(*check_exec_err(f))

    def test_setdefault_float_int_comparison(self) -> None:
        def f(a: Dict[int, int]):
            """
            pre: a == {2: 0}
            post: _ == 0
            """
            return a.setdefault(2.0, {True: "0"})  # type: ignore

        self.assertEqual(*check_ok(f))

    def test_dicts_complex_contents(self) -> None:
        def f(d: Dict[Tuple[int, str], Tuple[float, int]]) -> int:
            """
            post: _ > 0
            """
            if (42, "fourty-two") in d:
                return d[(42, "fourty-two")][1]
            else:
                return 42

        self.assertEqual(*check_fail(f))

    def test_runtime_type(self) -> None:
        def f(t: dict) -> dict:
            """ post: t != {1: 2} """
            return t

        self.assertEqual(*check_fail(f))

    def test_isinstance_check(self) -> None:
        def f(smtdict: Dict[int, int], heapdict: Dict) -> Tuple[bool, bool]:
            """ post: _ == (True, True)"""
            return (isinstance(smtdict, dict), isinstance(heapdict, dict))

        self.assertEqual(*check_ok(f))

    def test_dicts_subtype_lookup(self) -> None:
        def f(d: Dict[Tuple[int, str], int]) -> None:
            """
            pre: not d
            post[d]: [(42, 'fourty-two')] == list(d.keys())
            """
            d[(42, "fourty-two")] = 1

        self.assertEqual(*check_ok(f))

    def test_dicts_complex_keys(self) -> None:
        def f(dx: Dict[Tuple[int, str], int]) -> None:
            """
            pre: not dx
            post[dx]:
                len(dx) == 1
                dx[(42, 'fourty-two')] == 1
            """
            dx[(42, "fourty-two")] = 1

        self.assertEqual(*check_ok(f))

    def test_symbolic_dict_has_unique_keys(self) -> None:
        def f(d: Dict[Tuple[int, str], int]) -> None:
            """
            pre: (1, 'one') in d
            post[d]: (1, 'one') not in d
            """
            del d[(1, "one")]

        self.assertEqual(*check_unknown(f))

    def test_equality_fail(self) -> None:
        def f(d: Dict[int, int]) -> Dict[int, int]:
            """ post: _ != d """
            d = d.copy()
            d[40] = 42
            return d

        self.assertEqual(*check_fail(f))

    def test_equality_ok(self) -> None:
        def f(d: Dict[int, int]) -> Dict[int, int]:
            """ post: _ == {**_} """
            return d

        self.assertEqual(*check_unknown(f))

    def test_wrong_key_type(self) -> None:
        def f(d: Dict[int, int], s: str, i: int) -> bool:
            if i == 0:
                del d[s]  # type: ignore
            elif i < 0:
                d[s] = 7  # type: ignore
            else:
                _val = d[s]  # type: ignore
            return True

        self.assertEqual(*check_ok(f))

    def test_dict_key_type_union(self) -> None:
        def f(d: Dict[Union[int, str], int]) -> Dict:
            """
            pre: len(d) == 2
            post: not (42 in d and '42' in d)
            """
            return d

        self.assertEqual(*check_fail(f))

    def test_nonuniform_dict_types(self) -> None:
        def f(a: Dict[Hashable, int]) -> Dict[Hashable, int]:
            """
            pre: len(a) == 1
            post: _[0] == 100
            """
            b: Dict[Hashable, int] = {0: 100}
            b.update(a)
            return b

        self.assertEqual(*check_fail(f))

    def test_dicts_inside_lists(self) -> None:
        def f(dicts: List[Dict[int, int]]) -> Dict[int, int]:
            """
            pre: len(dicts) <= 1  # to narrow search space (would love to make this larger)
            post: len(_) <= len(dicts)
            """
            ret = {}
            for d in dicts:
                ret.update(d)
            return ret

        self.assertEqual(*check_fail(f))

    def test_dicts_inside_lists_with_identity(self) -> None:
        # NOTE: the message is a little confusing because repr()
        # hides the fact that the identity of the lists is the same.
        def f(dicts: List[Dict[int, int]]):
            """
            Removes duplicate keys.
            pre: len(dicts) == 2
            pre:  len(dicts[0]) == 1
            post: len(dicts[0]) == 1
            """
            seen: Set[int] = set()
            for d in dicts:
                for k in d.keys():
                    if k in seen:
                        del d[k]
                    else:
                        seen.add(k)

        self.assertEqual(*check_fail(f))

    def test_consistent_ordering(self) -> None:
        def f(symbolic: Dict[int, int]) -> Tuple[List[int], List[int]]:
            """ post: _[0] == _[1] """
            return (list(symbolic.keys()), list(symbolic.keys()))

        self.assertEqual(*check_unknown(f))

    def test_ordering_after_mutations(self) -> None:
        def f(d: Dict[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
            """
            pre: len(d) == 3
            post[d]: _[0] == _[1]
            """
            o1, middle, o2 = d.keys()
            d[o1] = 42
            d[o2] = 42
            del d[middle]
            n1, n2 = d.keys()
            return ((o1, o2), (n1, n2))

        self.assertEqual(*check_ok(f))

    def test_alternate_mapping_types(self) -> None:
        def f(m1: Mapping[int, int], m2: MutableMapping[int, int]) -> int:
            """
            pre: 1 in m1 and 2 in m2
            post: _ != 10
            """
            return m1[1] + m2[2]

        self.assertEqual(*check_fail(f))

    def test_implicit_conversion_for_keys(self) -> None:
        def f(m: Dict[float, float], b: bool, i: int):
            """
            post: len(m) >= len(__old__.m)
            """
            m[b] = 2.0
            m[i] = 3.0

        self.assertEqual(*check_ok(f))

    if sys.version_info >= (3, 8):

        def test_typed_dict_fail(self) -> None:
            def f(td: Movie):
                ''' post: _['year'] != 2020 or _['name'] != "hi"'''
                return td

            self.assertEqual(*check_fail(f))


class SetsTest(unittest.TestCase):
    def test_basic_fail(self) -> None:
        def f(a: Set[int], k: int) -> None:
            """
            post[a]: k+1 in a
            """
            a.add(k)

        self.assertEqual(*check_fail(f))

    def test_basic_ok(self) -> None:
        def f(a: Set[int], k: int) -> None:
            """
            post[a]: k in a
            """
            a.add(k)

        self.assertEqual(*check_ok(f))

    def test_union_fail(self) -> None:
        def f(a: Set[str], b: Set[str]) -> Set[str]:
            """
            pre: len(a) == len(b) == 1  # (just for test performance)
            post: all(((i in a) and (i in b)) for i in _)
            """
            return a | b

        self.assertEqual(*check_fail(f))

    def test_union_ok(self) -> None:
        def f(a: Set[str], b: Set[str]) -> Set[str]:
            """
            post: all(((i in a) or (i in b)) for i in _)
            """
            return a | b

        self.assertEqual(*check_unknown(f))

    def test_contains_different_but_equivalent(self) -> None:
        def f(s: Set[Union[int, str]]) -> str:
            """
            pre: "foobar" in s
            post: (_ + "bar") in s
            """
            return "foo"

        self.assertEqual(*check_unknown(f))

    # The heaprefs + deferred set assumptions make this too expensive.
    # TODO: Optimize & re-enable
    def TODO_test_subtype_union(self) -> None:
        def f(s: Set[Union[int, str]]) -> Set[Union[int, str]]:
            """ post: not ((42 in s) and ('42' in s)) """
            return s

        self.assertEqual(*check_fail(f, AnalysisOptionSet(per_condition_timeout=7.0)))

    def test_subset_compare_ok(self) -> None:
        # a >= b with {'a': {0.0, 1.0}, 'b': {2.0}}
        def f(s1: Set[float], s2: Set[float]) -> bool:
            """
            pre: s1 == {0.0, 1.0}
            pre: s2 == {2.0}
            post: not _
            """
            return s1 >= s2

        self.assertEqual(*check_ok(f))

    def test_set_numeric_promotion(self) -> None:
        def f(i: int, s: Set[float]) -> bool:
            """
            pre: i == 2
            pre: s == {2.0}
            post: _
            """
            return i in s

        self.assertEqual(*check_ok(f))

    def test_set_runtime_type_ok(self) -> None:
        def f(s: set) -> bool:
            """ post: _ """
            return True

        self.assertEqual(*check_ok(f))

    def test_isinstance_check(self) -> None:
        def f(s: Set[object]) -> bool:
            """ post: _ """
            return isinstance(s, set)

        self.assertEqual(*check_ok(f))

    def test_sets_eq(self) -> None:
        def f(a: Set[FrozenSet[int]]) -> object:
            """
            pre: a == {frozenset({7}), frozenset({42})}
            post: _ in ('{frozenset({7}), frozenset({42})}', '{frozenset({42}), frozenset({7})}')
            """
            return repr(a)

        self.assertEqual(
            *check_ok(
                f, AnalysisOptionSet(per_path_timeout=10, per_condition_timeout=10)
            )
        )

    def test_containment(self) -> None:
        def f(s: Set[int]) -> int:
            """
            pre: len(s) == 2
            post: _
            """
            i = iter(s)
            x = next(i)
            y = next(i)
            return x != y

        self.assertEqual(*check_ok(f))


class FunctionsTest(unittest.TestCase):
    def test_hash(self) -> None:
        def f(s: str) -> int:
            """ post: True """
            return hash(s)

        self.assertEqual(*check_ok(f))

    def test_getattr(self) -> None:
        class Otter:
            def do_cute_human_things_with_hands(self) -> str:
                return "cuteness"

        def f(s: str) -> str:
            """ post: _ != "cuteness" """
            try:
                return getattr(Otter(), s)()
            except:
                return ""

        messages = run_checkables(
            analyze_function(
                f, AnalysisOptionSet(max_iterations=20, per_condition_timeout=5)
            )
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0].message,
            "false when calling f(s = 'do_cute_human_things_with_hands') (which returns 'cuteness')",
        )


class ProtocolsTest(unittest.TestCase):
    # TODO: move most of this into a collectionslib_test.py file
    def test_hashable_values_fail(self) -> None:
        def f(b: bool, i: int, t: Tuple[str, ...], s: FrozenSet[float]) -> int:
            """ post: _ % 10 != 0 """
            return hash((i, t, s))

        self.assertEqual(*check_fail(f))

    def test_hashable_values_ok(self) -> None:
        def f(a: Tuple[str, int, float, bool], b: Tuple[str, int, float, bool]) -> int:
            """ post: _ or not (a == b) """
            return hash(a) == hash(b)

        self.assertEqual(*check_unknown(f))

    def test_symbolic_hashable(self) -> None:
        def f(a: Hashable) -> int:
            """ post[]: 0 <= _ <= 1 """
            return hash(a) % 2

        self.assertEqual(*check_ok(f))

    def test_symbolic_supports(self) -> None:
        def f(
            a: SupportsAbs,
            f: SupportsFloat,
            i: SupportsInt,
            r: SupportsRound,
            c: SupportsComplex,
            b: SupportsBytes,
        ) -> float:
            """ post: _.real <= 0 """
            return abs(a) + float(f) + int(i) + round(r) + complex(c) + len(bytes(b))

        self.assertEqual(*check_fail(f))

    def test_iterable(self) -> None:
        T = TypeVar("T")

        def f(a: Iterable[T]) -> T:
            """
            pre: a
            post: _ in a
            """
            return next(iter(a))

        self.assertEqual(*check_unknown(f))

    def test_bare_type(self) -> None:
        def f(a: List) -> bool:
            """
            pre: a
            post: _
            """
            return bool(a)

        self.assertEqual(*check_ok(f))


class EnumsTest(unittest.TestCase):
    def test_enum_identity_matches_equality(self) -> None:
        def f(color1: Color, color2: Color) -> bool:
            """ post: _ == (color1 is color2) """
            return color1 == color2

        self.assertEqual(*check_ok(f))

    def test_enum_in_container(self) -> None:
        def f(colors: List[Color]) -> bool:
            """ post: not _ """
            return Color.RED in colors and Color.BLUE in colors

        self.assertEqual(*check_fail(f))


class TypesTest(unittest.TestCase):
    def test_symbolic_types_ok(self) -> None:
        def f(typ: Type[SmokeDetector]):
            """ post: _ """
            return issubclass(typ, SmokeDetector)

        self.assertEqual(*check_ok(f))

    def test_symbolic_type_can_be_subclass(self) -> None:
        def f(typ: Type[Cat]):
            """ post: _ == "<class '__main__.Cat'>" """
            return str(typ)

        # False when the type is instantiated as "BiggerCat":
        self.assertEqual(*check_fail(f))

    def test_symbolic_types_fail(self) -> None:
        def f(typ: Type):
            """ post: _ """
            return issubclass(typ, str)

        self.assertEqual(*check_fail(f))

    def test_symbolic_types_without_literal_types(self) -> None:
        def f(typ1: Type, typ2: Type, typ3: Type):
            """ post: implies(_, issubclass(typ1, typ3)) """
            return issubclass(typ2, typ3) and typ2 != typ3

        self.assertEqual(
            *check_fail(
                f, AnalysisOptionSet(max_iterations=60, per_condition_timeout=10)
            )
        )

    def test_instance_creation(self) -> None:
        def f(t: Type[Cat]):
            """ post: _.size() > 0 """
            return t()

        self.assertEqual(*check_ok(f))

    def test_type_comparison(self) -> None:
        def f(t: Type) -> bool:
            """ post: _ """
            return t == int

        self.assertEqual(*check_fail(f))

    def test_type_as_bool(self) -> None:
        def f(t: Type) -> bool:
            """ post: _ """
            return bool(t)

        self.assertEqual(*check_ok(f))

    def test_generic_object_and_type(self) -> None:
        def f(thing: object, detector_kind: Type[SmokeDetector]):
            """ post: True """
            if isinstance(thing, detector_kind):
                return thing._is_plugged_in
            return False

        self.assertEqual(*check_ok(f))

    def test_generic_object_equality(self) -> None:
        def f(thing: object, i: int):
            """ post: not _ """
            return thing == i

        self.assertEqual(*check_fail(f))


class CallableTest(unittest.TestCase):
    def test_symbolic_zero_arg_callable(self) -> None:
        def f(size: int, initializer: Callable[[], int]) -> Tuple[int, ...]:
            """
            pre: size >= 1
            post: _[0] != 707
            """
            return tuple(initializer() for _ in range(size))

        self.assertEqual(*check_fail(f))

    def test_symbolic_one_arg_callable(self) -> None:
        def f(size: int, mapfn: Callable[[int], int]) -> Tuple[int, ...]:
            """
            pre: size >= 1
            post: _[0] != 707
            """
            return tuple(mapfn(i) for i in range(size))

        self.assertEqual(*check_fail(f))

    def test_symbolic_two_arg_callable(self) -> None:
        def f(i: int, callable: Callable[[int, int], int]) -> int:
            """ post: _ != i """
            return callable(i, i)

        self.assertEqual(*check_fail(f))

    def test_callable_as_bool(self) -> None:
        def f(fn: Callable[[int], int]) -> bool:
            """ post: _ """
            return bool(fn)

        self.assertEqual(*check_ok(f))

    def test_callable_repr(self) -> None:
        def f(f1: Callable[[int], int]) -> int:
            """ post: _ != 1234 """
            return f1(4)

        messages = run_checkables(analyze_function(f))
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0].message,
            "false when calling f(f1 = lambda a: 1234) (which returns 1234)",
        )

    def test_callable_with_typevar_in_args(self) -> None:
        # For now, just don't explode. But we should be able to make these fail with
        # some work. See https://github.com/pschanely/CrossHair/issues/85
        T = TypeVar("T")

        def f(a: Callable[[T], int], x: T) -> int:
            """post: _ != 42"""
            return a(x)

        self.assertEqual(*check_unknown(f))

    def test_callable_with_typevar_in_return(self) -> None:
        # For now, just don't explode. But we should be able to make these fail with
        # some work. See https://github.com/pschanely/CrossHair/issues/85
        T = TypeVar("T")

        def f(a: Callable[[int], T], x: int) -> T:
            """post: _"""
            return a(x)

        self.assertEqual(*check_unknown(f))


class ContractedBuiltinsTest(unittest.TestCase):
    def TODO_test_print_ok(self) -> None:
        def f(x: int) -> bool:
            """
            post: _ == True
            """
            print(x)
            return True

        self.assertEqual(*check_ok(f))

    def test_repr_ok(self):
        def f(x: int) -> str:
            """ post: len(_) == 0 or len(_) > 0 """
            return repr(x)

        self.assertEqual(*check_ok(f))

    def test_max_fail(self) -> None:
        def f(l: List[int]) -> int:
            """
            post: _ in l
            """
            return max(l)

        self.assertEqual(*check_exec_err(f))

    def test_max_ok(self) -> None:
        def f(l: List[int]) -> int:
            """
            pre: bool(l)
            post[]: _ in l
            """
            return max(l)

        self.assertEqual(*check_unknown(f))

    def test_min_ok(self) -> None:
        def f(l: List[float]) -> float:
            """
            pre: bool(l)
            post[]: _ in l
            """
            return min(l)

        self.assertEqual(*check_unknown(f))

    def test_list_index(self) -> None:
        def f(i: int) -> int:
            """ post: True """
            return [0, 1, 2].index(i)

        self.assertEqual(*check_exec_err(f, "ValueError: 3 is not in list"))

    def test_eval_namespaces(self) -> None:
        def f(i: int) -> int:
            """ post: _ == i + 1 """
            return eval("i + Color.BLUE.value")

        self.assertEqual(*check_ok(f))


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
