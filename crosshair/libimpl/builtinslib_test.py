import collections.abc
import copy
import dataclasses
import enum
import math
import operator
import sys
import unittest
from typing import *

from crosshair.libimpl.builtinslib import (
    SymbolicArrayBasedUniformTuple,
    SymbolicByteArray,
    SymbolicBytes,
    SymbolicList,
    SymbolicType,
)
from crosshair.libimpl.builtinslib import SymbolicBool
from crosshair.libimpl.builtinslib import SymbolicFloat
from crosshair.libimpl.builtinslib import SymbolicInt
from crosshair.libimpl.builtinslib import SymbolicObject
from crosshair.libimpl.builtinslib import LazyIntSymbolicStr
from crosshair.libimpl.builtinslib import crosshair_types_for_python_type
from crosshair.core import CrossHairValue
from crosshair.core import analyze_function
from crosshair.core import proxy_for_type
from crosshair.core import deep_realize
from crosshair.core import realize
from crosshair.core import standalone_statespace
from crosshair.core_and_libs import run_checkables
from crosshair.options import AnalysisOptionSet
from crosshair.statespace import MessageType
from crosshair.test_util import check_ok
from crosshair.test_util import check_exec_err
from crosshair.test_util import check_fail
from crosshair.test_util import check_states
from crosshair.test_util import check_unknown
from crosshair.test_util import summarize_execution
from crosshair.tracers import NoTracing
from crosshair.tracers import ResumedTracing
from crosshair.util import IgnoreAttempt, set_debug

import pytest
import z3  # type: ignore


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


@dataclasses.dataclass
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


INF = float("inf")
NAN = float("nan")


class UnitTests(unittest.TestCase):
    def test_crosshair_types_for_python_type(self) -> None:
        self.assertEqual(crosshair_types_for_python_type(int), (SymbolicInt,))
        self.assertEqual(crosshair_types_for_python_type(SmokeDetector), ())

    def test_isinstance(self):
        with standalone_statespace:
            with NoTracing():
                f = SymbolicFloat("f")
            self.assertTrue(isinstance(f, float))
            self.assertFalse(isinstance(f, int))

    def test_smtfloat_like_a_float(self):
        with standalone_statespace, NoTracing():
            self.assertEqual(type(SymbolicFloat(12)), float)
            self.assertEqual(SymbolicFloat(12), 12.0)


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
            return ((n % d), (realize(n) % realize(d)))

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
            """
            pre: len(byt) == 2
            post: _ != 5
            """
            return int.from_bytes(byt, byteorder="little")

        self.assertEqual(*check_fail(f))

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


@pytest.mark.parametrize("b", (False, 1, -2.0, NAN, INF, -INF))
@pytest.mark.parametrize("op", (operator.lt, operator.eq, operator.add, operator.mul))
def test_bool_ops(b, op):
    with standalone_statespace as space:
        with NoTracing():
            a = SymbolicBool("a")
            space.add(a.var)
        symbolic_ret = summarize_execution(lambda: op(a, b))
        concrete_ret = summarize_execution(lambda: op(realize(a), b), detach_path=False)
        assert symbolic_ret == concrete_ret


@pytest.mark.parametrize("b", (False, 1, -2.0, NAN, INF, -INF))
@pytest.mark.parametrize("op", (operator.lt, operator.eq, operator.add, operator.mul))
def test_float_ops(b, op):
    with standalone_statespace as space:
        with NoTracing():
            a = SymbolicFloat("a")
            space.add(a.var < 0)
        symbolic_ret = summarize_execution(lambda: op(a, b))
        concrete_ret = summarize_execution(lambda: op(realize(a), b), detach_path=False)
        assert symbolic_ret == concrete_ret


def test_int_from_str():
    def f(a: str) -> int:
        """
        post: _ != 7
        raises: ValueError
        """
        return int(a)

    assert check_states(f) == {MessageType.POST_FAIL}


def test_easy_float_from_str():
    def f(a: str) -> float:
        """
        post: _ != 0.0
        raises: ValueError
        """
        return float(a)

    assert check_states(
        f, AnalysisOptionSet(max_iterations=100, per_condition_timeout=10)
    ) == {MessageType.POST_FAIL}


def test_float_from_three_digit_str():
    with standalone_statespace as space:
        with NoTracing():
            codepoints = [
                proxy_for_type(int, "xat0"),
                proxy_for_type(int, "xat1"),
                proxy_for_type(int, "xat2"),
            ]
            for point in codepoints:
                space.add(point.var >= ord("0"))
                space.add(point.var <= ord("9"))
            x = LazyIntSymbolicStr(codepoints)
        asfloat = float(x)
        assert space.is_possible(asfloat.var <= 999)
        assert not space.is_possible(asfloat.var > 999)
        assert space.is_possible(asfloat.var == 0)  # (because "000" is a valid float)
        assert not space.is_possible(asfloat.var == 500.5)


@pytest.mark.parametrize("val", [-256, 2 ** 16] + list(range(-4, 9, 2)))
def test_int_bit_length(val):
    with standalone_statespace as space:
        x = proxy_for_type(int, "x")
        space.add(x.var == val)
        assert realize(x.bit_length()) == val.bit_length()


@pytest.mark.parametrize(
    "val", [-256, -(2 ** 15), 2 ** 9, 2 ** 15 - 1] + list(range(-4, 9, 3))
)
def test_int_to_bytes(val):
    with standalone_statespace as space:
        x = proxy_for_type(int, "x")
        space.add(x.var == val)
        assert realize(x.to_bytes(2, "big", signed=True)) == val.to_bytes(
            2, "big", signed=True
        )


def test_int_format():
    with standalone_statespace as space:
        with NoTracing():
            x = SymbolicInt("x")
            space.add(x.var == 42)
        assert x.__format__("") == "42"
        # TODO this fails:
        # assert x.__format__("f") == "42.000000"


class StringsTest(unittest.TestCase):
    def test_cast_to_bool_fail(self) -> None:
        def f(a: str) -> str:
            """ post: a """
            return a

        self.assertEqual(*check_fail(f))

    def test_multiply_fail(self) -> None:
        def f(a: str) -> str:
            """
            pre: len(a) == 2
            post: len(_) != 6
            """
            return 3 * a

        self.assertEqual(*check_fail(f))

    def test_multiply_ok(self) -> None:
        def f(a: str) -> str:
            """
            pre: len(a) == 2
            post: len(_) == 10
            """
            return a * 3 + 2 * a

        self.assertEqual(*check_ok(f))

    def test_str_multiply_by_symbolic_fail(self) -> None:
        def f(i: int) -> str:
            """ post: len(_) != 6 """
            return "a\x00b" * i

        self.assertEqual(*check_fail(f))

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

    def test_find_with_negative_limits_fail(self) -> None:
        def f(a: str) -> int:
            """ post: _ == -1 """
            return a.find("ab", -2, 3)

        self.assertEqual(*check_fail(f))

    def test_ljust_fail(self) -> None:
        def f(s: str) -> str:
            """ post: len(_) == len(s) """
            return s.ljust(3, "x")

        self.assertEqual(*check_fail(f))

    def test_rfind_with_limits_ok(self) -> None:
        def f(a: str) -> int:
            """ post: _ == -1 """
            return a.rfind("abc", 1, 3)

        self.assertEqual(*check_ok(f))

    def test_rfind_with_negative_limits_fail(self) -> None:
        def f(a: str) -> int:
            """ post: _ == -1 """
            return a.rfind("ab", -2, 3)

        self.assertEqual(*check_fail(f))

    def test_rindex_fail(self) -> None:
        def f(a: str) -> int:
            """ post: _ != 2 """
            try:
                return a.rindex("abc")
            except ValueError:
                return 0

        self.assertEqual(*check_fail(f))

    def test_rindex_err(self) -> None:
        def f(a: str) -> int:
            """ post: True """
            return a.rindex("abc", 1, 3)

        self.assertEqual(*check_exec_err(f))

    def test_rjust_fail(self) -> None:
        def f(s: str) -> str:
            """ post: len(_) == len(s) """
            return s.rjust(3, "x")

        self.assertEqual(*check_fail(f))

    def test_replace_fail(self) -> None:
        def f(a: str) -> str:
            """ post: _ == a """
            return a.replace("b", "x", 1)

        self.assertEqual(*check_fail(f))

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

    def test_count_fail(self) -> None:
        def f(s: str) -> int:
            """ post: _ != 1 """
            return s.count(":")

        self.assertEqual(*check_fail(f))

    def test_split_fail(self) -> None:
        def f(s: str) -> list:
            """ post: _ != ['a', 'b'] """
            return s.split(",")

        self.assertEqual(*check_fail(f))

    def test_rsplit_fail(self) -> None:
        def f(s: str) -> list:
            """ post: __return__ != ['a', 'b'] """
            return s.rsplit(":", 1)

        self.assertEqual(*check_fail(f, AnalysisOptionSet(per_path_timeout=2)))

    def test_partition_ok(self) -> None:
        def f(s: str) -> tuple:
            """
            pre: len(s) == 3
            post: len(_) == 3
            """
            return s.partition(":")

        self.assertEqual(*check_ok(f))

    def test_partition_fail(self) -> None:
        def f(s: str) -> tuple:
            """
            pre: len(s) == 4
            post: _ != ("a", "bc", "d")
            """
            return s.partition("bc")

        self.assertEqual(*check_fail(f, AnalysisOptionSet(per_path_timeout=5)))

    def test_rpartition_ok(self) -> None:
        def f(s: str) -> tuple:
            """
            pre: len(s) == 2
            post: len(_) == 3
            """
            return s.rpartition(":")

        self.assertEqual(*check_ok(f))

    def test_rpartition_fail(self) -> None:
        def f(s: str) -> tuple:
            """
            pre: len(s) == 4
            post: _ != ("abb", "b", "")
            """
            return s.rpartition("b")

        self.assertEqual(*check_fail(f, AnalysisOptionSet(per_path_timeout=5)))

    def test_str_comparison_fail(self) -> None:
        def f(s1: str, s2: str) -> bool:
            """ post: _ """
            return s1 >= s2

        self.assertEqual(*check_fail(f))

    def test_compare_fail(self) -> None:
        def f(a: str, b: str) -> bool:
            """
            pre: a and b
            post: a[0] < b[0]
            """
            return a < b

        self.assertEqual(*check_fail(f))

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

    def test_string_formatting_wrong_key(self) -> None:
        def f(o: object) -> str:
            """ post: True """
            return "object of type {typ} with repr {zzzzz}".format(  # type: ignore
                typ=type(o), rep=repr(o)
            )

        self.assertEqual(*check_exec_err(f))

    def test_string_format_symbolic_format(self) -> None:
        def f(fmt: str) -> str:
            """
            pre: '{}' in fmt
            post: True
            """
            return fmt.format(ver=sys.version, platform=sys.platform)

        self.assertEqual(*check_exec_err(f))

    def test_string_format_fail(self) -> None:
        def f(inner: str) -> str:
            """
            post: _ != "abcdef"
            """
            return "ab{}ef".format(inner)

        self.assertEqual(*check_fail(f))

    def test_percent_format_unknown(self) -> None:
        def f(fmt: str) -> str:
            """
            pre: '%' not in fmt
            post: True
            """
            return fmt % ()

        self.assertEqual(*check_unknown(f))

    def test_join_fail(self) -> None:
        def f(items: List[str]) -> str:
            """
            pre: len(items) == 2
            post: len(_) != 3
            """
            return "and".join(items)

        self.assertEqual(*check_fail(f))

    def test_upper_fail(self) -> None:
        def f(s: str) -> str:
            """
            pre: len(s) == 1
            pre: s != "F"
            post: __return__ != "F"
            """
            return s.upper()

        # TODO: make this use case more efficient.
        options = AnalysisOptionSet(per_condition_timeout=30.0, per_path_timeout=15.0)
        self.assertEqual(*check_fail(f, options))

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

    def test_zfill_fail(self) -> None:
        def f(s: str) -> str:
            """ post: _ == s """
            return s.zfill(3)

        self.assertEqual(*check_fail(f))


def test_string_str() -> None:
    with standalone_statespace:
        with NoTracing():
            x = LazyIntSymbolicStr("x")
        strx = x.__str__()
        with NoTracing():
            assert isinstance(strx, str)


def test_string_center():
    with standalone_statespace as space:
        with NoTracing():
            string = LazyIntSymbolicStr("string")
            space.add(string.__len__().var == 3)
            fillch = LazyIntSymbolicStr("fillch")
            space.add(fillch.__len__().var == 1)
            sz = SymbolicInt("sz")
            space.add(sz.var > 5)
            sz6 = SymbolicInt("sz6")
            space.add(sz6.var == 6)
        assert "boo".center(sz6) == " boo  "
        symbolic_centered = "boo".center(sz, fillch)
        starts_with_nonfill = ord(symbolic_centered[0]) != ord(fillch)
        with NoTracing():
            assert not space.is_possible(starts_with_nonfill.var)


def TODO_test_string_map_chars() -> None:
    # TODO map circumvents our interception logic
    with standalone_statespace:
        with NoTracing():
            string = LazyIntSymbolicStr(list(map(ord, "ab")))
        codepoints = list(map(ord, string))  # TypeError because ord() isn't intercepted


def test_string_add() -> None:
    def f(s: str) -> str:
        """ post: _ != "Hello World" """
        return s + "World"

    actual, expected = check_fail(f)
    assert actual == expected


def test_string_bool():
    with standalone_statespace as space, NoTracing():
        a = LazyIntSymbolicStr("a")
        space.add(a.__len__().var > 0)
        with ResumedTracing():
            assert bool(a)
        # Can we retain our symbolic state after forcing a positive truthiness?:
        assert space.is_possible((a == "this").var)
        assert space.is_possible((a == "that").var)


def test_string_eq():
    with standalone_statespace, NoTracing():
        assert LazyIntSymbolicStr([]) == ""


def test_string_getitem():
    with standalone_statespace, NoTracing():
        assert LazyIntSymbolicStr(list(map(ord, "abc")))[0] == "a"
        assert LazyIntSymbolicStr(list(map(ord, "abc")))[-1] == "c"
        assert LazyIntSymbolicStr(list(map(ord, "abc")))[-5:2] == "ab"


def test_string_find() -> None:
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "aabc")))
        assert string.find("ab") == 1


def test_string_find_symbolic() -> None:
    def f(s: str) -> int:
        """
        pre: len(s) == 3
        post: _ == -1
        """
        return "haystack".find(s)

    actual, expected = check_fail(f)
    assert actual == expected


def test_string_find_notfound() -> None:
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr([])
        assert string.find("abc", 1, 3) == -1


def test_string_format_basic():
    with standalone_statespace as space:
        with NoTracing():
            s = LazyIntSymbolicStr("s")
            space.add(s.__len__().var == 1)
        assert space.is_possible((s == "z").var)
        assert space.is_possible((ord("a{0}c".format(s)[1]) == ord("b")).var)


def test_string_format_map():
    with standalone_statespace as space:
        with NoTracing():
            s = LazyIntSymbolicStr("s")
            space.add(s.__len__().var == 1)
        assert space.is_possible((s == "z").var)
        assert space.is_possible(
            (ord("a{foo}c".format_map({"foo": s})[1]) == ord("b")).var
        )


def test_string_rfind() -> None:
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "ababb")))
        assert string.rfind("ab") == 2


def test_string_rfind_notfound() -> None:
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "ababb")))
        assert string.rfind("ab") == 2


def test_string_split():
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "a:b:c")))
        parts = realize(string.split(":", 1))
        assert parts == ["a", "b:c"]
        parts = realize(string.split(":"))
        assert parts == ["a", "b", "c"]


def test_string_rsplit():
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "a:b:c")))
        parts = realize(string.rsplit(":", 1))
        assert parts == ["a:b", "c"]


def test_string_contains():
    with standalone_statespace:
        with NoTracing():
            small = LazyIntSymbolicStr([ord("b"), ord("c")])
            big = LazyIntSymbolicStr([ord("a"), ord("b"), ord("c"), ord("d")])
        assert small in big
        assert big not in small
        assert small in "bc"
        assert small not in "b"
        assert "c" in small
        assert "cd" not in small


def test_string_deep_realize():
    with standalone_statespace, NoTracing():
        a = LazyIntSymbolicStr("a")
        tupl = (a, (a,))
        realized = deep_realize(tupl)
    assert list(map(type, realized)) == [str, tuple]
    assert list(map(type, realized[1])) == [str]
    assert realized[0] is realized[1][0]


def test_string_strip():
    with standalone_statespace:
        with NoTracing():
            x = LazyIntSymbolicStr(list(map(ord, "  A b\n")))
        assert x.strip() == "A b"


def test_string_lower():
    chr_Idot = "\u0130"  # Capital I with dot above
    # (it's the only unicde char that lower()s to 2 characters)
    with standalone_statespace:
        with NoTracing():
            x = LazyIntSymbolicStr(list(map(ord, "Ab" + chr_Idot)))
        assert x.lower() == "abi\u0307"


def test_string_title():
    chr_lj = "\u01C9"  # "lj"
    chr_Lj = "\u01c8"  # "Lj" (different from "LJ", "\u01c7")
    with standalone_statespace:
        with NoTracing():
            lj = LazyIntSymbolicStr(list(map(ord, chr_lj)))
            lja_b = LazyIntSymbolicStr(list(map(ord, chr_lj + "a_b")))
        assert lja_b.title() == chr_Lj + "a_B"


def test_object_deep_realize():
    @dataclasses.dataclass
    class Container:
        contents: int

    with standalone_statespace as space, NoTracing():
        a = SymbolicObject("a", Container)
        shallow = realize(a)
        assert type(shallow) is Container
        assert type(shallow.contents) is not int
        deep = deep_realize(a)
        assert type(deep) is Container
        assert type(deep.contents) is int


def test_seq_string_deep_realize():
    with standalone_statespace as space:
        tupl = SymbolicArrayBasedUniformTuple("s", List[str])
        space.add(tupl._len() == 2)
        realized = deep_realize(tupl)
    assert list(map(type, realized)) == [str, str]


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
        def f(a: Iterable[int]) -> int:
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

        self.assertEqual(*check_ok(f))

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


def test_list_shallow_realization():
    with standalone_statespace as space:
        nums = proxy_for_type(List[int], "nums")
        numslen = len(nums)
        with NoTracing():
            space.add(numslen.var == 1)
            realized = realize(nums)
            assert type(realized) is list
            assert len(realized) == 1
            assert type(realized[0]) is SymbolicInt


def test_list_concrete_with_symbolic_slice(space):
    idx = proxy_for_type(int, "i")
    space.add(1 <= idx.var)
    space.add(idx.var <= 3)
    with ResumedTracing():
        prefix = [0, 1, 2, 3][:idx]
        prefixlen = len(prefix)
    assert isinstance(prefix, CrossHairValue)
    assert isinstance(prefixlen, CrossHairValue)
    assert space.is_possible(prefixlen.var == 1)
    assert space.is_possible(prefixlen.var == 3)


def test_list_copy(space):
    lst = proxy_for_type(List[int], "lst")
    with ResumedTracing():
        # Mostly just ensure the various ways of copying don't explode
        assert lst[:] is not lst
        assert lst.copy() is not lst
        assert copy.deepcopy(lst) is not lst
        assert copy.copy(lst) is not lst


def test_list_copy_compare_without_forking(space):
    lst = proxy_for_type(List[int], "lst")
    with ResumedTracing():
        lst2 = copy.deepcopy(lst)
    assert type(lst2) is SymbolicList
    assert lst.inner.var is lst2.inner.var
    with ResumedTracing():
        are_same = lst == lst2
    assert type(are_same) is SymbolicBool
    assert not space.is_possible(z3.Not(are_same.var))


class DictionariesTest(unittest.TestCase):
    def test_dict_basic_fail(self) -> None:
        def f(a: Dict[int, int], k: int, v: int) -> None:
            """
            post[a]: a[k] == 42
            """
            a[k] = v

        self.assertEqual(*check_fail(f))

    def test_dict_basic_ok(self) -> None:
        def f(a: Dict[int, int], k: int, v: int) -> None:
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

    def test_dict_deep_equality(self) -> None:
        def f(a: Dict[bool, set], b: List[Set[float]]) -> object:
            """
            pre: a == {True: set()}
            pre: b == [set(), {1.0}]
            post: _
            """
            if a == {True: set()}:
                if b == [set(), {1.0}]:
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

    def test_dict_over_heap_objects(self) -> None:
        def f(a: Dict[Tuple[int], int]) -> Optional[int]:
            """
            post: _ != 10
            """
            return a.get((5,))

        self.assertEqual(*check_fail(f))

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
        def f(d: Dict[Tuple[int, bool], Tuple[float, int]]) -> int:
            """
            post: _ > 0
            """
            if (42, True) in d:
                return d[(42, True)][1]
            else:
                return 42

        self.assertEqual(*check_fail(f, AnalysisOptionSet(per_condition_timeout=5)))

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


def test_dict_get():
    a = {"two": 2, "four": 4, "six": 6}

    def numstr(x: str) -> int:
        """
        post: _ != 4
        """
        return a.get(x, 9)

    assert check_states(numstr) == {MessageType.POST_FAIL}


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


def test_set_iter_partial():
    with standalone_statespace as space:
        with NoTracing():
            x = proxy_for_type(Set[int], "x")
            space.add(x.__len__().var == 2)
            print(type(x))
        itr = iter(x)
        first = next(itr)
        # leave the iterator incomplete; looking for generator + context mgr problems
    return


class FunctionsTest(unittest.TestCase):
    def test_hash(self) -> None:
        def f(s: int) -> int:
            """ post: True """
            return hash(s)

        self.assertEqual(*check_ok(f))

    def test_getattr(self) -> None:
        class Otter:
            def do_things(self) -> bool:
                return True

        def f(s: str) -> bool:
            """ post: _ != True """
            try:
                return getattr(Otter(), s)()
            except:
                return False

        messages = run_checkables(
            analyze_function(
                f,
                AnalysisOptionSet(
                    max_iterations=20, per_condition_timeout=5, per_path_timeout=1
                ),
            )
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0].message,
            "false when calling f(s = 'do_things') (which returns True)",
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
            # c: SupportsComplex,  # TODO: symbolic complex not yet really working
            b: SupportsBytes,
        ) -> float:
            """ post: _.real <= 0 """
            return abs(a) + float(f) + int(i) + round(r) + len(bytes(b))
            # + complex(c)

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
        def f(typ1: Type, typ2: Type[bool], typ3: Type):
            """ post: implies(_, issubclass(typ1, typ3)) """
            # The counterexample we expect: typ1==str typ2==bool typ3==int
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

        self.assertEqual(*check_unknown(f))

    def test_generic_object_equality(self) -> None:
        def f(thing: object, i: int):
            """ post: not _ """
            return thing == i

        self.assertEqual(*check_fail(f))


def test_issubclass_abc():
    with standalone_statespace as space:
        with NoTracing():
            dict_subtype = SymbolicType("dict_subtype", Type[dict])
        issub = issubclass(dict_subtype, collections.abc.Mapping)
        with NoTracing():
            # `issub` is lazily determined:
            assert type(issub) is SymbolicBool
            assert space.is_possible(issub.var)
            assert space.is_possible(z3.Not(issub.var))
            # We can artificially assert that this dict type is somehow not a Mapping:
            space.add(z3.Not(issub.var))
            # And CrossHair will give up when it comes time to find some such a type:
            with pytest.raises(IgnoreAttempt):
                realize(dict_subtype)


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


class BytesTest(unittest.TestCase):
    def test_specific_length(self) -> None:
        def f(b: bytes) -> int:
            """ post: _ != 5 """
            return len(b)

        self.assertEqual(*check_fail(f))

    def test_out_of_range_byte(self) -> None:
        def f(b: bytes) -> bytes:
            """
            pre: len(b) == 1
            post: _[0] != 256
            """
            return b

        self.assertEqual(*check_ok(f))


def test_bytes_roundtrip_array_as_symbolic():
    with standalone_statespace as space:
        orig_bytes = proxy_for_type(bytes, "origbytes")
        as_array = bytearray(orig_bytes)
        new_bytes = bytes(as_array)
        with NoTracing():
            assert type(as_array) is SymbolicByteArray
            assert type(new_bytes) is SymbolicBytes
            assert new_bytes.inner is orig_bytes.inner


def test_extend_concrete_bytearray():
    with standalone_statespace as space:
        b = bytearray(b"abc")
        xyz = proxy_for_type(bytearray, "xyz")
        b.extend(xyz)
        assert not space.is_possible(b[0] != ord("a"))
        assert space.is_possible(len(b).var > 3)


def test_bytearray_slice():
    with standalone_statespace as space:
        xyz = proxy_for_type(bytearray, "xyz")
        space.add(xyz.__len__().var == 3)
        assert type(xyz[1:]) is bytearray


def test_memoryview_compare():
    with standalone_statespace as space:
        mv1 = proxy_for_type(memoryview, "mv1")
        mv2 = proxy_for_type(memoryview, "mv2")
        len1, len2 = len(mv1), len(mv2)
        with NoTracing():
            space.add(len1.var == 0)
            space.add(len2.var == 0)
        views_equal = mv1 == mv2
        with NoTracing():
            assert views_equal is True


def test_memoryview_cast():
    """ post: _ """
    with standalone_statespace as space:
        val = proxy_for_type(int, "val")
        space.add(val.var == 254)
        mv = memoryview(bytearray([val]))
        assert mv.cast("b")[0] == -2


def test_memoryview_toreadonly():
    """ post: _ """
    with standalone_statespace as space:
        mv = proxy_for_type(memoryview, "mv")
        space.add(mv.__len__().var == 1)
        mv2 = mv.toreadonly()
        mv[0] = 12
        assert mv2[0] == 12
        with pytest.raises(TypeError):
            mv2[0] = 24


def test_memoryview_properties():
    """ post: _ """
    with standalone_statespace as space:
        symbolic_mv = proxy_for_type(memoryview, "symbolic_mv")
        space.add(symbolic_mv.__len__().var == 1)
        concrete_mv = memoryview(bytearray(b"a"))
        assert symbolic_mv.contiguous == concrete_mv.contiguous
        assert symbolic_mv.c_contiguous == concrete_mv.c_contiguous
        assert symbolic_mv.f_contiguous == concrete_mv.f_contiguous
        assert symbolic_mv.readonly == concrete_mv.readonly
        assert symbolic_mv.format == concrete_mv.format
        assert symbolic_mv.itemsize == concrete_mv.itemsize
        assert symbolic_mv.nbytes == concrete_mv.nbytes
        assert symbolic_mv.ndim == concrete_mv.ndim
        assert symbolic_mv.shape == concrete_mv.shape
        assert symbolic_mv.strides == concrete_mv.strides
        assert symbolic_mv.suboffsets == concrete_mv.suboffsets


def test_chr():
    with standalone_statespace as space:
        i = proxy_for_type(int, "i")
        space.add(z3.And(10 <= i.var, i.var < 256))
        c = chr(i)
        assert space.is_possible(c._codepoints[0].var == ord("a"))
        assert not space.is_possible(c._codepoints[0].var == 0)


def test_ord():
    with standalone_statespace as space:
        s = proxy_for_type(str, "s")
        space.add(len(s).var == 1)
        i = ord(s)
        assert space.is_possible(i.var == 42)
        assert not space.is_possible(i.var > sys.maxunicode)


def test_unicode_support():
    with standalone_statespace as space:
        s = proxy_for_type(str, "s")
        space.add(len(s).var == 1)
        assert space.is_possible((s == "a").var)
        assert space.is_possible((s == "\u1234").var)


@pytest.mark.parametrize("concrete_x", (25, 15, 6, -4, -15, -25))
def test_int_round(concrete_x):
    with standalone_statespace as space:
        concrete_ret = round(concrete_x, -1)
        x = proxy_for_type(int, "x")
        d = proxy_for_type(int, "d")
        space.add(x.var == concrete_x)
        space.add(d.var == -1)
        assert not space.is_possible((round(x, d) != concrete_ret).var)


def TODO_test_int_mod_float():
    with standalone_statespace as space:
        x = proxy_for_type(int, "x")
        y = proxy_for_type(float, "y")
        modval = x % y
        with NoTracing():
            assert type(modval) == SymbolicFloat
            assert space.is_possible(modval.var == 12.12)


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
