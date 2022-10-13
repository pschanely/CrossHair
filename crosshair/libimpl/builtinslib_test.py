# -*- coding: utf-8 -*-

import collections.abc
import copy
import dataclasses
import enum
import math
import operator
import sys
import unittest
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    SupportsAbs,
    SupportsBytes,
    SupportsFloat,
    SupportsInt,
    SupportsRound,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pytest
import z3  # type: ignore

from crosshair.core import (
    CrossHairValue,
    analyze_function,
    deep_realize,
    proxy_for_type,
    realize,
    standalone_statespace,
)
from crosshair.core_and_libs import run_checkables
from crosshair.libimpl.builtinslib import (
    LazyIntSymbolicStr,
    SymbolicArrayBasedUniformTuple,
    SymbolicBool,
    SymbolicByteArray,
    SymbolicBytes,
    SymbolicFloat,
    SymbolicInt,
    SymbolicList,
    SymbolicObject,
    SymbolicType,
    crosshair_types_for_python_type,
)
from crosshair.options import AnalysisOptionSet
from crosshair.statespace import (
    CANNOT_CONFIRM,
    CONFIRMED,
    EXEC_ERR,
    POST_FAIL,
    MessageType,
)
from crosshair.test_util import check_exec_err, check_states, summarize_execution
from crosshair.tracers import NoTracing, ResumedTracing
from crosshair.util import IgnoreAttempt, set_debug


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
    """inv: not (self._is_plugged_in and self._in_original_packaging)"""

    _in_original_packaging: bool
    _is_plugged_in: bool

    def signaling_alarm(self, air_samples: List[str]) -> bool:
        """
        pre: self._is_plugged_in
        post: implies('smoke' in air_samples, _ == True)
        """
        return "smoke" in air_samples


if sys.version_info >= (3, 8):
    from typing import TypedDict

    class Movie(TypedDict):
        name: str
        year: int


INF = float("inf")
NAN = float("nan")


def test_crosshair_types_for_python_type() -> None:
    assert crosshair_types_for_python_type(int) == (SymbolicInt,)
    assert crosshair_types_for_python_type(SmokeDetector) == ()


def test_isinstance():
    with standalone_statespace:
        with NoTracing():
            f = SymbolicFloat("f")
        assert isinstance(f, float)
        assert not isinstance(f, int)


def test_smtfloat_like_a_float():
    with standalone_statespace:
        with NoTracing():
            f1 = SymbolicFloat("f")
        f2 = type(f1)(12)
        with NoTracing():
            assert isinstance(f2, float)
            assert f2 == 12.0


def test_bool_simple_conditional_fail() -> None:
    def f(a: bool, b: bool) -> bool:
        """post: _ == a"""
        return True if a else b

    check_states(f, POST_FAIL)


def test_bool_simple_conditional_ok() -> None:
    def f(a: bool, b: bool) -> bool:
        """post: _ == a or b"""
        return True if a else b

    check_states(f, CONFIRMED)


def test_bool_ors_fail() -> None:
    def f(a: bool, b: bool, c: bool, d: bool) -> bool:
        """post: _ == (a ^ b) or (c ^ d)"""
        return a or b or c or d

    check_states(f, POST_FAIL)


def test_bool_ors() -> None:
    def f(a: bool, b: bool, c: bool, d: bool) -> bool:
        """
        pre: (not a) and (not d)
        post: _ == (a ^ b) or (c ^ d)
        """
        return a or b or c or d

    check_states(f, CONFIRMED)


def test_bool_as_numbers() -> None:
    def f(a: bool, b: bool) -> int:
        """post: _ in (1, 2)"""
        return (a * b) + True

    check_states(f, CONFIRMED)


def test_int___floordiv___ok() -> None:
    def f(n: int, d: int) -> Tuple[int, int]:
        """
        pre: n in (5, -5)
        pre: d in (5, 3, -3, -5)
        post: _[0] == _[1]
        """
        return ((n // d), (int(n) // int(d)))

    check_states(f, CONFIRMED)


def test_number_simple_compare_ok() -> None:
    def f(i: List[int]) -> bool:
        """
        pre: 10 < len(i)
        post: _
        """
        return 9 < len(i[1:])

    check_states(f, CONFIRMED)


def test_number_promotion_compare_unknown() -> None:
    def f(i: int, f: float) -> bool:
        """
        pre: i == 7
        pre: f == 7.0
        post: _
        """
        return i == f and f >= i and i >= f

    check_states(f, CANNOT_CONFIRM)


def test_numeric_promotions() -> None:
    def f(b: bool, i: int) -> Tuple[int, float, float]:
        """
        post: _ != (101, 4.14, 13.14)
        """
        return ((b + 100), (b + 3.14), (i + 3.14))

    check_states(f, POST_FAIL)


def test_float_as_bool() -> None:
    def f(x: float, y: float):
        """
        pre: math.isfinite(x) and math.isfinite(y)
        post: _ == x or _ == y
        """
        return x or y

    check_states(f, CANNOT_CONFIRM)


def test_int_reverse_operators() -> None:
    def f(i: int) -> float:
        """
        pre: i != 0
        post: _ != 1
        """
        return (1 + i) + (1 - i) + (1 / i)

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_int___add___method():
    def f(a: int, b: int) -> int:
        """
        Can the sum of two consecutive integers be 37?

        pre: a + 1 == b
        post: _ != 37
        """
        return a + b

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_int___mod___method():
    def f(a: int) -> int:
        """
        Can the last digit of a given large number be 3?

        pre: a > 1234
        post: _ != 3
        """
        return a % 10

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_int___mul___method():
    def f(a: int, b: int) -> int:
        """
        Can we multiply two integers and return 42?

        NOTE: Although this example works, nonlinear integer arithmetic can not
        always be effectively reasoned about.

        pre: a > b > 1
        post: _ != 42
        """
        return a * b

    check_states(f, POST_FAIL)


@pytest.mark.demo("yellow")
def test_int___pow___method():
    def f(a: int) -> int:
        """
        Can the given integer, cubed, equal 343?

        NOTE: Although this example works, nonlinear integer arithmetic can not
        always be effectively reasoned about. This is particularly true when
        the exponent is symbolic.


        post: _ != 343
        """
        return a**3

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_int___sub___method():
    def f(a: int) -> int:
        """
        Can we subtract from 42 and get something larger?

        post: _ <= 42
        """
        return 42 - a

    check_states(f, POST_FAIL)


def test_int___rsub__() -> None:
    def f(i: int) -> float:
        """
        post: _ != 42
        """
        return 1 - i

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_int___floordiv___method() -> None:
    def f(a: int, b: int) -> int:
        """
        Can the average of two integers equal either one?

        pre: a < b
        post: a < _ < b
        """
        return (a + b) // 2

    check_states(f, POST_FAIL)


def test_int___floordiv___bounds() -> None:
    def f(a: int, b: int) -> int:
        """
        pre: a < b
        post: a <= _ < b
        """
        return (a + b) // 2

    check_states(f, CONFIRMED)


def test_int_bitwise_fail() -> None:
    def f(a: int, b: int) -> int:
        """
        pre: 0 <= a <= 3
        pre: 0 <= b <= 3
        post: _ < 7
        """
        return (a << 1) ^ b

    check_states(f, POST_FAIL)


def test_int_bitwise_ok() -> None:
    def f(a: int, b: int) -> int:
        """
        pre: 0 <= a <= 3
        pre: 0 <= b <= 3
        post: _ <= 7
        """
        return (a << 1) ^ b

    check_states(f, CONFIRMED)


@pytest.mark.demo
def test_int___truediv___method() -> None:
    def f(a: int, b: int) -> float:
        """
        Can we find an integer that is half as large as another?

        pre: b != 0
        post: _ != 0.5
        """
        return a / b

    check_states(f, POST_FAIL)


def test_trunc_fail() -> None:
    def f(n: float) -> int:
        """
        pre: n > 100
        post: _ < n
        """
        return math.trunc(n)

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_range() -> None:
    def f(n: int) -> Sequence[int]:
        """
        Can the length of range() be different than the integer we pass to it?

        post: len(_) == n
        """
        return range(n)

    check_states(f, POST_FAIL)


def test_round_fail() -> None:
    def f(n1: int, n2: int) -> Tuple[int, int]:
        """
        pre: n1 < n2
        post: _[0] < _[1] # because we round towards even
        """
        return (round(n1 + 0.5), round(n2 + 0.5))

    check_states(f, POST_FAIL)


def test_round_unknown() -> None:
    def f(num: float, ndigits: Optional[int]) -> float:
        """
        post: isinstance(_, int) == (ndigits is None)
        """
        return round(num, ndigits)

    # TODO: this is unknown (rounding reals is hard)
    check_states(f, CANNOT_CONFIRM)


def test_float_isinstance() -> None:
    def f(x: float) -> float:
        """post: isinstance(_, float)"""
        return x

    check_states(f, CANNOT_CONFIRM)


def test_mismatched_types() -> None:
    def f(x: float, y: list) -> float:
        """
        pre: x == 1.0 and y == []
        post: _ == 1
        """
        return x + y  # type: ignore

    (actual, expected) = check_exec_err(f, "TypeError: unsupported operand type")
    assert actual == expected


def test_bool_bitwise_negation() -> None:
    def f(x: bool) -> float:
        """
        pre: x == True
        post: _ == -2
        """
        return ~x

    check_states(f, CONFIRMED)


def test_float_from_hex() -> None:
    def f(s: str) -> float:
        """
        pre: s == '0x3.a7p10'
        post: _ == 3740.0
        """
        return float.fromhex(s)

    check_states(f, CONFIRMED)


def test_int_from_bytes() -> None:
    def f(byt: bytes) -> int:
        """
        pre: len(byt) == 2
        post: _ != 5
        """
        return int.from_bytes(byt, byteorder="little")

    check_states(f, POST_FAIL)


def test_int_nonlinear() -> None:
    def make_bigger(x: int, e: int) -> float:
        """
        pre: e > 1
        post: __return__ !=  592704
        """
        # Expenentation is not SMT-solvable. (z3 gives unsat for this)
        # But CrossHair gracefully falls back to realized values, yielding
        # the counterexample of: 84 ** 3
        return x**e

    check_states(make_bigger, POST_FAIL)


@pytest.mark.demo
def test_int___str___method() -> None:
    def f(x: int) -> str:
        """
        Can any input make this function return the string "321"?

        post: _ != "321"
        """
        return str(x)

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_int___repr___method() -> None:
    def f(x: int) -> str:
        """
        Can any input make this function return the string "321"?

        post: _ != '321'
        """
        return repr(x)

    check_states(f, POST_FAIL)


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

    check_states(f, POST_FAIL)


def test_easy_float_from_str():
    def f(a: str) -> float:
        """
        post: _ != 0.0
        raises: ValueError
        """
        return float(a)

    check_states(
        f,
        MessageType.POST_FAIL,
        AnalysisOptionSet(max_iterations=100, per_condition_timeout=10),
    )


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


def test_int_bitwise_find_negative_input():
    def f(x: int) -> int:
        """
        pre: x < 0
        post: _ != 7
        """
        return x & 255

    check_states(f, POST_FAIL)


@pytest.mark.parametrize("val", [-256, 2**16] + list(range(-4, 9, 2)))
def test_int_bit_length(val):
    with standalone_statespace as space:
        x = proxy_for_type(int, "x")
        space.add(x.var == val)
        assert realize(x.bit_length()) == val.bit_length()


@pytest.mark.parametrize(
    "val", [-256, -(2**15), 2**9, 2**15 - 1] + list(range(-4, 9, 3))
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


def test_class_format():
    with standalone_statespace as space:
        with NoTracing():
            t = SymbolicType("t", Type[int])
            space.add(t.var == SymbolicType._coerce_to_smt_sort(int))
        assert "a{}b".format(t) == "a<class 'int'>b"


@pytest.mark.demo
def test_sorted() -> None:
    def f(lst: List[int]) -> List[int]:
        """
        Can sorting shift the number 4002 to the front?

        pre: len(lst) >= 3
        pre: lst[0] > 4002
        post: _[0] != 4002
        """
        return list(sorted(lst))

    check_states(f, POST_FAIL)


def test_str___bool__() -> None:
    def f(a: str) -> str:
        """post: a"""
        return a

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_str___mul___method() -> None:
    def f(a: str) -> str:
        """
        Can this string-trippling-function produce a 6-character string?

        post: len(_) != 6
        """
        return 3 * a

    check_states(f, POST_FAIL)


def test_str___mul___ok() -> None:
    def f(a: str) -> str:
        """
        pre: len(a) == 2
        post: len(_) == 10
        """
        return a * 3 + 2 * a

    check_states(f, CONFIRMED)


def test_str___mul___by_symbolic_fail() -> None:
    def f(i: int) -> str:
        """post: len(_) != 6"""
        return "a\x00b" * i

    check_states(f, POST_FAIL)


def test_str___mul___full_symbolic_multiply_unknown() -> None:
    def f(s: str, i: int) -> str:
        """
        pre: s and i > 0
        post: _[0] == s[0]
        """
        return s * i

    check_states(f, CANNOT_CONFIRM)


def test_str___add___prefixing_fail() -> None:
    def f(a: str, indent: bool) -> str:
        """post: len(_) == len(a) + indent"""
        return ("  " if indent else "") + a

    check_states(f, POST_FAIL)


def test_str___add___prefixing_ok() -> None:
    def f(a: str, indent: bool) -> str:
        """post: len(_) == len(a) + (2 if indent else 0)"""
        return ("  " if indent else "") + a

    check_states(f, CONFIRMED)


def test_str_find_with_limits_ok() -> None:
    def f(a: str) -> int:
        """post: _ == -1"""
        return a.find("abc", 1, 3)

    check_states(f, CONFIRMED)


def test_str_find_with_negative_limits_fail() -> None:
    def f(a: str) -> int:
        """post: _ == -1"""
        return a.find("ab", -2, 3)

    check_states(f, POST_FAIL)


def test_str_ljust_fail() -> None:
    def f(s: str) -> str:
        """post: len(_) == len(s)"""
        return s.ljust(3, " ")

    check_states(f, POST_FAIL)


def test_str_rfind_with_limits_ok() -> None:
    def f(a: str) -> int:
        """post: _ == -1"""
        return a.rfind("abc", 1, 3)

    check_states(f, CONFIRMED)


def test_str_rfind_with_negative_limits_fail() -> None:
    def f(a: str) -> int:
        """post: _ == -1"""
        return a.rfind("ab", -2, 3)

    check_states(f, POST_FAIL)


def test_str_rindex_fail() -> None:
    def f(a: str) -> int:
        """post: _ != 2"""
        try:
            return a.rindex("abc")
        except ValueError:
            return 0

    check_states(f, POST_FAIL)


def test_str_rindex_err() -> None:
    def f(a: str) -> int:
        """post: True"""
        return a.rindex("abc", 1, 3)

    check_states(f, EXEC_ERR)


def test_str_rjust_fail() -> None:
    def f(s: str) -> str:
        """post: len(_) == len(s)"""
        return s.rjust(3, "x")

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_str_replace_method() -> None:
    def f(a: str) -> str:
        """
        Can this function return a changed string?

        post: _ == a
        """
        return a.replace("abc", "x", 1)

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_str_index_method() -> None:
    def f(a: str) -> int:
        """
        Can we find "abc" at position 2?

        raises: ValueError
        post: _ != 2
        """
        return a.rindex("abc")

    check_states(f, POST_FAIL)


def test_str_index_err() -> None:
    def f(s1: str, s2: str) -> int:
        """
        pre: s1 == 'aba'
        pre: 'ab' in s2
        post: True
        """
        return s1.index(s2)

    # index() raises ValueError when a match isn't found:
    (actual, expected) = check_exec_err(f, "ValueError")
    assert actual == expected


def test_str_negative_index_slicing() -> None:
    def f(s: str) -> Tuple[str, str]:
        """post: sum(map(len, _)) == len(s) - 1"""
        idx = s.find(":")
        return (s[:idx], s[idx + 1 :])

    check_states(f, POST_FAIL)  # (fails when idx == -1)


def test_str_starts_and_ends_ok() -> None:
    def f(s: str) -> str:
        """
        pre: s == 'aba'
        post: s.startswith('ab')
        post: s.endswith('ba')
        """
        return s

    check_states(f, CONFIRMED)


@pytest.mark.demo
def test_str_count_method() -> None:
    def f(s: str) -> int:
        """
        Can this function find two "a" characters?

        post: _ != 2
        """
        return s.count("a")

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_str_split_method() -> None:
    def f(s: str) -> list:
        """
        Does any string comma-split into "a" and "b"?

        post: _ != ['a', 'b']
        """
        return s.split(",")

    check_states(f, POST_FAIL)


def test_str_rsplit_fail() -> None:
    def f(s: str) -> list:
        """post: __return__ != ['a', 'b']"""
        return s.rsplit(":", 1)

    check_states(f, POST_FAIL)


def test_str_partition_ok() -> None:
    def f(s: str) -> tuple:
        """
        pre: len(s) == 3
        post: len(_) == 3
        """
        return s.partition(":")

    check_states(f, CONFIRMED)


@pytest.mark.demo
def test_str_partition_method() -> None:
    def f(s: str) -> tuple:
        """
        Does any input to this partitioning yield ("a", "bc", "d")?

        post: _ != ("a", "bc", "d")
        """
        return s.partition("bc")

    check_states(f, POST_FAIL)


def test_str_rpartition_ok() -> None:
    def f(s: str) -> tuple:
        """
        pre: len(s) == 2
        post: len(_) == 3
        """
        return s.rpartition(":")

    check_states(f, CONFIRMED)


def test_str_rpartition_fail() -> None:
    def f(s: str) -> tuple:
        """
        pre: len(s) == 4
        post: _ != ("abb", "b", "")
        """
        return s.rpartition("b")

    check_states(f, POST_FAIL)


def test_str___ge___fail() -> None:
    def f(s1: str, s2: str) -> bool:
        """post: _"""
        return s1 >= s2

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_str___le___method() -> None:
    def f(a: str, b: str) -> bool:
        """
        Can a be greater than b, even though its first charater is not?

        pre: a[0] <= b[0]
        post: _
        """
        return a <= b

    check_states(f, POST_FAIL)


def test_str_realized_compare() -> None:
    def f(a: str, b: str) -> bool:
        """
        post: implies(_, a == b)
        """
        return realize(a) == b

    check_states(f, CANNOT_CONFIRM)


def test_str_int_comparison_fail() -> None:
    def f(a: int, b: str) -> Tuple[bool, bool]:
        """post: (not _[0]) or (not _[1])"""
        return (a != b, b != a)

    check_states(f, POST_FAIL)


def test_str_int_comparison_ok() -> None:
    def f(a: int, b: str) -> bool:
        """post: _ == False"""
        return a == b or b == a

    check_states(f, CONFIRMED)


def test_str_formatting_wrong_key() -> None:
    def f(o: object) -> str:
        """post: True"""
        return "object of type {typ} with repr {zzzzz}".format(  # type: ignore
            typ=type(o), rep=repr(o)
        )

    check_states(f, EXEC_ERR)


def test_str_format_symbolic_format() -> None:
    def f(fmt: str) -> str:
        """
        pre: '{}' in fmt
        post: True
        """
        return fmt.format(ver=sys.version, platform=sys.platform)

    check_states(f, EXEC_ERR)


def test_str_format_percent_unknown() -> None:
    def f(fmt: str) -> str:
        """
        pre: '%' not in fmt
        post: True
        """
        return fmt % ()

    check_states(f, CANNOT_CONFIRM)


@pytest.mark.demo
def test_str_join_method() -> None:
    def f(items: List[str]) -> str:
        """
        Any inputs that produce a 5-character string?

        pre: len(items) == 2
        post: len(_) != 5
        """
        return "and".join(items)

    check_states(f, POST_FAIL)


def test_str_upper_fail() -> None:
    def f(s: str) -> str:
        """
        Does any character uppercase to "F"?

        pre: len(s) == 1
        pre: s != "F"
        post: __return__ != "F"
        """
        return s.upper()

    # TODO: make this use case more efficient.
    options = AnalysisOptionSet(per_condition_timeout=60.0, per_path_timeout=20.0)
    check_states(f, POST_FAIL, options)


def test_csv_example() -> None:
    def f(lines: List[str]) -> List[str]:
        """
        pre: all(',' in line for line in lines)
        post: __return__ == [line.split(',')[0] for line in lines]
        """
        return [line[: line.index(",")] for line in lines]

    options = AnalysisOptionSet(per_condition_timeout=5)
    check_states(f, CANNOT_CONFIRM, options)


@pytest.mark.demo
def test_str_zfill_method() -> None:
    def f(s: str) -> str:
        """
        Can zero-filling a two-character string produce "0ab"?

        pre: len(s) == 2
        post: _ != "0ab"
        """
        return s.zfill(3)

    check_states(f, POST_FAIL)


@pytest.mark.demo("yellow")
def test_str_format_method() -> None:
    def f(s: str) -> str:
        """
        Does any substitution produce the string "abcdef"?

        NOTE: CrossHair will not be effective with a symbolic template string;
        e.g. trying to solve s.format("cd") is much more difficult.

        post: _ != "abcdef"
        """
        return "ab{}ef".format(s)

    check_states(f, POST_FAIL)


def test_str_constructor() -> None:
    with standalone_statespace as space:
        with NoTracing():
            x = LazyIntSymbolicStr("x")
        assert str(x) is x


def test_str_str() -> None:
    with standalone_statespace:
        with NoTracing():
            x = LazyIntSymbolicStr("x")
        strx = x.__str__()
        with NoTracing():
            assert isinstance(strx, str)


def test_str_center():
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


def test_str_map_chars() -> None:
    with standalone_statespace:
        with NoTracing():
            string = LazyIntSymbolicStr(list(map(ord, "ab")))
        codepoints = list(map(ord, string))


@pytest.mark.demo
def test_str___add___method() -> None:
    def f(s: str) -> str:
        """
        Can any input make this function return "Hello World"?

        post: _ != "Hello World"
        """
        return s + "World"

    check_states(f, POST_FAIL)


def test_str_bool():
    with standalone_statespace as space, NoTracing():
        a = LazyIntSymbolicStr("a")
        space.add(a.__len__().var > 0)
        with ResumedTracing():
            assert bool(a)
        # Can we retain our symbolic state after forcing a positive truthiness?:
        assert space.is_possible((a == "this").var)
        assert space.is_possible((a == "that").var)


def test_str_eq():
    with standalone_statespace, NoTracing():
        assert LazyIntSymbolicStr([]) == ""


def test_str_getitem():
    with standalone_statespace, NoTracing():
        assert LazyIntSymbolicStr(list(map(ord, "abc")))[0] == "a"
        assert LazyIntSymbolicStr(list(map(ord, "abc")))[-1] == "c"
        assert LazyIntSymbolicStr(list(map(ord, "abc")))[-5:2] == "ab"


def test_str_filter():
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "  ")))
        with ResumedTracing():
            ret = list(filter(str.isspace, [string]))
        assert ret == [string]


def test_str_find() -> None:
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "aabc")))
        assert string.find("ab") == 1


def test_str_find_symbolic() -> None:
    def f(s: str) -> int:
        """
        pre: len(s) == 3
        post: _ == -1
        """
        return "haystack".find(s)

    check_states(f, POST_FAIL)


def test_str_find_notfound() -> None:
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr([])
        assert string.find("abc", 1, 3) == -1


def test_str_format_basic():
    with standalone_statespace as space:
        with NoTracing():
            s = LazyIntSymbolicStr("s")
            space.add(s.__len__().var == 1)
        assert space.is_possible((s == "z").var)
        assert space.is_possible((ord("a{0}c".format(s)[1]) == ord("b")).var)


def test_str_format_map():
    with standalone_statespace as space:
        with NoTracing():
            s = LazyIntSymbolicStr("s")
            space.add(s.__len__().var == 1)
        assert space.is_possible((s == "z").var)
        assert space.is_possible(
            (ord("a{foo}c".format_map({"foo": s})[1]) == ord("b")).var
        )


def test_str_rfind() -> None:
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "ababb")))
        assert string.rfind("ab") == 2


def test_str_rfind_notfound() -> None:
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "ababb")))
        assert string.rfind("ab") == 2


def test_str_split_limits():
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "a:b:c")))
        parts = realize(string.split(":", 1))
        assert parts == ["a", "b:c"]
        parts = realize(string.split(":"))
        assert parts == ["a", "b", "c"]


def test_str_rsplit():
    with standalone_statespace, NoTracing():
        string = LazyIntSymbolicStr(list(map(ord, "a:b:c")))
        parts = realize(string.rsplit(":", 1))
        assert parts == ["a:b", "c"]


def test_str_contains():
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


def test_str_deep_realize():
    with standalone_statespace, NoTracing():
        a = LazyIntSymbolicStr("a")
        tupl = (a, (a,))
        realized = deep_realize(tupl)
    assert list(map(type, realized)) == [str, tuple]
    assert list(map(type, realized[1])) == [str]
    assert realized[0] is realized[1][0]


def test_str_strip():
    with standalone_statespace:
        with NoTracing():
            x = LazyIntSymbolicStr(list(map(ord, "  A b\n")))
        assert x.strip() == "A b"


def test_str_lower():
    chr_Idot = "\u0130"  # Capital I with dot above
    # (it's the only unicde char that lower()s to 2 characters)
    with standalone_statespace:
        with NoTracing():
            x = LazyIntSymbolicStr(list(map(ord, "Ab" + chr_Idot)))
        assert x.lower() == "abi\u0307"


def test_str_title():
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


@pytest.mark.demo
def test_tuple___add___method():
    def f(a: Tuple[int, ...]):
        """
        Can we get this function to return (1, 2, 3, 4)?

        post: _ != (1, 2, 3, 4)
        """
        return (1,) + a + (4,)

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_tuple___getitem___method() -> None:
    def f(t: Tuple[int, ...], idx: int) -> int:
        """
        Can we find 42 in the given tuple at the given index?

        pre: idx >= 0 and idx < len(t)
        post: _ != 42
        """
        return t[idx]

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_tuple___len___method():
    def f(a: Tuple[int, ...]):
        """
        Can we find a tuple of length 8?

        post: _ != 8
        """
        return len(a)

    check_states(f, POST_FAIL)


def test_tuple_range_intersection_fail() -> None:
    def f(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        pre: a[0] < a[1] and b[0] < b[1]
        post: _[0] <= _[1]
        """
        return (max(a[0], b[0]), min(a[1], b[1]))

    check_states(f, POST_FAIL)


def test_tuple_range_intersection_ok() -> None:
    def f(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        pre: a[0] < a[1] and b[0] < b[1]
        post: _ is None or _[0] <= _[1]
        """
        if a[1] > b[0] and a[0] < b[1]:  # (if the ranges overlap)
            return (max(a[0], b[0]), min(a[1], b[1]))
        else:
            return None

    check_states(f, CONFIRMED)


def test_tuple_with_uniform_values_fail() -> None:
    def f(a: Tuple[int, ...]) -> float:
        """
        post: True
        """
        return sum(a) / len(a)

    check_states(f, EXEC_ERR)


def test_tuple_with_uniform_values_ok() -> None:
    def f(a: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        pre: len(a) < 4
        post: 0 not in _
        """
        return tuple(x for x in a if x)

    check_states(f, CONFIRMED)


def test_tuple_runtime_type() -> None:
    def f(t: Tuple) -> Tuple:
        """post: t != (1, 2)"""
        return t

    check_states(f, POST_FAIL)


def test_tuple_isinstance_check() -> None:
    def f(uniform_tuple: Tuple[List, ...], basic_tuple: tuple) -> Tuple[bool, bool]:
        """post: _ == (True, True)"""
        return (isinstance(uniform_tuple, tuple), isinstance(basic_tuple, tuple))

    check_states(f, CONFIRMED)


def test_range_unknown() -> None:
    def f(a: int) -> Iterable[int]:
        """post: len(_) == a or a < 0"""
        return range(a)

    check_states(f, CANNOT_CONFIRM)


@pytest.mark.demo
def test_list___contains___method() -> None:
    def f(a: int, b: List[int]) -> bool:
        """
        Is full containment checking equivalent to checking the first 3 elements?

        post: _ == (a in b[:3])
        """
        return a in b

    check_states(f, POST_FAIL)


def test_list___contains___ok() -> None:
    def f(a: int, b: List[int]) -> bool:
        """
        pre: 1 == len(b)
        post: _ == (a == b[0])
        """
        return a in b

    check_states(f, CONFIRMED)


@pytest.mark.demo
def test_list___add___method() -> None:
    def f(a: List[int]) -> List[int]:
        """
        Does doubling the list always make it longer?

        post: len(_) > len(a)
        """
        return a + a

    check_states(f, POST_FAIL)


def test_list_doubling_ok() -> None:
    def f(a: List[int]) -> List[int]:
        """
        post: len(_) > len(a) or not a
        """
        return a + a

    check_states(f, CONFIRMED)


def test_list_multiply_ok() -> None:
    def f(a: List[int]) -> List[int]:
        """post: len(_) == len(a) * 5"""
        return a * 3 + 2 * a

    check_states(f, CONFIRMED)


def test_list_average() -> None:
    def average(numbers: List[float]) -> float:
        """
        pre: len(numbers) > 0
        post: min(numbers) <= _ <= max(numbers)
        """
        return sum(numbers) / len(numbers)

    check_states(average, CANNOT_CONFIRM)


def test_list_mixed_symbolic_and_literal_concat_ok() -> None:
    def f(ls: List[int], i: int) -> List[int]:
        """
        pre: i >= 0
        post: len(_) == len(ls) + 1
        """
        return (
            ls[:i]
            + [
                42,
            ]
            + ls[i:]
        )

    check_states(f, CONFIRMED)


def test_list_range_fail() -> None:
    def f(ls: List[int]) -> List[int]:
        """
        pre: len(ls) == 3
        post: len(_) > len(ls)
        """
        n: List[int] = []
        for i in range(len(ls)):
            n.append(ls[i] + 1)
        return n

    check_states(f, POST_FAIL)


def test_list_range_ok() -> None:
    def f(ls: List[int]) -> List[int]:
        """
        pre: ls and len(ls) < 10  # (max is to cap runtime)
        post: _[0] == ls[0] + 1
        """
        n: List[int] = []
        for i in range(len(ls)):
            n.append(ls[i] + 1)
        return n

    check_states(f, CONFIRMED)


def test_list_equality() -> None:
    def f(ls: List[int]) -> List[int]:
        """
        pre: len(ls) > 0
        post: _ != ls
        """
        # extra check for positive equality:
        assert ls == [x for x in ls], "list does not equal itself"
        nl = ls[:]
        nl[0] = 42
        return nl

    check_states(f, POST_FAIL)


def test_list_extend_literal_unknown() -> None:
    def f(ls: List[int]) -> List[int]:
        """
        post: _[:2] == [1, 2]
        """
        r = [1, 2, 3]
        r.extend(ls)
        return r

    check_states(f, CANNOT_CONFIRM)


@pytest.mark.demo
def test_list___getitem___method() -> None:
    def f(ls: List[int], idx: int) -> int:
        """
        Can we find 42 in the given list at the given index?

        pre: idx >= 0 and idx < len(ls)
        post: _ != 42
        """
        return ls[idx]

    check_states(f, POST_FAIL)


def test_list____getitem___error() -> None:
    def f(ls: List[int], idx: int) -> int:
        """
        pre: idx >= 0 and len(ls) > 2
        post: True
        """
        return ls[idx]

    (actual, expected) = check_exec_err(f, "IndexError")
    assert actual == expected


def test_list____getitem___type_error() -> None:
    def f(ls: List[int]) -> int:
        """post: True"""
        return ls[0.0:]  # type: ignore

    (actual, expected) = check_exec_err(f, "TypeError")
    assert actual == expected


def test_list____getitem___ok() -> None:
    def f(ls: List[int]) -> bool:
        """
        pre: len(ls) <= 3
        post: _ == (7 in ls)
        """
        try:
            return ls[ls.index(7)] == 7
            return True
        except ValueError:
            return False

    check_states(f, CONFIRMED)


def test_list_nested_lists_fail() -> None:
    def f(ls: List[List[int]]) -> int:
        """
        post: _ > 0
        """
        total = 0
        for i in ls:
            total += len(i)
        return total

    check_states(f, POST_FAIL)


def test_list_nested_lists_ok() -> None:
    def f(ls: List[List[int]]) -> int:
        """
        pre: len(ls) < 4
        post: _ >= 0
        """
        total = 0
        for i in ls:
            total += len(i)
        return total

    check_states(f, CONFIRMED)


def test_list_iterable() -> None:
    def f(a: Iterable[int]) -> int:
        """
        pre: a
        post: _ in a
        """
        return next(iter(a))

    check_states(f, CONFIRMED)


def test_list_isinstance_check() -> None:
    def f(ls: List) -> bool:
        """post: _"""
        return isinstance(ls, list)

    check_states(f, CONFIRMED)


def test_list_slice_outside_range_ok() -> None:
    def f(ls: List[int], i: int) -> List[int]:
        """
        pre: i >= len(ls)
        post: _ == ls
        """
        return ls[:i]

    check_states(f, CONFIRMED)


def test_list_slice_amount() -> None:
    def f(ls: List[int]) -> List[int]:
        """
        pre: len(ls) >= 3
        post: len(_) == 1
        """
        return ls[2:3]

    check_states(f, CONFIRMED)


def test_list____setitem___ok() -> None:
    def f(ls: List[int]) -> None:
        """
        pre: len(ls) >= 2
        post[ls]:
            ls[1] == 42
            ls[2] == 43
            len(ls) == 4
        """
        ls[1:-1] = [42, 43]

    check_states(f, CONFIRMED)


def test_list___setitem___out_of_bounds() -> None:
    def f(ls: List[int], i: int) -> None:
        """
        pre: i != -1
        post: ls == __old__.ls[:i] + __old__.ls[i+1:]
        """
        ls[i : i + 1] = []

    check_states(f, CANNOT_CONFIRM)


def test_list_insert_ok() -> None:
    def f(ls: List[int]) -> None:
        """
        pre: len(ls) == 4
        post[ls]:
            len(ls) == 5
            ls[2] == 42
        """
        ls.insert(-2, 42)

    check_states(f, CONFIRMED)


def test_list_insert_with_conversions() -> None:
    def f(ls: List[Set[int]], a: bool, b: int) -> None:
        """
        # self.insert(a,b) with {'a': True, 'b': 10, 'self': [{0}]}
        post: True
        """
        ls.insert(a, b)  # type: ignore

    check_states(f, CONFIRMED)


def test_list_pop_ok() -> None:
    def f(ls: List[int]) -> None:
        """
        pre: ls == [4, 5]
        post: ls == [4]
        """
        ls.pop()

    check_states(f, CONFIRMED)


def test_list_count_ok() -> None:
    def f(ls: List[Dict[int, Dict[int, int]]]) -> int:
        """
        pre: ls == [{1: {2: 3}}]
        post: _ == 1
        """
        return ls.count({1: {2: 3}})

    check_states(f, CONFIRMED)


def test_list___setitem___ok() -> None:
    def f(ls: List[int]) -> None:
        """
        pre: len(ls) >= 4
        post[ls]: ls[3] == 42
        """
        ls[3] = 42

    check_states(f, CONFIRMED)


@pytest.mark.demo
def test_list___delitem___method() -> None:
    def f(ls: List[int]) -> None:
        """
        Can we trim the tail two elements and have three left over?

        post[ls]: len(ls) != 3
        """
        del ls[-2:]

    check_states(f, POST_FAIL)


def test_list___delitem___ok() -> None:
    def f(ls: List[int]) -> None:
        """
        pre: len(ls) == 5
        post[ls]: len(ls) == 4
        """
        del ls[2]

    check_states(f, CONFIRMED)


def test_list___delitem___type_error() -> None:
    def f(ls: List[float]) -> None:
        """
        pre: len(ls) == 0
        post: True
        """
        del ls[1.0]  # type: ignore

    (actual, expected) = check_exec_err(f, "TypeError")
    assert actual == expected


def test_list___delitem___out_of_bounds() -> None:
    def f(ls: List[float]) -> None:
        """post: True"""
        del ls[1]

    (actual, expected) = check_exec_err(f, "IndexError")
    assert actual == expected


def test_list_sort_ok() -> None:
    def f(ls: List[int]) -> None:
        """
        pre: len(ls) == 3
        post[ls]: ls[0] == min(ls)
        """
        ls.sort()

    check_states(f, CONFIRMED)


def test_list_reverse_ok() -> None:
    def f(ls: List[int]) -> None:
        """
        pre: len(ls) == 2
        post[ls]: ls[0] == 42
        """
        ls.append(42)
        ls.reverse()

    check_states(f, CONFIRMED)


def test_list_comparison_type_error() -> None:
    def f(a: List[Set], b: str):
        """post: True"""
        return a <= b  # type: ignore

    (actual, expected) = check_exec_err(f, "TypeError")
    assert actual == expected


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


def test_dict___bool___ok() -> None:
    def f(a: Dict[int, str]) -> bool:
        """
        post[a]: _ == True
        """
        a[0] = "zero"
        return bool(a)

    check_states(f, CONFIRMED)


def test_dict___iter__() -> None:
    def f(a: Dict[int, str]) -> List[int]:
        """
        post[a]: 5 in _
        """
        a[10] = "ten"
        return list(a.__iter__())

    check_states(f, POST_FAIL)


def test_dict___iter___ok() -> None:
    def f(a: Dict[int, str]) -> List[int]:
        """
        pre: len(a) < 3
        post[a]: 10 in _
        """
        a[10] = "ten"
        return list(a.__iter__())

    check_states(f, CONFIRMED)


@pytest.mark.demo("yellow")
def test_dict___delitem___method() -> None:
    def f(a: Dict[str, int]) -> None:
        """
        Can deleting the key "foo" leave an empty dictionary?

        NOTE: Deleting a symbolic key from a concrete dictionary is not effectively
        reasoned about at present:

            dictionary | key      | effective?
            -----------+----------+-----------
            symbolic  | *        | yes
            *         | concrete | yes
            concrete  | symbolic | no


        raises: KeyError
        post[a]: len(a) != 0
        """
        del a["foo"]

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_dict___eq___method() -> None:
    def f(t: dict) -> dict:
        """
        Can we find a dictionary that maps 50 to 100?

        post: t != {50: 100}
        """
        return t

    check_states(f, POST_FAIL)


def test_dict___eq___deep() -> None:
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

    check_states(f, POST_FAIL)


def test_dict___eq___ok() -> None:
    def f(d: Dict[int, int]) -> Dict[int, int]:
        """post: _ == {**_}"""
        return d

    check_states(f, CANNOT_CONFIRM)


@pytest.mark.demo
def test_dict___getitem___method() -> None:
    def f(m: Dict[int, int]):
        """
        Can we make a path from 0 to 2, by indexing into the dictionary twice?

        pre: len(m) == 2
        raises: KeyError
        post: _ != 2
        """
        return m[m[0]]

    check_states(f, POST_FAIL)


def test_dict___getitem___implicit_conversion_for_keys_fail() -> None:
    def f(m: Dict[complex, float], b: bool, i: int):
        """
        pre: not m
        post: len(m) != 1
        """
        m[b] = 2.0
        m[i] = 3.0

    check_states(f, POST_FAIL)


@pytest.mark.demo("yellow")
def test_dict___setitem___method() -> None:
    def f(a: Dict[int, int], k: int, v: int) -> None:
        """
        Can we make a dictionary assignment, and be left with {4: 5, 10: 20}?

        NOTE: CrossHair cannot effectively handle the assignment of a symbolic key on a
        concrete dictionary, e.g. `d={4:5}; d[k] = 20`

            dictionary | key      | value | effective?
            -----------+----------+-------+-----------
            symbolic  | *        | *     | yes
            *         | concrete | *     | yes
            concrete  | symbolic | *     | no


        post[a]: a != {4: 5, 10: 20}
        """
        a[k] = v

    check_states(f, POST_FAIL)


def test_dict___setitem___ok() -> None:
    def f(a: Dict[int, int], k: int, v: int) -> None:
        """
        post[a]: a[k] == v
        """
        a[k] = v

    check_states(f, CONFIRMED)


def test_dict___setitem___on_copy() -> None:
    def f(d: Dict[int, int]) -> Dict[int, int]:
        """post: _ != d"""
        d = d.copy()
        d[42] = 100
        return d

    check_states(f, POST_FAIL)


def TODO_test_dict___setitem___on_concrete() -> None:
    # NOTE: This is very challenging to implement: opcode interception could upgrade
    # the concrete dictionary to a symbolic, but the original dictionary may be aliased.
    # One investigation: start everything out as symbolic by intercepting BUILD_MAP and
    # BUILD_SET. This potentially also lets us detect writes to persistent state
    # (because all pre-existing dicts/sets will be concrete).
    def f(k: int, v: int) -> Dict[int, int]:
        """post: _[100] == 200"""
        d = {100: 200}
        d[k] = v
        return d

    check_states(f, POST_FAIL)


def test_dict___str__() -> None:
    def f(a: Dict[int, str]) -> str:
        """
        pre: len(a) == 0
        post: _ == '{}'
        """
        return str(a)

    check_states(f, CONFIRMED)


@pytest.mark.demo
def test_dict_get_method():
    def f(x: int) -> int:
        """
        Can we find the key that is mapped to 5?

        post: _ != 5
        """
        a = {2: 3, 4: 5, 6: 7}
        return a.get(x, 9)

    check_states(f, POST_FAIL)


def test_dict_get_with_defaults_ok() -> None:
    def f(a: Dict[int, int]) -> int:
        """post: (_ == 2) or (_ == a[4])"""
        return a.get(4, 2)

    check_states(f, CONFIRMED)


def test_dict_items_ok() -> None:
    def f(a: Dict[int, str]) -> Iterable[Tuple[int, str]]:
        """
        pre: len(a) < 5
        post[a]: (10,'ten') in _
        """
        a[10] = "ten"
        return a.items()

    check_states(f, CONFIRMED)


def test_dict_setdefault_float_int_comparison() -> None:
    def f(a: Dict[int, int]):
        """
        pre: a == {2: 0}
        post: _ == 0
        """
        return a.setdefault(2.0, {True: "0"})  # type: ignore

    check_states(f, CONFIRMED)


def test_dict_over_objects() -> None:
    def f(a: Dict[object, object]) -> int:
        """
        post: _ >= 0
        """
        return len(a)

    check_states(f, CONFIRMED)


def test_dict_over_heap_objects() -> None:
    def f(a: Dict[Tuple[int], int]) -> Optional[int]:
        """
        post: _ != 10
        """
        return a.get((5,))

    check_states(f, POST_FAIL)


def test_dict_complex_contents() -> None:
    def f(d: Dict[Tuple[int, bool], Tuple[float, int]]) -> int:
        """
        post: _ > 0
        """
        if (42, True) in d:
            return d[(42, True)][1]
        else:
            return 42

    check_states(f, MessageType.POST_FAIL, AnalysisOptionSet(per_condition_timeout=5))


def test_dict_isinstance_check() -> None:
    def f(smtdict: Dict[int, int], heapdict: Dict) -> Tuple[bool, bool]:
        """post: _ == (True, True)"""
        return (isinstance(smtdict, dict), isinstance(heapdict, dict))

    check_states(f, CONFIRMED)


def test_dict_subtype_lookup() -> None:
    def f(d: Dict[Tuple[int, str], int]) -> None:
        """
        pre: not d
        post[d]: [(42, 'fourty-two')] == list(d.keys())
        """
        d[(42, "fourty-two")] = 1

    check_states(f, CONFIRMED)


def test_dict_complex_keys() -> None:
    def f(dx: Dict[Tuple[int, str], int]) -> None:
        """
        pre: not dx
        post[dx]:
            len(dx) == 1
            dx[(42, 'fourty-two')] == 1
        """
        dx[(42, "fourty-two")] = 1

    check_states(f, CONFIRMED)


def test_dict_has_unique_keys() -> None:
    def f(d: Dict[Tuple[int, str], int]) -> None:
        """
        pre: len(d) == 2 and (1, 'one') in d
        post[d]: (1, 'one') not in d
        """
        del d[(1, "one")]

    check_states(f, CONFIRMED)


def test_dict_wrong_key_type() -> None:
    def f(d: Dict[int, int], s: str, i: int) -> bool:
        """
        post: _
        raises: KeyError
        """
        if i == 0:
            del d[s]  # type: ignore
        elif i < 0:
            d[s] = 7  # type: ignore
        else:
            _val = d[s]  # type: ignore
        return True

    check_states(f, CANNOT_CONFIRM)


def test_dict_key_type_union() -> None:
    def f(d: Dict[Union[int, str], int]) -> Dict:
        """
        pre: len(d) == 2
        post: not (42 in d and '42' in d)
        """
        return d

    check_states(f, POST_FAIL)


if sys.version_info >= (3, 10):

    def test_dict_type_union_operator() -> None:
        def f(a: int | str, b: int | str) -> Tuple[int | str, int | str]:
            """post: _ != (42, "hi")"""
            return (a, b)

        check_states(f, POST_FAIL)


def test_dict_nonuniform_dict_key_types() -> None:
    def f(a: Dict[Hashable, int]) -> Dict[Hashable, int]:
        """
        pre: len(a) == 1
            post: _[0] == 100
        """
        b: Dict[Hashable, int] = {0: 100}
        b.update(a)
        return b

    check_states(f, POST_FAIL)


def test_dict_inside_lists() -> None:
    def f(dicts: List[Dict[int, int]]) -> Dict[int, int]:
        """
        pre: len(dicts) <= 1  # to narrow search space (would love to make this larger)
        post: len(_) <= len(dicts)
        """
        ret = {}
        for d in dicts:
            ret.update(d)
        return ret

    check_states(f, POST_FAIL)


def test_dict_inside_lists_with_identity() -> None:
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

    check_states(f, POST_FAIL)


def test_dict_consistent_ordering() -> None:
    def f(symbolic: Dict[int, int]) -> Tuple[List[int], List[int]]:
        """post: _[0] == _[1]"""
        return (list(symbolic.keys()), list(symbolic.keys()))

    check_states(f, CANNOT_CONFIRM)


def test_dict_ordering_after_mutations() -> None:
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

    check_states(f, CONFIRMED)


def test_dict_alternate_mapping_types() -> None:
    def f(m1: Mapping[int, int], m2: MutableMapping[int, int]) -> int:
        """
        pre: 1 in m1 and 2 in m2
        post: _ != 10
        """
        return m1[1] + m2[2]

    check_states(f, POST_FAIL)


def test_dict_untyped_access():
    def f(d: dict, k: int) -> dict:
        """
        pre: 42 in d
        post: 42 in __return__
        raises: KeyError
        """
        del d[k]
        return d

    # TODO: profile / optimize
    check_states(
        f,
        MessageType.POST_FAIL,
        AnalysisOptionSet(per_condition_timeout=90),
    )


if sys.version_info >= (3, 8):

    def test_TypedDict_fail() -> None:
        def f(td: Movie):
            '''post: _['year'] != 2020 or _['name'] != "hi"'''
            return td

        check_states(f, POST_FAIL)


def test_set_basic_fail() -> None:
    def f(a: Set[int], k: int) -> None:
        """
        post[a]: k+1 in a
        """
        a.add(k)

    check_states(f, POST_FAIL)


def test_set_basic_ok() -> None:
    def f(a: Set[int], k: int) -> None:
        """
        post[a]: k in a
        """
        a.add(k)

    check_states(f, CONFIRMED)


def test_set_union_fail() -> None:
    def f(a: Set[str], b: Set[str]) -> Set[str]:
        """
        pre: len(a) == len(b) == 1  # (just for test performance)
        post: all(((i in a) and (i in b)) for i in _)
        """
        return a | b

    check_states(f, POST_FAIL)


def test_set_union_ok() -> None:
    def f(a: Set[str], b: Set[str]) -> Set[str]:
        """
        post: all(((i in a) or (i in b)) for i in _)
        """
        return a | b

    check_states(f, CANNOT_CONFIRM)


def test_set_contains_different_but_equivalent() -> None:
    def f(s: Set[Union[int, str]]) -> str:
        """
        pre: "foobar" in s
        post: (_ + "bar") in s
        """
        return "foo"

    check_states(f, CANNOT_CONFIRM)


# The heaprefs + deferred set assumptions make this too expensive.
# TODO: Optimize & re-enable
def TODO_set_test_subtype_union() -> None:
    def f(s: Set[Union[int, str]]) -> Set[Union[int, str]]:
        """post: not ((42 in s) and ('42' in s))"""
        return s

    check_states(f, MessageType.POST_FAIL, AnalysisOptionSet(per_condition_timeout=7.0))


def test_set_subset_compare_ok() -> None:
    # a >= b with {'a': {0.0, 1.0}, 'b': {2.0}}
    def f(s1: Set[int], s2: Set[int]) -> bool:
        """
        pre: s1 == {0, 1}
        pre: s2 == {2}
        post: not _
        """
        return s1 >= s2

    check_states(f, CONFIRMED)


def test_set_numeric_promotion() -> None:
    def f(b: bool, s: Set[int]) -> bool:
        """
        pre: b == True
        pre: s == {1}
        post: _
        """
        return b in s

    check_states(f, CONFIRMED)


def test_set_runtime_type_ok() -> None:
    def f(s: set) -> bool:
        """post: _"""
        return True

    check_states(f, CONFIRMED)


def test_set_isinstance_check() -> None:
    def f(s: Set[object]) -> bool:
        """post: _"""
        return isinstance(s, set)

    check_states(f, CONFIRMED)


def test_set___eq__() -> None:
    def f(a: Set[FrozenSet[int]]) -> object:
        """
        pre: a == {frozenset({7}), frozenset({42})}
        post: _ in ('{frozenset({7}), frozenset({42})}', '{frozenset({42}), frozenset({7})}')
        """
        return repr(a)

    check_states(
        f,
        MessageType.CONFIRMED,
        AnalysisOptionSet(per_path_timeout=10, per_condition_timeout=10),
    )


def test_set_no_duplicates() -> None:
    def f(s: Set[int]) -> int:
        """
        pre: len(s) == 2
        post: _
        """
        i = iter(s)
        x = next(i)
        y = next(i)
        return x != y

    check_states(f, CONFIRMED)


def test_frozenset_realize():
    with standalone_statespace as space:
        with NoTracing():
            x = proxy_for_type(FrozenSet[int], "x")
            y = realize(x)
            assert type(y) is frozenset
        assert type(x) is frozenset


def test_set_realize():
    with standalone_statespace as space:
        with NoTracing():
            x = proxy_for_type(Set[str], "x")
            assert type(x) is not set
            y = realize(x)
            assert type(y) is set
        assert type(x) is set


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


class ProtocolsTest(unittest.TestCase):
    # TODO: move most of this into a collectionslib_test.py file
    def test_hashable_values_fail(self) -> None:
        def f(b: bool, i: int, t: Tuple[str, ...]) -> int:
            """post: _ % 5 != 0"""
            return hash((b, i, t))

        check_states(f, POST_FAIL)

    def test_hashable_values_ok(self) -> None:
        def f(a: Tuple[str, int, float, bool], b: Tuple[str, int, float, bool]) -> int:
            """post: _ or not (a == b)"""
            return hash(a) == hash(b)

        check_states(f, CANNOT_CONFIRM)

    def test_symbolic_hashable(self) -> None:
        def f(a: Hashable) -> int:
            """post[]: 0 <= _ <= 1"""
            return hash(a) % 2

        check_states(f, CONFIRMED)

    def test_symbolic_supports(self) -> None:
        def f(
            a: SupportsAbs,
            f: SupportsFloat,
            i: SupportsInt,
            r: SupportsRound,
            # c: SupportsComplex,  # TODO: symbolic complex not yet really working
            b: SupportsBytes,
        ) -> float:
            """post: _.real <= 0"""
            return abs(a) + float(f) + int(i) + round(r) + len(bytes(b))
            # + complex(c)

        check_states(f, POST_FAIL)

    def test_iterable(self) -> None:
        T = TypeVar("T")

        def f(a: Iterable[T]) -> T:
            """
            pre: a
            post: _ in a
            """
            return next(iter(a))

        check_states(f, CANNOT_CONFIRM)

    def test_bare_type(self) -> None:
        def f(a: List) -> bool:
            """
            pre: a
            post: _
            """
            return bool(a)

        check_states(f, CONFIRMED)


def test_enum_identity_matches_equality() -> None:
    def f(color1: Color, color2: Color) -> bool:
        """post: _ == (color1 is color2)"""
        return color1 == color2

    check_states(f, CONFIRMED)


def test_enum_in_container() -> None:
    def f(colors: List[Color]) -> bool:
        """post: not _"""
        return Color.RED in colors and Color.BLUE in colors

    check_states(f, POST_FAIL)


def test_type_issubclass_ok() -> None:
    def f(typ: Type[SmokeDetector]):
        """post: _"""
        return issubclass(typ, SmokeDetector)

    check_states(f, CONFIRMED)


def test_type_can_be_a_subclass() -> None:
    def f(typ: Type[Cat]):
        """post: _ == "<class '__main__.Cat'>" """
        return str(typ)

    # False when the type is instantiated as "BiggerCat":
    check_states(f, POST_FAIL)


def test_type_issubclass_fail() -> None:
    def f(typ: Type):
        """post: _"""
        return issubclass(typ, str)

    check_states(f, POST_FAIL)


def test_type_symbolics_without_literal_types() -> None:
    def f(typ1: Type, typ2: Type[bool], typ3: Type):
        """post: implies(_, issubclass(typ1, typ3))"""
        # The counterexample we expect: typ1==str typ2==bool typ3==int
        return issubclass(typ2, typ3) and typ2 != typ3

    check_states(
        f,
        POST_FAIL,
        AnalysisOptionSet(max_iterations=60, per_condition_timeout=10),
    )


def test_type_instance_creation() -> None:
    def f(t: Type[Cat]):
        """post: _.size() > 0"""
        return t()

    check_states(f, CONFIRMED)


def test_type_comparison() -> None:
    def f(t: Type) -> bool:
        """post: _"""
        return t == int

    check_states(f, POST_FAIL)


def test_type_as_bool() -> None:
    def f(t: Type) -> bool:
        """post: _"""
        return bool(t)

    check_states(f, CONFIRMED)


def test_type_generic_object_and_type() -> None:
    def f(thing: object, detector_kind: Type[SmokeDetector]):
        """post: True"""
        if isinstance(thing, detector_kind):
            return thing._is_plugged_in
        return False

    check_states(f, CANNOT_CONFIRM)


def test_object___eq__() -> None:
    def f(thing: object, i: int):
        """post: not _"""
        return thing == i

    check_states(f, POST_FAIL)


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


def test_object_with_comparison():
    def f(obj):
        """post: _"""
        return obj != b"abc"

    check_states(f, POST_FAIL)


def test_callable_zero_args() -> None:
    def f(size: int, initializer: Callable[[], int]) -> Tuple[int, ...]:
        """
        pre: size >= 1
        post: _[0] != 707
        """
        return tuple(initializer() for _ in range(size))

    check_states(f, POST_FAIL)


def test_callable_one_arg() -> None:
    def f(size: int, mapfn: Callable[[int], int]) -> Tuple[int, ...]:
        """
        pre: size >= 1
        post: _[0] != 707
        """
        return tuple(mapfn(i) for i in range(size))

    check_states(f, POST_FAIL)


def test_callable_two_args() -> None:
    def f(i: int, c: Callable[[int, int], int]) -> int:
        """post: _ != i"""
        return c(i, i)

    check_states(f, POST_FAIL)


def test_callable_as_bool() -> None:
    def f(fn: Callable[[int], int]) -> bool:
        """post: _"""
        return bool(fn)

    check_states(f, CONFIRMED)


def test_callable_repr() -> None:
    def f(f1: Callable[[int], int]) -> int:
        """post: _ != 1234"""
        return f1(4)

    messages = run_checkables(analyze_function(f))
    assert len(messages) == 1
    assert (
        messages[0].message
        == "false when calling f(lambda *a: 1234) (which returns 1234)"
    )


def test_callable_with_typevar_in_args() -> None:
    T = TypeVar("T")

    def f(a: Callable[[T], int], x: T) -> int:
        """post: _ != 42"""
        return a(x)

    check_states(f, POST_FAIL)


def test_callable_with_typevar_in_return() -> None:
    T = TypeVar("T")

    def f(a: Callable[[int], T], x: int) -> T:
        """post: _"""
        return a(x)

    check_states(f, POST_FAIL)


def TODO_test_callable_with_typevars() -> None:
    # Right now, this incorrectly reports a counterexample like:
    # a=`lambda x : 42` and k=''
    # (the type vars preclude such a counterexample)
    # Note also a related issue: https://github.com/pschanely/CrossHair/issues/85
    T = TypeVar("T")

    def f(a: Callable[[T], T], k: T) -> T:
        """post: _ != 42"""
        if isinstance(k, int):
            return 0  # type: ignore
        return a(k)

    check_states(f, CANNOT_CONFIRM)  # or maybe CONFIRMED?


def test_hash() -> None:
    def f(s: int) -> int:
        """post: True"""
        return hash(s)

    check_states(f, CONFIRMED)


@pytest.mark.demo
def test_getattr() -> None:
    def f(s: str) -> int:
        """
        Can this function return 42?

        post: _ != 42
        """

        class Otter:
            def do_things(self) -> int:
                return 42

        try:
            return getattr(Otter(), s)()
        except Exception:
            return 0

    check_states(f, POST_FAIL)


def TODO_test_print_ok() -> None:
    def f(x: int) -> bool:
        """
        post: _ == True
        """
        print(x)
        return True

    check_states(f, CONFIRMED)


def test_repr_ok():
    def f(x: int) -> str:
        """post: len(_) == 0 or len(_) > 0"""
        return repr(x)

    check_states(f, CONFIRMED)


@pytest.mark.demo
def test_map() -> None:
    def f(ls: List[int]) -> List[int]:
        """
        Can an incremented list equal [4, 9, 0]?

        post: _ != [4, 9, 0]
        """
        return list(map(lambda x: x + 1, ls))

    check_states(f, POST_FAIL)


def test_max_fail() -> None:
    def f(ls: List[int]) -> int:
        """
        post: _ in ls
        """
        return max(ls)

    check_states(f, EXEC_ERR)


def test_max_ok() -> None:
    def f(ls: List[int]) -> int:
        """
        pre: bool(ls)
        post[]: _ in ls
        """
        return max(ls)

    check_states(f, CANNOT_CONFIRM)


def test_min_ok() -> None:
    def f(ls: List[float]) -> float:
        """
        pre: bool(ls)
        post[]: _ in ls
        """
        return min(ls)

    check_states(f, CANNOT_CONFIRM)


def test_list_index_on_concrete() -> None:
    def f(i: int) -> int:
        """post: True"""
        return [0, 1, 2].index(i)

    (actual, expected) = check_exec_err(f, "ValueError:")
    assert actual == expected


def test_eval_namespaces() -> None:
    def f(i: int) -> int:
        """post: _ == i + 1"""
        return eval("i + Color.BLUE.value")

    check_states(f, CONFIRMED)


def test_bytes_specific_length() -> None:
    def f(b: bytes) -> int:
        """post: _ != 5"""
        return len(b)

    check_states(f, POST_FAIL)


def test_bytes_out_of_range_byte() -> None:
    def f(b: bytes) -> bytes:
        """
        pre: len(b) == 1
        post: _[0] != 256
        """
        return b

    check_states(f, CONFIRMED)


def test_bytes_roundtrip_array_as_symbolic():
    with standalone_statespace as space:
        orig_bytes = proxy_for_type(bytes, "origbytes")
        as_array = bytearray(orig_bytes)
        new_bytes = bytes(as_array)
        with NoTracing():
            assert type(as_array) is SymbolicByteArray
            assert type(new_bytes) is SymbolicBytes
            assert new_bytes.inner is orig_bytes.inner


@pytest.mark.demo
def test_bytes_decode_method():
    def f(b: bytes) -> str:
        """
        Does any 2-byte sequence represent the character "ε" in UTF-8?

        NOTE: The process of decoding involves a lot of branching;
        most problems will require minutes of processing or more.

        pre: len(b) == 2
        post: _ != "ε"
        raises: UnicodeDecodeError
        """
        return b.decode("utf8")

    check_states(f, POST_FAIL)


@pytest.mark.demo("red")
def test_bytes___str___method():
    def f(b: bytes):
        """
        Is the string form of any byte array equal to b''?

        NOTE: This conversion does not have symbolic support (yet). We are able to find
        the enpty string, but nearly any other bytes string cannot be found.

        post: _ != "b''"
        """
        return str(b)

    check_states(f, POST_FAIL)


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
    """post: _"""
    with standalone_statespace as space:
        val = proxy_for_type(int, "val")
        space.add(val.var == 254)
        mv = memoryview(bytearray([val]))
        assert mv.cast("b")[0] == -2


def test_memoryview_toreadonly():
    """post: _"""
    with standalone_statespace as space:
        mv = proxy_for_type(memoryview, "mv")
        space.add(mv.__len__().var == 1)
        mv2 = mv.toreadonly()
        mv[0] = 12
        assert mv2[0] == 12
        with pytest.raises(TypeError):
            mv2[0] = 24


def test_memoryview_properties():
    """post: _"""
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
    # (errors at the Z3 level presently)
    with standalone_statespace as space:
        x = proxy_for_type(int, "x")
        y = proxy_for_type(float, "y")
        modval = x % y
        with NoTracing():
            assert type(modval) == SymbolicFloat
            assert space.is_possible(modval.var == 12.12)


def TODO_test_deepcopy_independence():
    # (snapshotting should take care of this for us(?), but it doesn't seem to)
    with standalone_statespace as space:
        ls = proxy_for_type(List[List[int]], "ls")
        lscopy = copy.deepcopy(ls)
        with NoTracing():
            assert ls[0] is not lscopy[0]
        # Next try mutation on one and test the other...


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
