import inspect
import json
import sys
import textwrap
import unittest
from typing import List

import pytest

from crosshair.condition_parser import (
    AssertsParser,
    CompositeConditionParser,
    DealParser,
    HypothesisParser,
    IcontractParser,
    Pep316Parser,
    parse_sections,
    parse_sphinx_raises,
)
from crosshair.fnutil import FunctionInfo
from crosshair.util import AttributeHolder, debug

try:
    import icontract  # type: ignore
except ImportError:
    icontract = None  # type: ignore

try:
    import deal  # type: ignore
except ImportError:
    deal = None  # type: ignore

try:
    import hypothesis  # type: ignore
except ImportError:
    hypothesis = None  # type: ignore


class LocallyDefiendException(Exception):
    pass


class Foo:
    """A thingy.

    Examples::
        >>> 'blah'
        'blah'

    inv:: self.x >= 0

    inv:
        # a blank line with no indent is ok:

        self.y >= 0
    notasection:
        self.z >= 0
    """

    x: int

    def isready(self) -> bool:
        """
        Checks for readiness

        post[]::
            __return__ == (self.x == 0)
        """
        return self.x == 0


def single_line_condition(x: int) -> int:
    """post: __return__ >= x"""
    return x


def implies_condition(record: dict) -> object:
    """post: implies('override' in record, _ == record['override'])"""
    return record["override"] if "override" in record else 42


def locally_defined_raises_condition(record: dict) -> object:
    """
    raises: LocallyDefiendException
    """
    raise KeyError("")


def tricky_raises_condition(record: dict) -> object:
    """
    raises: KeyError, json.JSONDecodeError # comma , then junk
    """
    raise KeyError("")


def sphinx_raises(record: dict) -> object:
    """
    Do things.
    :raises LocallyDefiendException: when blah
    """
    raise LocallyDefiendException("")


class BaseClassExample:
    """
    inv: True
    """

    def foo(self) -> int:
        return 4


class SubClassExample(BaseClassExample):
    def foo(self) -> int:
        """
        post: False
        """
        return 5


def test_parse_sections_variants() -> None:
    parsed = parse_sections([(1, " :post: True ")], ("post",), "")
    assert set(parsed.sections.keys()) == {"post"}
    parsed = parse_sections([(1, "post::True")], ("post",), "")
    assert set(parsed.sections.keys()) == {"post"}
    parsed = parse_sections([(1, ":post True")], ("post",), "")
    assert set(parsed.sections.keys()) == set()


def test_parse_sections_empty_vs_missing_mutations() -> None:
    mutations = parse_sections([(1, "post: True")], ("post",), "").mutable_expr
    assert mutations is None
    mutations = parse_sections([(1, "post[]: True")], ("post",), "").mutable_expr
    assert mutations == ""


def test_parse_sphinx_raises() -> None:
    assert parse_sphinx_raises(sphinx_raises) == {LocallyDefiendException}


class Pep316ParserTest(unittest.TestCase):
    def test_class_parse(self) -> None:
        class_conditions = Pep316Parser().get_class_conditions(Foo)
        self.assertEqual(
            set([c.expr_source for c in class_conditions.inv]),
            set(["self.x >= 0", "self.y >= 0"]),
        )
        self.assertEqual(
            set(class_conditions.methods.keys()), set(["isready", "__init__"])
        )
        method = class_conditions.methods["isready"]
        self.assertEqual(
            set([c.expr_source for c in method.pre]),
            set(["self.x >= 0", "self.y >= 0"]),
        )
        startlineno = inspect.getsourcelines(Foo)[1]
        self.assertEqual(
            set([(c.expr_source, c.line) for c in method.post]),
            set(
                [
                    ("self.x >= 0", startlineno + 7),
                    ("self.y >= 0", startlineno + 12),
                    ("__return__ == (self.x == 0)", startlineno + 24),
                ]
            ),
        )

    def test_single_line_condition(self) -> None:
        conditions = Pep316Parser().get_fn_conditions(
            FunctionInfo.from_fn(single_line_condition)
        )
        assert conditions is not None
        self.assertEqual(
            set([c.expr_source for c in conditions.post]), set(["__return__ >= x"])
        )

    def test_implies_condition(self):
        conditions = Pep316Parser().get_fn_conditions(
            FunctionInfo.from_fn(implies_condition)
        )
        assert conditions is not None
        # This shouldn't explode (avoid a KeyError on record['override']):
        conditions.post[0].evaluate({"record": {}, "_": 0})

    def test_locally_defined_raises_condition(self) -> None:
        conditions = Pep316Parser().get_fn_conditions(
            FunctionInfo.from_fn(locally_defined_raises_condition)
        )
        assert conditions is not None
        self.assertEqual([], list(conditions.syntax_messages()))
        self.assertEqual(set([LocallyDefiendException]), conditions.raises)

    def test_tricky_raises_condition(self) -> None:
        conditions = Pep316Parser().get_fn_conditions(
            FunctionInfo.from_fn(tricky_raises_condition)
        )
        assert conditions is not None
        self.assertEqual([], list(conditions.syntax_messages()))
        self.assertEqual(conditions.raises, set([KeyError, json.JSONDecodeError]))

    def test_invariant_is_inherited(self) -> None:
        class_conditions = Pep316Parser().get_class_conditions(SubClassExample)
        self.assertEqual(set(class_conditions.methods.keys()), set(["foo", "__init__"]))
        method = class_conditions.methods["foo"]
        self.assertEqual(len(method.pre), 1)
        self.assertEqual(set([c.expr_source for c in method.pre]), set(["True"]))
        self.assertEqual(len(method.post), 2)
        self.assertEqual(
            set([c.expr_source for c in method.post]), set(["True", "False"])
        )

    def test_invariant_applies_to_init(self) -> None:
        class_conditions = Pep316Parser().get_class_conditions(BaseClassExample)
        self.assertEqual(set(class_conditions.methods.keys()), set(["__init__", "foo"]))

    def test_builtin_conditions_are_null(self) -> None:
        self.assertIsNone(Pep316Parser().get_fn_conditions(FunctionInfo.from_fn(zip)))

    def test_conditions_with_closure_references_and_string_type(self) -> None:
        # This is a function that refers to something in its closure.
        # Ensure we can still look up string-based types:
        def referenced_fn():
            return 4

        def fn_with_closure(foo: "Foo"):
            referenced_fn()

        # Ensure we don't error trying to resolve "Foo":
        Pep316Parser().get_fn_conditions(FunctionInfo.from_fn(fn_with_closure))


@pytest.mark.skipif(not icontract, reason="icontract is not installed")
class IcontractParserTest(unittest.TestCase):
    def test_simple_parse(self):
        @icontract.require(lambda l: len(l) > 0)
        @icontract.ensure(lambda l, result: min(l) <= result <= max(l))
        def avg(ls):
            return sum(ls) / len(ls)

        conditions = IcontractParser().get_fn_conditions(FunctionInfo.from_fn(avg))
        assert conditions is not None
        self.assertEqual(len(conditions.pre), 1)
        self.assertEqual(len(conditions.post), 1)
        self.assertEqual(conditions.pre[0].evaluate({"l": []}), False)
        post_args = {
            "l": [42, 43],
            "__old__": AttributeHolder({}),
            "__return__": 40,
            "_": 40,
        }
        self.assertEqual(conditions.post[0].evaluate(post_args), False)
        self.assertEqual(len(post_args), 4)  # (check args are unmodified)

    def test_simple_class_parse(self):
        @icontract.invariant(lambda self: self.i >= 0)
        class Counter(icontract.DBC):
            def __init__(self):
                self.i = 0

            @icontract.ensure(lambda self, result: result >= 0)
            def count(self) -> int:
                return self.i

            @icontract.ensure(lambda self: self.count() > 0)
            def incr(self):
                self.i += 1

            @icontract.require(lambda self: self.count() > 0)
            def decr(self):
                self.i -= 1

        conditions = IcontractParser().get_class_conditions(Counter)
        self.assertEqual(len(conditions.inv), 1)

        decr_conditions = conditions.methods["decr"]
        self.assertEqual(len(decr_conditions.pre), 2)
        # decr() precondition: count > 0
        self.assertEqual(decr_conditions.pre[0].evaluate({"self": Counter()}), False)
        # invariant: count >= 0
        self.assertEqual(decr_conditions.pre[1].evaluate({"self": Counter()}), True)

        class TruncatedCounter(Counter):
            @icontract.require(
                lambda self: self.count() == 0
            )  # super already allows count > 0
            def decr(self):
                if self.i > 0:
                    self.i -= 1

        conditions = IcontractParser().get_class_conditions(TruncatedCounter)
        decr_conditions = conditions.methods["decr"]
        self.assertEqual(
            decr_conditions.pre[0].evaluate({"self": TruncatedCounter()}), True
        )

        # check the weakened precondition
        self.assertEqual(
            len(decr_conditions.pre), 2
        )  # one for the invariant, one for the disjunction
        ctr = TruncatedCounter()
        ctr.i = 1
        self.assertEqual(decr_conditions.pre[1].evaluate({"self": ctr}), True)
        self.assertEqual(decr_conditions.pre[0].evaluate({"self": ctr}), True)
        ctr.i = 0
        self.assertEqual(decr_conditions.pre[1].evaluate({"self": ctr}), True)
        self.assertEqual(decr_conditions.pre[0].evaluate({"self": ctr}), True)


@pytest.mark.skipif(not deal, reason="deal is not installed")
def test_deal_basics():
    @deal.raises(ZeroDivisionError)
    @deal.pre(lambda a, b: a >= 0 and b >= 0)
    @deal.ensure(lambda a, b, result: result <= a)
    def f(a: int, b: int) -> float:
        return a / b

    conditions = DealParser().get_fn_conditions(FunctionInfo.from_fn(f))
    (pre,) = conditions.pre
    (post,) = conditions.post

    assert conditions.fn(12, b=6) == 2.0
    assert conditions.raises == {ZeroDivisionError}
    assert pre.evaluate({"a": -2, "b": 3}) == False  # noqa: E712
    assert pre.evaluate({"a": 2, "b": 3}) == True  # noqa: E712
    post_args = {
        "a": 6,
        "b": 2,
        "__old__": AttributeHolder({}),
        "_": 3.0,
        "__return__": 3.0,
    }
    assert post.evaluate(post_args) == True  # noqa: E712


@pytest.mark.skipif(not deal, reason="deal is not installed")
def test_deal_postcondition():
    @deal.raises(ZeroDivisionError)
    @deal.post(lambda r: r >= 0)
    def f(a: int, b: int) -> float:
        return a / b

    conditions = DealParser().get_fn_conditions(FunctionInfo.from_fn(f))
    (post,) = conditions.post

    post_args = {
        "a": 6,
        "b": 2,
        "__old__": AttributeHolder({}),
        "_": 3.0,
        "__return__": 3.0,
    }
    assert post.evaluate(post_args) == True  # noqa: E712
    post_args["__return__"] = -1.0
    assert post.evaluate(post_args) == False  # noqa: E712


@pytest.mark.skipif(not deal, reason="deal is not installed")
def test_deal_ensure_with_magic_single_arg():
    @deal.ensure(lambda _: _.result == 0 or _["item"] in _["items"])
    @deal.pure
    def count(items: List[str], item: str) -> int:
        return items.count(item)

    conditions = DealParser().get_fn_conditions(FunctionInfo.from_fn(count))
    (post,) = conditions.post
    post_args = {
        "item": "a",
        "items": ["b", "c"],
        "__old__": AttributeHolder({}),
        "_": 1,
        "__return__": 1,
    }
    assert post.evaluate(post_args) == False  # noqa: E712


def avg_with_asserts(items: List[float]) -> float:
    assert items
    avgval = sum(items) / len(items)
    assert avgval <= 10
    return avgval


def no_leading_assert(x: int) -> int:
    x = x + 1
    assert x != 100
    x = x + 1
    return x


def fn_with_docstring_comments_and_assert(numbers: List[int]) -> None:
    """Removes the smallest number in the given list."""
    # The precondition: CrossHair will assume this to be true:
    assert len(numbers) > 0
    smallest = min(numbers)
    numbers.remove(smallest)
    # The postcondition: CrossHair will find examples to make this be false:
    assert min(numbers) > smallest


class AssertsParserTest(unittest.TestCase):
    def tests_simple_parse(self) -> None:
        conditions = AssertsParser().get_fn_conditions(
            FunctionInfo.from_fn(avg_with_asserts)
        )
        assert conditions is not None
        conditions.fn([])
        self.assertEqual(conditions.fn([2.2]), 2.2)
        with self.assertRaises(AssertionError):
            conditions.fn([9.2, 17.8])

    def tests_empty_parse(self) -> None:
        conditions = AssertsParser().get_fn_conditions(FunctionInfo.from_fn(debug))
        self.assertEqual(conditions, None)

    def tests_extra_ast_nodes(self) -> None:
        conditions = AssertsParser().get_fn_conditions(
            FunctionInfo.from_fn(fn_with_docstring_comments_and_assert)
        )
        assert conditions is not None

        # Empty list does not pass precondition, ignored:
        conditions.fn([])

        # normal, passing case:
        nums = [3, 1, 2]
        conditions.fn(nums)
        self.assertEqual(nums, [3, 2])

        # Failing case (duplicate minimum values):
        with self.assertRaises(AssertionError):
            nums = [3, 1, 1, 2]
            conditions.fn(nums)


def test_CompositeConditionParser():
    composite = CompositeConditionParser()
    composite.parsers.append(Pep316Parser(composite))
    composite.parsers.append(AssertsParser(composite))
    assert composite.get_fn_conditions(
        FunctionInfo.from_fn(single_line_condition)
    ).has_any()
    assert composite.get_fn_conditions(FunctionInfo.from_fn(avg_with_asserts)).has_any()


def no_postconditions(items: List[float]) -> float:
    """pre: items"""
    return sum(items) / len(items)


def test_adds_completion_postconditions():
    pep316_parser = Pep316Parser()
    fn = FunctionInfo.from_fn(no_postconditions)
    assert len(pep316_parser.get_fn_conditions(fn).post) == 1


def test_raw_docstring():
    def linelen(s: str) -> int:
        r"""
        pre: '\n' not in s
        """
        return len(s)

    conditions = Pep316Parser().get_fn_conditions(FunctionInfo.from_fn(linelen))
    assert len(conditions.pre) == 1
    assert conditions.pre[0].expr_source == r"'\n' not in s"


def test_regular_docstrings_parsed_like_raw():
    def linelen(s: str) -> int:
        """pre: '\n' not in s"""
        return len(s)

    conditions = Pep316Parser().get_fn_conditions(FunctionInfo.from_fn(linelen))
    assert len(conditions.pre) == 1
    assert conditions.pre[0].expr_source == r"'\n' not in s"


def test_lines_with_trailing_comment():
    def foo():
        """
        post: True"""  # A trailing comment
        ...

    conditions = Pep316Parser().get_fn_conditions(FunctionInfo.from_fn(foo))
    assert len(conditions.post) == 1
    assert conditions.post[0].expr_source == "True"


def test_format_counterexample_positional_only():
    if sys.version_info >= (3, 8):
        # Use exec() here because the "/" marker is a syntax error in Python 3.7
        ns = {}
        foo = exec(
            textwrap.dedent(
                '''
                def foo(a=10, /, b=20):
                    """post: True"""
                '''
            ),
            ns,
        )
        foo = ns["foo"]
        args = inspect.BoundArguments(inspect.signature(foo), {"a": 1, "b": 2})
        conditions = Pep316Parser().get_fn_conditions(FunctionInfo.from_fn(foo))
        assert conditions.format_counterexample(args, None, {}) == (
            "foo(1, b = 2)",
            "None",
        )


def test_format_counterexample_keyword_only():
    def foo(a, *, b):
        """post: True"""

    args = inspect.BoundArguments(inspect.signature(foo), {"a": 1, "b": 2})
    conditions = Pep316Parser().get_fn_conditions(FunctionInfo.from_fn(foo))
    assert conditions.format_counterexample(args, None, {}) == ("foo(1, b = 2)", "None")


@pytest.mark.skipif(not hypothesis, reason="hypothesis is not installed")
def test_hypothesis_arg_regen():
    @hypothesis.given(hypothesis.strategies.integers())
    def hypothesis_fn_int(x):
        pass

    parser = HypothesisParser(None)
    # NOTE: Enocding not stable across hypothesis versions.
    # Byte string may need to be updated when our hypothesis dev version changes.
    ret = parser._generate_args(b"\x01\x04T", hypothesis_fn_int)
    assert ret == {"x": 42}
