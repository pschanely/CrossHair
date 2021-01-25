import unittest
from typing import cast, Generic, Optional, List, TypeVar

from crosshair.condition_parser import *
from crosshair.fnutil import FunctionInfo
from crosshair.util import set_debug
from crosshair.util import debug
from crosshair.util import AttributeHolder


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
    ''' post: __return__ >= x '''
    return x


def implies_condition(record: dict) -> object:
    ''' post: implies('override' in record, _ == record['override']) '''
    return record['override'] if 'override' in record else 42


def raises_condition(record: dict) -> object:
    '''
    raises: KeyError, OSError # comma , then junk
    '''
    raise KeyError('')


class BaseClassExample:
    '''
    inv: True
    '''
    def foo(self) -> int:
        return 4


class SubClassExample(BaseClassExample):
    def foo(self) -> int:
        '''
        post: False
        '''
        return 5


class Pep316ParserTest(unittest.TestCase):

    def test_class_parse(self) -> None:
        class_conditions = Pep316Parser().get_class_conditions(Foo)
        self.assertEqual(set([c.expr_source for c in class_conditions.inv]),
                         set(['self.x >= 0', 'self.y >= 0']))
        self.assertEqual(set(class_conditions.methods.keys()),
                         set(['isready']))
        method = class_conditions.methods['isready']
        self.assertEqual(set([c.expr_source for c in method.pre]),
                         set(['self.x >= 0', 'self.y >= 0']))
        self.assertEqual(set([c.expr_source for c in method.post]),
                         set(['__return__ == (self.x == 0)', 'self.x >= 0', 'self.y >= 0']))

    def test_single_line_condition(self) -> None:
        conditions = Pep316Parser().get_fn_conditions(
            FunctionInfo.from_fn(single_line_condition))
        assert conditions is not None
        self.assertEqual(set([c.expr_source for c in conditions.post]),
                         set(['__return__ >= x']))

    def test_implies_condition(self) -> None:
        conditions = Pep316Parser().get_fn_conditions(
            FunctionInfo.from_fn(implies_condition))
        assert conditions is not None
        # This shouldn't explode (avoid a KeyError on record['override']):
        conditions.post[0].evaluate({'record': {}, '_': 0})

    def test_raises_condition(self) -> None:
        conditions = Pep316Parser().get_fn_conditions(
            FunctionInfo.from_fn(raises_condition))
        assert conditions is not None
        self.assertEqual([], list(conditions.syntax_messages()))
        self.assertEqual(set([KeyError, OSError]), conditions.raises)

    def test_invariant_is_inherited(self) -> None:
        class_conditions = Pep316Parser().get_class_conditions(SubClassExample)
        self.assertEqual(set(class_conditions.methods.keys()), set(['foo']))
        method = class_conditions.methods['foo']
        self.assertEqual(len(method.pre), 1)
        self.assertEqual(set([c.expr_source for c in method.pre]),
                         set(['True']))
        self.assertEqual(len(method.post), 2)
        self.assertEqual(set([c.expr_source for c in method.post]),
                         set(['True', 'False']))

    def test_builtin_conditions_are_null(self) -> None:
        self.assertIsNone(Pep316Parser().get_fn_conditions(
            FunctionInfo.from_fn(zip)))

    def test_conditions_with_closure_references_and_string_type(self) -> None:
        # This is a function that refers to something in its closure.
        # Ensure we can still look up string-based types:
        def referenced_fn():
            return 4
        def fn_with_closure(foo: "Foo"):
            referenced_fn()
        # Ensure we don't error trying to resolve "Foo":
        Pep316Parser().get_fn_conditions(
            FunctionInfo.from_fn(fn_with_closure))

    def test_empty_vs_missing_mutations(self) -> None:
        self.assertIsNone(parse_sections([(1,'post: True')], ('post',), '').mutable_expr)
        self.assertEqual('', parse_sections([(1,'post[]: True')], ('post',), '').mutable_expr)


try:
    import icontract
except:
    icontract = None  # type: ignore

if icontract:
    class IcontractParserTest(unittest.TestCase):

        def test_simple_parse(self) -> None:
            @icontract.require(lambda l: len(l) > 0)
            @icontract.ensure(lambda l, result: min(l) <= result <= max(l))
            def avg(l):
                return sum(l) / len(l)

            conditions = IcontractParser().get_fn_conditions(
                FunctionInfo.from_fn(avg))
            assert conditions is not None
            self.assertEqual(len(conditions.pre), 1)
            self.assertEqual(len(conditions.post), 1)
            self.assertEqual(conditions.pre[0].evaluate({'l': []}), False)
            self.assertEqual(conditions.post[0].evaluate(
                {'l': [42,43],
                 '__old__': AttributeHolder({}),
                 '__return__': 40,
                 '_': 40}), False)

        def test_simple_class_parse(self) -> None:
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

            decr_conditions = conditions.methods['decr']
            self.assertEqual(len(decr_conditions.pre), 2)
            # decr() precondition: count > 0
            self.assertEqual(decr_conditions.pre[0].evaluate({'self': Counter()}), False)
            # invariant: count >= 0
            self.assertEqual(decr_conditions.pre[1].evaluate({'self': Counter()}), True)

            class TruncatedCounter(Counter):
                @icontract.require(lambda self: self.count() == 0) # super already allows count > 0
                def decr(self):
                    if self.i > 0:
                        self.i -= 1
            conditions = IcontractParser().get_class_conditions(TruncatedCounter)
            decr_conditions = conditions.methods['decr']
            self.assertEqual(decr_conditions.pre[0].evaluate({'self': TruncatedCounter()}), True)

            # check the weakened precondition
            self.assertEqual(len(decr_conditions.pre), 2) # one for the invariant, one for the disjunction
            ctr = TruncatedCounter()
            ctr.i = 1
            self.assertEqual(decr_conditions.pre[1].evaluate({'self': ctr}), True)
            self.assertEqual(decr_conditions.pre[0].evaluate({'self': ctr}), True)
            ctr.i = 0
            self.assertEqual(decr_conditions.pre[1].evaluate({'self': ctr}), True)
            self.assertEqual(decr_conditions.pre[0].evaluate({'self': ctr}), True)

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
  ''' Removes the smallest number in the given list. '''
  # The precondition: CrossHair will assume this to be true:
  assert len(numbers) > 0
  smallest = min(numbers)
  numbers.remove(smallest)
  # The postcondition: CrossHair will find examples to make this be false:
  assert min(numbers) > smallest

class AssertsParserTest(unittest.TestCase):
    def tests_simple_parse(self) -> None:
        conditions = AssertsParser().get_fn_conditions(
            FunctionInfo.from_fn(avg_with_asserts))
        assert conditions is not None
        conditions.fn([])
        self.assertEqual(conditions.fn([2.2]), 2.2)
        with self.assertRaises(AssertionError):
            conditions.fn([9.2, 17.8])

    def tests_empty_parse(self) -> None:
        conditions = AssertsParser().get_fn_conditions(
            FunctionInfo.from_fn(debug))
        self.assertEqual(conditions, None)

    def tests_extra_ast_nodes(self) -> None:
        conditions = AssertsParser().get_fn_conditions(
            FunctionInfo.from_fn(fn_with_docstring_comments_and_assert))
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


if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
