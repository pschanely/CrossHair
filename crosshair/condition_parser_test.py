import unittest
from typing import cast, Generic, Optional, List, TypeVar

from crosshair.condition_parser import *


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


def with_invalid_type_annotation(x: 'TypeThatIsNotDefined'):
    pass

class BaseClassExample:
    '''
    inv: True
    '''


class SubClassExample(BaseClassExample):
    def foo(self) -> int:
        '''
        post: False
        '''
        return 5


class ConditionParserTest(unittest.TestCase):

    def test_class_parse(self) -> None:
        class_conditions = get_class_conditions(Foo)
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
        conditions = get_fn_conditions(single_line_condition)
        assert conditions is not None
        self.assertEqual(set([c.expr_source for c in conditions.post]),
                         set(['__return__ >= x']))

    def test_implies_condition(self) -> None:
        conditions = get_fn_conditions(implies_condition)
        assert conditions is not None
        # This shouldn't explode (avoid a KeyError on record['override']):
        conditions.post[0].evaluate({'record': {}, '_': 0})

    def test_raises_condition(self) -> None:
        conditions = get_fn_conditions(raises_condition)
        assert conditions is not None
        self.assertEqual([], list(conditions.syntax_messages()))
        self.assertEqual(set([KeyError, OSError]), conditions.raises)

    def test_invariant_is_inherited(self) -> None:
        class_conditions = get_class_conditions(SubClassExample)
        self.assertEqual(set(class_conditions.methods.keys()), set(['foo']))
        method = class_conditions.methods['foo']
        self.assertEqual(set([c.expr_source for c in method.pre]),
                         set(['True']))
        self.assertEqual(set([c.expr_source for c in method.post]),
                         set(['True', 'False']))

    def test_builtin_conditions_are_null(self) -> None:
        self.assertIsNone(get_fn_conditions(zip))

    def test_fn_globals_on_builtin(self) -> None:
        self.assertIs(fn_globals(zip), builtins.__dict__)

    def test_resolve_signature_invalid_annotations(self) -> None:
        sig, err = resolve_signature(with_invalid_type_annotation)
        self.assertIsNotNone(err)
        self.assertEqual("name 'TypeThatIsNotDefined' is not defined", err.message)
        self.assertIsNotNone(sig)

    def test_resolve_signature_c_function(self) -> None:
        sig, err = resolve_signature(map)
        self.assertIsNone(sig)
        self.assertIsNone(err)

    def test_conditions_with_closure_references_and_string_type(self) -> None:
        # This is a function that refers to something in its closure.
        # Ensure we can still look up string-based types:
        def referenced_fn():
            return 4
        def fn_with_closure(foo: "Foo"):
            referenced_fn()
        # Ensure we don't error trying to resolve "Foo":
        get_fn_conditions(fn_with_closure)

    def test_empty_vs_missing_mutations(self) -> None:
        self.assertIsNone(parse_sections([(1,'post: True')], ('post',), '').mutable_expr)
        self.assertEqual('', parse_sections([(1,'post[]: True')], ('post',), '').mutable_expr)

    def test_set_first_arg_type(self) -> None:
        sig = inspect.signature(Foo.isready)
        typed_sig = set_first_arg_type(sig, Foo)
        self.assertEqual(typed_sig.parameters['self'].annotation, Foo)

if __name__ == '__main__':
    unittest.main()
