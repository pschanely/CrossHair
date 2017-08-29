import unittest

from crosshairlib import *

class PurelibTest(unittest.TestCase):

    def _assertion(self, assertion):
        definition = astparse(assertion)
        if type(definition) == ast.Expr:
            definition = definition.value
        return check_assertion_fn(definition)

    def prove(self, assertion):
        self.assertTrue(self._assertion(assertion))
    def do_NOT_prove(self, assertion):
        self.assertFalse(self._assertion(assertion))


    def test_assertion_true(self):
        self.prove('lambda : True')
    def test_assertion_false(self):
        self.do_NOT_prove('lambda : False')
    def test_assertion_not_false(self):
        self.prove('lambda : not False')
    def test_assertion_defined_not(self):
        self.prove('def p(x:isdefined): return isdefined(not x)')

    def test_assertion_0_is_falsey(self):
        self.prove('lambda : not 0')
    def test_assertion_int_is_truthy(self):
        self.prove('lambda : 4')

    def test_assertion_int_compare(self):
        self.prove('lambda : 4 < 7')
    def test_assertion_int_compare_to_bool(self):
        self.prove('lambda : 4 != True')
    def test_assertion_int_compare_and_conjunction(self):
        self.prove('lambda : 4 < 7 and 7 >= 7')

    def test_assertion_implication1(self):
        self.prove('lambda x : implies(False, False)')
    def test_assertion_implication2(self):
        self.prove('lambda x : implies(x, x != 0)')
    def test_assertion_implication3(self):
        self.do_NOT_prove('lambda x : implies(x != 0, x)')

    def test_assertion_isint1(self):
        self.prove('lambda : isint(4)')

    def test_assertion_isdefined1(self):
        self.prove('lambda : isdefined(isbool(7))')

    def test_assertion_isbool1(self):
        self.prove('lambda : isbool(True)')
    def test_assertion_isbool2(self):
        self.prove('lambda : isbool(False)')
    def test_assertion_isbool3(self):
        self.prove('lambda : not (not True)')
    def test_assertion_isbool4(self):
        self.prove('lambda : isdefined(isbool(7))')
    def test_assertion_isbool5(self):
        self.prove('lambda : not isbool(7)')
    def test_assertion_isbool6(self):
        self.prove('def p(x:isdefined, y:isdefined): return implies(not (x and y), (not x) or (not y))')

    def test_assertion_one_plus_one_is_not_zero(self):
        self.prove('lambda : 1 + 1 != 0')

    def test_assertion_adding_symmetry(self):
        self.prove('def p(x :isint, y :isint): return x + y == y + x')
    def test_assertion_adding_increases1(self):
        self.do_NOT_prove('lambda x : x + 1 > x')
        # ... because if x is None, for example, the result is undefined
    def test_assertion_adding_increases2(self):
        self.prove('lambda x : implies(isint(x), x + 1 > x)')
    def test_assertion_adding_increases3(self):
        self.prove('def p(x : isint): return x + 1 > x')
    def test_assertion_adding_increases4(self):
        self.do_NOT_prove('def p(x : isbool): return x + 1 > x')

    def test_assertion_with_lambda(self):
        self.prove('lambda : (lambda x:x)(7) == 7')

    def test_assertion_with_tuples1(self):
        self.prove('lambda x, y : len(()) == 0')
    def test_assertion_with_tuples2(self):
        self.prove('def p(): return (1,2,2) == (1, *(2,), 2)')
    def test_assertion_with_tuples3(self):
        self.prove('lambda : istuple((1, *(2,3)))')
    def test_assertion_with_tuples4(self):
        self.prove('lambda : len((1, 2)) == 2')
    def test_assertion_with_tuples5(self):
        self.prove('lambda : len((1,*(2,3,4),5)) == 5')
    def test_assertion_with_tuples6(self):
        self.prove('def p(t:istuple) : return len(t) < len((*t,1))')

    def test_assertion_with_all1(self):
        self.prove('def p(): return all(())')
    def test_assertion_with_all2(self):
        self.prove('def p(t:istuple): return implies(all(t), all((*t, True)))')
    # I think this requires an inductive proof?
    # def test_assertion_with_all3(self):
    #     self.prove('def p(t:istuple): return implies(all(t), all((True, *t)))')

    # def test_assertion_with_map1(self):
    #     self.prove('def p(t:istuple): return implies(all(map(isnat, t)), all(map(isint, t)))')
    #     # self.prove('def p(x:isint): return all(map(isint, range(x)))')


if __name__ == '__main__':
    unittest.main()
