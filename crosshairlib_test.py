import unittest

from crosshairlib import *

class CrossHairLibTest(unittest.TestCase):

    def test_to_z3(self):
        env = Z3BindingEnv()
        ret = to_z3(exprparse('lambda x:x'), env)
        self.assertEqual('func(lambda_1_0)', str(ret))
        self.assertEqual('ForAll(x, .(func(lambda_1_0), a(_, x)) == x)', str(env.support[0]))

    def test_order_axioms(self):
        self.assertEqual([
            'filter(isint, (2, b, c))',
            '(2, 3, 4)',
            'foo(filter(isint, (2, b, c)))',
            '(2, x, 4)',
            'istuple(2, x, 4)',
        ], list(map(unparse, order_axioms([
                    exprparse('(2, x, 4)'),
                    exprparse('istuple(2, x, 4)'),
                    exprparse('(2, 3, 4)'),
                    exprparse('filter(isint, (2, b, c))'),
                    exprparse('foo(filter(isint, (2, b, c)))'),
                ],
                exprparse('filter(isint, (2, 3, 4))')
            ))))

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
    def test_assertion_true_or_false_or_undef(self):
        self.prove('def p(x:isdefined): return x or not x')
        # self.prove('lambda x : isdefined(x or (not x) or (not isdefined(x)))')

    def test_assertion_0_is_falsey(self):
        self.prove('lambda : not 0')
    def test_assertion_int_is_truthy(self):
        self.prove('lambda : 4')

    def test_assertion_int_compare(self):
        # self.prove('lambda : isint(4) and 4 < 7')
        # self.prove('lambda : (lambda x:x)')
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

    def test_assertion_with_lambda1(self):
        self.prove('lambda : (lambda x:x)(7) == 7')
    def test_assertion_with_lambda2(self):
        self.prove('def p(): return isfunc(isint)')
    def test_assertion_with_lambda3(self):
        self.prove('def p(): return isdefined(forall(lambda x,y:False))')

    def test_assertion_with_tuples1(self):
        self.prove('def p(): return (1,2) == (*(1,), 2)')
    def test_assertion_with_tuples2(self):
        self.prove('def p(): return (1,2,2) == (1, *(2,), 2)')
    def test_assertion_with_tuples3(self):
        self.prove('lambda : istuple((1, *(2,3)))')

    def test_assertion_with_len1(self):
        self.prove('lambda x, y : len(()) == 0')
    def test_assertion_with_len2(self):
        self.prove('lambda : len((1,)) == 1')
    def test_assertion_with_len3(self):
        self.prove('lambda : len((*(1,), 2)) == 2')
    def test_assertion_with_len4(self):
        self.prove('lambda : len((2, *(1,))) == 2')
    def test_assertion_with_len5(self):
        self.prove('lambda : isdefined(len((1,3)))')
    def test_assertion_with_len6(self):
        self.prove('lambda : len((1,3)) == 2')
    def test_assertion_with_len7(self):
        self.prove('lambda : len((1,*(2,3,4),5)) == 5')
    def test_assertion_with_len8(self):
        self.prove('def p(t:istuple) : return len(t) < len((*t,1))')

    def test_assertion_with_all1(self):
        self.prove('def p(): return all(())')
    def test_assertion_with_all2(self):
        self.prove('def p(t:istuple): return implies(all(t), all((*t, True)))')
    # # I think this requires an inductive proof?
    # def test_assertion_with_all3(self):
    #     self.prove('def p(t:istuple): return implies(all(t), all((True, *t)))')

    def test_assertion_with_map1(self):
        self.prove('def p(t:istuple): return map(isint, ()) == ()')
    def test_assertion_with_map2(self):
        self.prove('def p(t:istuple): return map(isint, (2,3)) == (True, True)')
    def test_assertion_with_map3(self):
        self.prove('def p(t:istuple): return all(map(isint, (2, 3)))')
    def test_assertion_with_map4(self):
        self.prove('def p(x): return isdefined(range(5))')
        # self.prove('def p(x): return implies(
        # self.prove('def p(t:istuple): return isdefined(all(map(isint, (2, 3))))')
        # self.prove('def p(t:istuple): return implies(all(map(isnat, t)), all(map(isint, t)))')
    def test_assertion_with_map5(self):
        # self.prove('def p(x): return implies(isnat(x), isint(x))')
        # self.prove('def p(): return isdefined(forall(lambda x:x and False))')
        # self.prove('def p(): return isdefined(forall(lambda x:implies(isnat(x), isint(x))))')
        # self.prove('def p(): return all(map(isint, range(5)))')
        # self.prove('def p(t:istuple): return implies(all(map(isnat, t)), all(map(isint, t)))')
        #                                      implies(all(map(isnat, t)), all(map(isint, t)))
        self.prove('def p(x:isint): return all(map(isint, range(x)))')
    def test_assertion_with_map6(self):
        self.prove('def p(x:isint): return range(1) == (0,)')
        # self.prove('def p(x:isint): return len(range(1)) == 1')


if __name__ == '__main__':
    unittest.main()
