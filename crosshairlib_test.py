import os.path
import time
import unittest

from crosshairlib import *
from crosshair import *
import prooforacle

# lambda 3 len 8 map 4 map 6

class CrossHairLibTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._oracle = None
        if os.path.exists('.crosshairlibtest.model'):
            cls._oracle = prooforacle.ProofOracle('.crosshairlibtest.model')

    @classmethod
    def tearDownClass(cls):
        if cls._oracle:
            cls._oracle.save_log()

    # def _assertion(self, assertion):
    def _assertion(self, fn):
        # definition = astparse(assertion)
        # if type(definition) == ast.Expr:
        #     definition = definition.value
        #     fn = eval(assertion)
        # else:
        #     g = {}
        #     exec(assertion, crosshair.__dict__, g)
        #     if len(g) != 1:
        #         raise Exception()
        #     fn = list(g.values())[0]
        t0 = time.time()
        ret, support = check_assertion_fn(fn, self._oracle)
        if self._oracle:
            self._oracle.add_to_log(prooforacle.ProofLog(
                conclusion=inspect.getsource(fn), support=support, proven=ret, tm=time.time()-t0, opts={}, kind='unittest'))
        return ret

    def test_to_z3(self):
        env = Z3BindingEnv()
        ret = to_z3(exprparse('lambda x:x'), env)
        self.assertEqual('func(lambda_1_0)', str(ret))
        self.assertEqual('ForAll(x, .(func(lambda_1_0), a(_, x)) == x)', str(env.support[0]))

    # def test_implication_equality_form(self):
    #     self.assertEqual(2, implication_equality_form(exprparse('a==4')))
    #     self.assertEqual(2, len(implication_equality_form(exprparse('implies(True, a==4)'))))
    #     self.assertEqual(1, len(implication_equality_form(exprparse('implies(True, a<4)'))))
    #     self.assertEqual(None, implication_equality_form(exprparse('iff(True, a==4)')))

    # def test_oracle_training(self):
    #     ret = prooforacle.train(prove_assertion_fn)
    #     print(ret)

    def prove(self, assertion):
        self.assertTrue(self._assertion(assertion))
    def do_NOT_prove(self, assertion):
        self.assertFalse(self._assertion(assertion))


    def test_assertion_true(self):
        def p(): return True
        self.prove(p)
    def test_assertion_false(self):
        def p(): return False
        self.do_NOT_prove(p)
    def test_assertion_not_false(self):
        def p(): return (not False)
        self.prove(p)
    def test_assertion_defined_not(self):
        def p(x:isdefined): return isdefined(not x)
        self.prove(p)
    def test_assertion_true_or_false_or_undef(self):
        def p(x:isdefined): return x or not x
        self.prove(p)

    def test_assertion_0_is_falsey(self):
        def p(): return (not 0)
        self.prove(p)
    def test_assertion_int_is_truthy(self):
        def p(): return (4)
        self.prove(p)

    def test_assertion_int_compare(self):
        # def p(): return (isint(4) and 4 < 7')
        # def p(): return ((lambda x:x)')
        def p(): return (4 < 7)
        self.prove(p)
    def test_assertion_int_compare_to_bool(self):
        def p(): return (4 != True)
        self.prove(p)
    def test_assertion_int_compare_and_conjunction(self):
        def p(): return (4 < 7 and 7 >= 7)
        self.prove(p)

    def test_assertion_implication1(self):
        def p(): return implies(False, False)
        self.prove(p)
    def test_assertion_implication2(self):
        def p(x): return implies(x, x != 0)
        self.prove(p)
    def test_assertion_implication3(self):
        def p(x): return implies(x != 0, x)
        self.do_NOT_prove(p)

    def test_assertion_isint1(self):
        def p(): return (isint(4))
        self.prove(p)

    def test_assertion_isdefined1(self):
        def p(): return (isdefined(isbool(7)))
        self.prove(p)

    def test_assertion_isbool1(self):
        def p(): return (isbool(True))
        self.prove(p)
    def test_assertion_isbool2(self):
        def p(): return (isbool(False))
        self.prove(p)
    def test_assertion_isbool3(self):
        def p(): return (not (not True))
        self.prove(p)
    def test_assertion_isbool4(self):
        def p(): return (isdefined(isbool(7)))
        self.prove(p)
    def test_assertion_isbool5(self):
        def p(): return (not isbool(7))
        self.prove(p)
    def test_assertion_isbool6(self):
        def p(x:isdefined, y:isdefined):
            return implies(not (x and y), (not x) or (not y))
        self.prove(p)

    def test_assertion_one_plus_one_is_not_zero(self):
        def p(): return (1 + 1 != 0)
        self.prove(p)
    def test_assertion_literal_subtraction(self):
        def p(): return 1 - 1 < 1
        self.prove(p)

    def test_assertion_adding_symmetry(self):
        def p(x :isint, y :isint): return x + y == y + x
        self.prove(p)
    def test_assertion_adding_increases1(self):
        def p(x): return x + 1 > x
        self.do_NOT_prove(p)
        # ... because if x is None, for example, the result is undefined
    def test_assertion_adding_increases2(self):
        def p(x): return implies(isint(x), x + 1 > x)
        self.prove(p)
    def test_assertion_adding_increases3(self):
        def p(x : isint): return x + 1 > x
        self.prove(p)
    def test_assertion_adding_increases4(self):
        def p(x : isbool): return x + 1 > x
        self.do_NOT_prove(p)

    def test_assertion_with_lambda1(self):
        def p(): return ((lambda x:x)(7) == 7)
        self.prove(p)
    def test_assertion_with_lambda2(self):
        def p(): return isfunc(isint)
        self.prove(p)
    # def test_assertion_with_lambda3(self):
    #     self.prove('def p(): return isdefined(forall(lambda x,y:False)))

    def test_assertion_with_tuples1(self):
        def p(): return (1,2) == (*(1,), 2)
        self.prove(p)
    def test_assertion_with_tuples2(self):
        def p(): return (1,2,2) == (1, *(2,), 2)
        self.prove(p)
    def test_assertion_with_tuples3(self):
        def p(): return istuple((1, *(2,3)))
        self.prove(p)

    def test_assertion_with_len1(self):
        def p(): return len(()) == 0
        self.prove(p)
    def test_assertion_with_len2(self):
        def p(): return len((1,)) == 1
        self.prove(p)
    def test_assertion_with_len3(self): #TODO soundness failure here
        def p(): return len((*(1,), 2)) == 2
        self.prove(p)
    def test_assertion_with_len4(self):
        def p(): return len((2, *(1,))) == 2
        self.prove(p)
    def test_assertion_with_len5(self):
        def p(): return isdefined(len((1,3)))
        self.prove(p)
    def test_assertion_with_len6(self):
        def p(): return len((1,3)) == 2
        self.prove(p)
    def test_assertion_with_len7(self):
        def p(): return len((1,*(2,3,4),5)) == 5
        self.prove(p)
    def test_assertion_with_len8(self):
        def p(t:istuple): return len(t) < len((*t,1))
        self.prove(p)

    def test_assertion_with_all1(self):
        def p(): return all(())
        self.prove(p)
    def test_assertion_with_all2(self):
        def p(t:istuple): return implies(all(t), all((*t, True)))
        self.prove(p)
    def test_assertion_with_all3(self):
        def p(t:istuple): return implies(all(t), all((True, *t)))
        self.prove(p)
    def test_assertion_with_all4(self):
        def p(t:istuple): return all((True,))
        self.prove(p)
    def test_assertion_with_all5(self):
        def p(t:istuple): return all((True, True, True, False))
        self.do_NOT_prove(p)

    def test_assertion_with_map1(self):
        def p(): return map(isint, ()) == () # incorrect; map produces an iterable!
        self.prove(p)
    def test_assertion_with_map2(self):
        def p(): return map(isint, (2,3)) == (True, True)
        self.prove(p)
    def test_assertion_with_map3(self):
        def p(): return all(map(isint, (2, 3)))
        self.prove(p)
    def test_assertion_with_map4(self):
        def p(): return not all(map(isint, (2, False)))
        self.prove(p)

    def test_assertion_with_range1(self):
        def p(): return isdefined(range(5))
        self.prove(p)
    def test_assertion_with_range2(self): # TODO runs forever
        def p(x:isint): return all(map(isnat, range(x)))
        self.prove(p)
    def test_assertion_with_range3(self):
        def p(x:isint): return isdefined(all(map(isint, range(x))))
        self.prove(p)
    def test_assertion_with_range4(self):
        def p(x:isint): return all(map(isint, range(x)))
        self.prove(p)
    def test_assertion_with_range5(self):
        def p(): return range(1) == (0,)
        self.prove(p)
    def test_assertion_with_range6(self):
        def p(): return range(2) == (0,1)
        self.prove(p)


if __name__ == '__main__':
    unittest.main()
