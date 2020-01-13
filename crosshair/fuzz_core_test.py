import random
import sys
import time
import unittest
import traceback
from typing import *
from crosshair.core import proxy_for_type, coerce_to_smt_sort, origin_of, type_args_of, realize
from crosshair.statespace import SinglePathNode, TrackingStateSpace, CallAnalysis, VerificationStatus, IgnoreAttempt, CrosshairInternal
from crosshair.util import debug, set_debug

IMMUTABLE_BASE_TYPES = [bool, int, float, str, frozenset]
ALL_BASE_TYPES = IMMUTABLE_BASE_TYPES + [set, dict, list]
def gen_type(r: random.Random, immutable_only: bool = False) -> type:
    base = r.choice(IMMUTABLE_BASE_TYPES if immutable_only else ALL_BASE_TYPES)
    if base is dict:
        return Dict[gen_type(r, immutable_only=True), gen_type(r)] # type: ignore
    elif base is list:
        return List[gen_type(r)] # type: ignore
    elif base is set:
        return Set[gen_type(r, immutable_only=True)] # type: ignore
    elif base is frozenset:
        return FrozenSet[gen_type(r, immutable_only=True)] # type: ignore
    else:
        return base


def value_for_type(typ: Type, r: random.Random) -> object:
    '''
    post: isinstance(_, typ)
    '''
    origin = origin_of(typ)
    type_args = type_args_of(typ)
    if typ is bool:
        ret: object = r.choice([True, False])
    elif typ is int:
        ret = r.choice([-1, 0, 1, 2, 10])
    elif typ is float:
        ret = r.choice([-1.0, 0.0, 1.0, 2.0, 10.0])  # TODO: Inf, NaN
    elif typ is str:
        ret = r.choice(['', 'x', '0', 'xyz'])#, '\0']) # TODO: null does not work properly yet
    elif origin in (list, set, frozenset):
        (item_type,) = type_args
        ret = origin([value_for_type(item_type, r) for _ in range(r.choice([0, 1, 2]))])
    elif origin is dict:
        (key_type, val_type) = type_args
        ret = {value_for_type(key_type, r): value_for_type(val_type, r)
               for _ in range(r.choice([0, 1, 2]))}
    else:
        raise NotImplementedError
    return ret


class FuzzTest(unittest.TestCase):
    r: random.Random

    def __init__(self, *a):
        self.r = random.Random(23472)
        super().__init__(*a)

    def gen_binary_op(self) -> str:
        '''
        post: _.format(a='', b='')
        '''
        return self.r.choice([
            '{a} + {b}',
            '{a} - {b}',
            '{a} * {b}',
            '{a} / {b}',
            '{a} < {b}',
            '{a} <= {b}',
            '{a} >= {b}',
            '{a} > {b}',
            '{a} == {b}',
            '{a}[{b}]',
            '{a} in {b}',
            '{a} & {b}',
            '{a} | {b}',
            '{a} ^ {b}',
            '{a} and {b}',
            '{a} or {b}',
            '{a} // {b}',
            '{a} ** {b}',
            '{a} % {b}',
        ])

    def symbolic_run(self, fn: Callable[[TrackingStateSpace], bool]) -> Tuple[object, Optional[BaseException]]:
        search_root = SinglePathNode(True)
        itr = 0
        for _ in range(200):
            itr += 1
            space = TrackingStateSpace(time.time() + 10.0, 1.0, search_root=search_root)
            try:
                return (realize(fn(space)), None)
            except IgnoreAttempt:
                pass
            except BaseException as e:
                #import traceback
                #traceback.print_exc()
                return (None, e)
            top_analysis, space_exhausted = space.bubble_status(CallAnalysis())
            if space_exhausted:
                return (None, CrosshairInternal(f'exhausted after {itr} iterations'))
        return (None, CrosshairInternal('Unable to find a successful symbolic execution'))

    def gen_binary_expr(self) -> Optional[Tuple[str, Mapping, Callable[[TrackingStateSpace], object]]]:
        ta, tb = gen_type(self.r), gen_type(self.r)
        va = value_for_type(ta, self.r)
        vb = value_for_type(tb, self.r)
        op = self.gen_binary_op()
        literal_bindings = {'a': va, 'b': vb}
        expr = op.format(a='a', b='b')
        def checker(space):
            a = proxy_for_type(ta, space, 'a')
            b = proxy_for_type(tb, space, 'b')
            if a != va:
                raise IgnoreAttempt
            if b != vb:
                raise IgnoreAttempt
            return eval(expr)
        return (expr, literal_bindings, checker)

    def genexprs(self, count):
        while count > 0:
            tupl = self.gen_binary_expr()
            if tupl is not None:
                yield tupl
                count -= 1

    def runexpr(self, expr, bindings):
        try:
            return (eval(expr, {}, bindings), None)
        except Exception as e:
            debug(f'eval of "{expr}" produced exception "{e}"')
            return (None, e)

    # Note that test case generation doesn't seem to be deterministic
    # between Python 3.7 and 3.8.
    def test_binary_op(self) -> None:
        NUM_TRIALS = 32 # raise this as we make fixes
        for expr, literal_bindings, symbolic_checker in self.genexprs(NUM_TRIALS):
            with self.subTest(msg=f'evaluating {expr} with {literal_bindings}'):
                debug(f'  =====  {expr} with {literal_bindings}  =====  ')
                compiled = compile(expr, '<string>', 'eval')
                literal_result = self.runexpr(expr, literal_bindings)
                symbolic_result = self.symbolic_run(symbolic_checker)
                if (literal_result[0] != symbolic_result[0] or
                    type(literal_result[1]) != type(symbolic_result[1])):
                    debug(
                        f'  *****  BEGIN FAILURE FOR {expr} WITH {literal_bindings}  *****  ')
                    debug(f'  *****  Expected: {literal_result}')
                    debug(f'  *****  Symbolic result: {symbolic_result}')
                    debug(f'  *****  END FAILURE FOR {expr}  *****  ')
                    self.assertEqual(literal_result, symbolic_result)
                debug(' OK ', literal_result, symbolic_result)


if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
