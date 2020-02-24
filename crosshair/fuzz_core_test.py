import builtins
import copy
import inspect
import random
import sys
import time
import unittest
import traceback
from typing import *
from crosshair.core import proxy_for_type, type_args_of, realize, Patched, builtin_patches
import crosshair.core_and_libs
from crosshair.condition_parser import resolve_signature
from crosshair.libimpl.builtinslib import coerce_to_smt_sort, origin_of
from crosshair.statespace import SinglePathNode, TrackingStateSpace, CallAnalysis, VerificationStatus, IgnoreAttempt, CrosshairInternal
from crosshair.util import debug, set_debug, IdentityWrapper

T = TypeVar('T')


IMMUTABLE_BASE_TYPES = [bool, int, float, str, frozenset]
ALL_BASE_TYPES = IMMUTABLE_BASE_TYPES + [set, dict, list]
def gen_type(r: random.Random, type_root: Type) -> type:
    if type_root is Hashable:
        base = r.choice(IMMUTABLE_BASE_TYPES)
    elif type_root is object:
        base = r.choice(ALL_BASE_TYPES)
    else:
        base = type_root
    if base is dict:
        kt = gen_type(r, Hashable)
        vt = gen_type(r, object)
        return Dict[kt, vt] # type: ignore
    elif base is list:
        return List[gen_type(r, object)] # type: ignore
    elif base is set:
        return Set[gen_type(r, Hashable)] # type: ignore
    elif base is frozenset:
        return FrozenSet[gen_type(r, Hashable)] # type: ignore
    else:
        return base


def value_for_type(typ: Type, r: random.Random) -> object:
    '''
    post: isinstance(_, typ)
    '''
    origin = origin_of(typ)
    type_args = type_args_of(typ)
    if typ is bool:
        return r.choice([True, False])
    elif typ is int:
        return r.choice([-1, 0, 1, 2, 10])
    elif typ is float:
        return r.choice([-1.0, 0.0, 1.0, 2.0, 10.0])  # TODO: Inf, NaN
    elif typ is str:
        return r.choice(['', 'x', '0', 'xyz'])#, '\0']) # TODO: null does not work properly yet
    elif origin in (list, set, frozenset):
        (item_type,) = type_args
        items = []
        for _ in range(r.choice([0, 1, 2])):
            items.append(value_for_type(item_type, r))
        return origin(items)
    elif origin is dict:
        (key_type, val_type) = type_args
        ret = {}
        for _ in range(r.choice([0, 1, 2])):
            ret[value_for_type(key_type, r)] = value_for_type(val_type, r) # type: ignore
        return ret
    raise NotImplementedError

class FuzzTest(unittest.TestCase):
    r: random.Random
    def __init__(self, *a):
        super().__init__(*a)

    def setUp(self) -> None:
        self.r = random.Random(1348)

    def gen_unary_op(self) -> Tuple[str, Type]:
        return self.r.choice([
            #('iter({})', object),
            #('reversed({})', object),
            ('+{}', object),
            ('-{}', object),
            ('~{}', object),
            # dir(), pickling?
        ])

    def gen_binary_op(self) -> Tuple[str, Type, Type]:
        '''
        post: _[0].format('a', 'b')
        '''
        return self.r.choice([
            ('{} + {}', object, object),
            ('{} - {}', object, object),
            ('{} * {}', object, object),
            ('{} / {}', object, object),
            ('{} < {}', object, object),
            ('{} <= {}', object, object),
            ('{} >= {}', object, object),
            ('{} > {}', object, object),
            ('{} == {}', object, object),
            ('{}[{}]', object, object),
            ('{}.__delitem__({})', object, object),
            ('{} in {}', object, object),
            ('{} & {}', object, object),
            ('{} | {}', object, object),
            ('{} ^ {}', object, object),
            ('{} and {}', object, object),
            ('{} or {}', object, object),
            ('{} // {}', object, object),
            ('{} ** {}', object, object),
            ('{} % {}', object, object),
        ])

    def symbolic_run(self, fn: Callable[[TrackingStateSpace], object]) -> Tuple[object, Optional[BaseException]]:
        search_root = SinglePathNode(True)
        patches = Patched(enabled=lambda: True)
        with patches:
            for itr in range(1, 200):
                debug('iteration', itr)
                space = TrackingStateSpace(time.time() + 10.0, 1.0, search_root=search_root)
                try:
                    return (realize(fn(space)), None)
                except IgnoreAttempt as e:
                    debug('ignore iteration attempt: ', str(e))
                    pass
                except BaseException as e:
                    debug(traceback.format_exc())
                    return (None, e)
                top_analysis, space_exhausted = space.bubble_status(CallAnalysis())
                if space_exhausted:
                    return (None, CrosshairInternal(f'exhausted after {itr} iterations'))
        return (None, CrosshairInternal('Unable to find a successful symbolic execution'))

    def runexpr(self, expr, bindings):
        try:
            return (eval(expr, {}, bindings), None)
        except Exception as e:
            debug(f'eval of "{expr}" produced exception "{e}"')
            return (None, e)

    def run_class_method_trials(self, cls: Type, min_trials: int) -> None:
        debug('Checking class', cls)
        for method_name, method in list(inspect.getmembers(cls)):
            # We expect some methods to be different (at least, for now):
            if method_name.startswith('__'):
                continue
            if method_name.startswith('_c_'):  # Leftovers from forbiddenfruit curses
                continue
            if not (inspect.isfunction(method) or inspect.ismethoddescriptor(method)):
                continue
            sig = resolve_signature(method)
            if sig is None:
                continue
            debug('Checking method', method_name)
            num_trials = min_trials # TODO: something like this?:  min_trials + round(len(sig.parameters) ** 1.5)
            arg_names = [chr(ord('a') + i - 1) for i in range(1, len(sig.parameters))]
            # TODO: some methods take kw-only args (list.sort for example):
            expr_str = 'self.' + method_name + '(' + ','.join(arg_names) + ')'
            arg_type_roots = {name: object for name in arg_names}
            arg_type_roots['self'] = cls
            for trial_num in range(num_trials):
                self.run_trial(expr_str, arg_type_roots, f'{method_name} #{trial_num}')

    def run_trial(self, expr_str: str, arg_type_roots: Dict[str, Type], trial_desc: str) -> None:
        expr = expr_str.format(*arg_type_roots.keys())
        typed_args = {name: gen_type(self.r, type_root)
                      for name, type_root in arg_type_roots.items()}
        literal_args = {name: value_for_type(typ, self.r)
                        for name, typ in typed_args.items()}
        def symbolic_checker(space: TrackingStateSpace) -> object:
            symbolic_args = {name: proxy_for_type(typ, space, name)
                             for name, typ in typed_args.items()}
            for name in typed_args.keys():
                if literal_args[name] != symbolic_args[name]:
                    raise IgnoreAttempt(f'symbolic "{name}" not equal to literal "{name}"')
            return eval(expr, symbolic_args)
        with self.subTest(msg=f'Trial {trial_desc}: evaluating {expr} with {literal_args}'):
            debug(f'  =====  {expr} with {literal_args}  =====  ')
            compiled = compile(expr, '<string>', 'eval')
            literal_result = self.runexpr(expr, copy.deepcopy(literal_args))
            symbolic_result = self.symbolic_run(symbolic_checker)
            if (literal_result[0] != symbolic_result[0] or
                type(literal_result[1]) != type(symbolic_result[1])):
                debug(
                    f'  *****  BEGIN FAILURE FOR {expr} WITH {literal_args}  *****  ')
                debug(f'  *****  Expected: {literal_result}')
                debug(f'  *****  Symbolic result: {symbolic_result}')
                debug(f'  *****  END FAILURE FOR {expr}  *****  ')
                self.assertEqual(literal_result, symbolic_result)
            debug(' OK ', literal_result, symbolic_result)

    #
    # Actual tests below:
    #

    def test_unary_ops(self) -> None:
        NUM_TRIALS = 50 # raise this as we make fixes
        for i in range(NUM_TRIALS):
            expr_str, type_root = self.gen_unary_op()
            arg_type_roots = {'a': type_root}
            self.run_trial(expr_str, arg_type_roots, str(i))

    def test_binary_ops(self) -> None:
        NUM_TRIALS = 100 # raise this as we make fixes
        for i in range(NUM_TRIALS):
            expr_str, type_root1, type_root2 = self.gen_binary_op()
            arg_type_roots = {'a': type_root1, 'b': type_root2}
            self.run_trial(expr_str, arg_type_roots, str(i))

    def test_str_methods(self) -> None:
        self.run_class_method_trials(str, 3)

    def test_list_methods(self) -> None:
        self.run_class_method_trials(list, 1)

    def test_dict_methods(self) -> None:
        self.run_class_method_trials(dict, 1)

if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    unittest.main()
