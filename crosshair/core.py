# TODO: Can we pass any value for object? (b/c it is syntactically bound to a limited set of operations?)
# TODO: mutating symbolic Callables?
# TODO: shallow immutability checking? Clarify design here.
# TODO: standard library contracts
# TODO: Type[T] values
# TODO: conditions on Callable arguments/return values
# TODO: Subclass constraint rules
# TODO: Symbolic subclasses
# TODO: Test Z3 Arrays nested inside Datastructures
# TODO: identity-aware repr'ing for result messages
# TODO: larger examples
# TODO: increase test coverage: Any, object, and bounded type vars
# TODO: graceful handling of expression parse errors on conditions
# TODO: double-check counterexamples
# TODO: non-dataclass not copyable?

from dataclasses import dataclass, replace
from typing import *
import ast
import builtins
import collections
import copy
import enum
import inspect
import io
import itertools
import functools
import operator
import os.path
import random
import sys
import time
import traceback
import types
import typing

import typing_inspect  # type: ignore
import z3  # type: ignore

from crosshair.util import CrosshairInternal, IdentityWrapper, debug, set_debug, extract_module_from_file
from crosshair.abcstring import AbcString
from crosshair.condition_parser import get_fn_conditions, get_class_conditions, ConditionExpr, Conditions, fn_globals
from crosshair import contracted_builtins
from crosshair import dynamic_typing
from crosshair.enforce import EnforcedConditions, PostconditionFailed

_CHOICE_RANDOMIZATION = 4221242075

_UNIQ = 0
_HEAP:List[Tuple[z3.ExprRef, Type, object]] = []
def reset_for_iteration():
    global _UNIQ
    _UNIQ = 0
    global _HEAP
    _HEAP[:] = []

def uniq():
    global _UNIQ
    _UNIQ += 1
    if _UNIQ >= 1000000:
        raise Exception('Exhausted var space')
    return '{:06d}'.format(_UNIQ)

def frame_summary_for_fn(frames:traceback.StackSummary, fn:Callable) -> traceback.FrameSummary:
    fn_name = fn.__name__
    fn_file = inspect.getsourcefile(fn)
    for frame in reversed(frames):
        if frame.name == fn_name and os.path.samefile(frame.filename, fn_file):
            return frame
    raise CrosshairInternal('Unable to find function {} in stack frames'.format(fn_name))

_MISSING = object()
class SearchTreeNode:
    exhausted :bool = False
    positive :Optional['SearchTreeNode'] = None
    negative :Optional['SearchTreeNode'] = None
    model_condition :Any = _MISSING
    statehash :Optional[str] = None
    def choose(self, favor_true=False) -> Tuple[bool, 'SearchTreeNode']:
        assert self.positive is not None
        assert self.negative is not None
        positive_ok = not self.positive.exhausted
        negative_ok = not self.negative.exhausted
        if positive_ok and negative_ok:
            if favor_true:
                choice = True
            else:
                choice = bool(random.randint(0, 1))
        else:
            choice = positive_ok
        if choice:
            return (True, self.positive)
        else:
            return (False, self.negative)

    @classmethod
    def check_exhausted(cls, history:List['SearchTreeNode'], terminal_node:'SearchTreeNode') -> bool:
        terminal_node.exhausted = True
        for node in reversed(history):
            if (node.positive and node.positive.exhausted and
                node.negative and node.negative.exhausted):
                node.exhausted = True
            else:
                return False
        return True

class IgnoreAttempt(Exception):
    def __init__(self, *a):
        CrosshairInternal.__init__(self, *a)
        debug('IgnoreAttempt', str(self))
    
class UnknownSatisfiability(Exception):
    pass

class NotDeterministic(CrosshairInternal):
    pass

class CrosshairUnsupported(CrosshairInternal):
    def __init__(self, *a):
        CrosshairInternal.__init__(self, *a)
        debug('CrosshairUnsupported. Stack trace:\n' + ''.join(traceback.format_stack()))

def model_value_to_python(value):
    if z3.is_string(value):
        return value.as_string()
    elif z3.is_real(value):
        return float(value.as_fraction())
    else:
        return ast.literal_eval(repr(value))

class StateSpace:
    def __init__(self, previous_searches:Optional[SearchTreeNode]=None, execution_deadline:Optional[float]=None):
        if previous_searches is None:
            previous_searches = SearchTreeNode()
        self.solver = z3.OrElse('smt', 'qfnra-nlsat').solver()
        self.solver.set(mbqi=True)
        self.search_position = previous_searches
        self.choices_made :List[SearchTreeNode] = []
        self.model_additions :MutableMapping[str,object] = {}
        self.execution_deadline = execution_deadline if execution_deadline else time.time() + 10.0
        self.running_framework_code = False

    def framework(self) -> ContextManager:
        return WithFrameworkCode(self)
            
    def add(self, expr:z3.ExprRef) -> None:
        #debug('Committed to ', expr)
        self.solver.add(expr)

    def check(self, expr:z3.ExprRef) -> z3.CheckSatResult:
        if time.time()  > self.execution_deadline:
            debug('Path execution timeout after making ', len(self.choices_made), ' choices.')
            raise UnknownSatisfiability()
        solver = self.solver
        solver.push()
        solver.add(expr)
        #debug('CHECK ? ' + str(solver))
        ret = solver.check()
        #debug('CHECK => ' + str(ret))
        if ret not in (z3.sat, z3.unsat):
            #alt_solver z3.Then('qfnra-nlsat').solver()
            raise UnknownSatisfiability(str(ret)+': '+str(solver))
        solver.pop()
        return ret

    def choose_possible(self, expr:z3.ExprRef, favor_true=False) -> bool:
        with self.framework():
            notexpr = z3.Not(expr)
            node = self.search_position
            statedesc = ''.join(traceback.format_stack())
            if node.statehash is None:
                node.statehash = statedesc
            else:
                if node.statehash != statedesc:
                    debug(' *** Begin Not Deterministic Debug *** ')
                    debug('     First state: ', len(node.statehash))
                    debug(node.statehash)
                    debug('     Last state: ', len(statedesc))
                    debug(statedesc)
                    debug('     Stack Diff: ')
                    import difflib
                    debug('\n'.join(difflib.context_diff(node.statehash.split('\n'), statedesc.split('\n'))))
                    debug(' *** End Not Deterministic Debug *** ')
                    raise NotDeterministic()
            if node.positive is None and node.negative is None:
                node.positive = SearchTreeNode()
                node.negative = SearchTreeNode()
                true_sat, false_sat = self.check(expr), self.check(notexpr)
                could_be_true = (true_sat == z3.sat)
                could_be_false = (false_sat == z3.sat)
                if (not could_be_true) and (not could_be_false):
                    debug(' *** Reached impossible code path *** ', true_sat, false_sat, expr)
                    raise Exception('Reached impossible code path')
                if not could_be_true:
                    node.positive.exhausted = True
                if not could_be_false:
                    node.negative.exhausted = True

            (choose_true, new_search_node) = self.search_position.choose(favor_true=favor_true)
            self.choices_made.append(self.search_position)
            self.search_position = new_search_node
            expr = expr if choose_true else notexpr
            #debug('CHOOSE', expr)
            self.add(expr)
            return choose_true

    def find_model_value(self, expr:z3.ExprRef) -> object:
        while True:
            node = self.search_position
            if node.model_condition is _MISSING:
                if self.solver.check() != z3.sat:
                    raise Exception('model unexpectedly became unsatisfiable')
                node.model_condition = self.solver.model().evaluate(expr, model_completion=True)
            value = node.model_condition
            if self.choose_possible(expr == value, favor_true=True):
                if self.solver.check() != z3.sat:
                    raise Exception('could not confirm model satisfiability after fixing value')
                return model_value_to_python(value)

    def find_model_value_for_function(self, expr:z3.ExprRef) -> object:
        wrapper = IdentityWrapper(expr)
        while True:
            node = self.search_position
            if node.model_condition is _MISSING:
                if self.solver.check() != z3.sat:
                    raise Exception('model unexpectedly became unsatisfiable')
                finterp = self.solver.model()[expr]
                node.model_condition = (wrapper, finterp)
            cmpvalue, finterp = node.model_condition
            if self.choose_possible(wrapper == cmpvalue, favor_true=True):
                if self.solver.check() != z3.sat:
                    raise Exception('could not confirm model satisfiability after fixing value')
                return finterp

    def check_exhausted(self) -> bool:
        return SearchTreeNode.check_exhausted(self.choices_made, self.search_position)

class ExceptionFilter:
    ignore: bool = False
    user_exc: Optional[Tuple[BaseException, traceback.StackSummary]] = None
    def has_user_exception(self) -> bool:
        return self.user_exc is not None
    def __enter__(self) -> 'ExceptionFilter':
        return self
    def __exit__(self, exc_type, exc_value, tb):
        if isinstance(exc_value, (PostconditionFailed, IgnoreAttempt)):
            self.ignore = True
            return True # suppress
        # Postcondition : although this indicates a problem, it's with a subroutine; not this function.
        if isinstance(exc_value, (UnknownSatisfiability, CrosshairInternal)):
            return False # internal issue: re-raise
        if isinstance(exc_value, BaseException): # TODO: Exception?
            # Most other issues are assumed to be user-level exceptions:
            self.user_exc = (exc_value, traceback.extract_tb(sys.exc_info()[2]))
            return True # suppress user-level exception
        return False # re-raise resource and system issues


class WithFrameworkCode:
    def __init__(self, space:StateSpace):
        self.space = space
        self.previous = None
    def __enter__(self):
        assert self.previous is None # not reentrant
        self.previous = self.space.running_framework_code
        self.space.running_framework_code = True
    def __exit__(self, exc_type, exc_value, tb):
        assert self.previous is not None
        self.space.running_framework_code = self.previous
        
def could_be_instanceof(v:object, typ:Type) -> bool:
    ret = isinstance(v, origin_of(typ))
    return ret or isinstance(v, ProxiedObject)

HeapRef = z3.DeclareSort('HeapRef')
def find_key_in_heap(space:StateSpace, ref:z3.ExprRef, typ:Type) -> object:
    with space.framework():
        global _HEAP
        debug('HEAP key lookup ', ref, ' out of ', len(_HEAP), ' items')
        for (curref, curtyp, curval) in _HEAP:
            if not dynamic_typing.unify(curtyp, typ):
                continue
            if smt_fork(space, curref == ref):
                return curval
        ret = proxy_for_type(typ, space, 'heapval' + str(typ) + uniq())
        assert dynamic_typing.unify(python_type(ret), typ),'proxy type was {} and type required was {}'.format(type(ret), typ)
        _HEAP.append((ref, typ, ret))
        return ret

def find_val_in_heap(space:StateSpace, value:object) -> z3.ExprRef:
    with space.framework():
        global _HEAP
        for (curref, curtyp, curval) in _HEAP:
            if curval is value:
                return curref
        ref = z3.Const('heapkey'+str(value)+uniq(), HeapRef)
        for (curref, _, _) in _HEAP:
            space.add(ref != curref)
        _HEAP.append((ref, type(value), value))
        return ref

def normalize_pytype(typ:Type) -> Type:
    if typing_inspect.is_typevar(typ):
        # we treat type vars in the most general way possible (the bound, or as 'object')
        bound = typing_inspect.get_bound(typ)
        if bound is not None:
            return normalize_pytype(bound)
        constraints = typing_inspect.get_constraints(typ)
        if constraints:
            raise CrosshairUnsupported
            # TODO: not easy; interpreting as a Union allows the type to be
            # instantiated differently in different places:
            # return Union.__getitem__(tuple(map(normalize_pytype, constraints)))
        return object
    if typ is Any:
        # The distinction between any and object is for type checking, crosshair treats them the same
        return object
    return typ

def python_type(o:object) -> Type:
    if isinstance(o, SmtBackedValue):
        return o.python_type
    elif isinstance(o, ProxiedObject):
        return object.__getattribute__(o, '_cls')
    else:
        return type(o)

def origin_of(typ:Type) -> Type:
    typ = _WRAPPER_TYPE_TO_PYTYPE.get(typ, typ)
    if hasattr(typ, '__origin__'):
        return typ.__origin__
    return typ

def type_args_of(typ:Type) -> Tuple[Type, ...]:
    if getattr(typ, '__args__', None):
        return typing_inspect.get_args(typ, evaluate=True)
    else:
        return ()

def name_of_type(typ:Type) -> str:
    return typ.__name__ if hasattr(typ, '__name__') else str(typ).split('.')[-1]

_SMT_FLOAT_SORT = z3.RealSort() # difficulty getting the solver to use z3.Float64()

_TYPE_TO_SMT_SORT = {
    bool : z3.BoolSort(),
    int : z3.IntSort(),
    float : _SMT_FLOAT_SORT,
    str : z3.StringSort(),
}

def possibly_missing_sort(sort):
    datatype = z3.Datatype('optional('+str(sort)+')')
    datatype.declare('missing')
    datatype.declare('present', ('valueat', sort))
    ret = datatype.create()
    return ret
    
def tuple_sort(names, sorts):
    datatype = z3.Datatype('tupl(' + ','.join(map(str, sorts)) + ')')
    datatype.declare('tupl', *zip(names, sorts))
    ret = datatype.create()
    return ret
    

def type_to_smt_sort(t: Type):
    t = normalize_pytype(t)
    if t in _TYPE_TO_SMT_SORT:
        return _TYPE_TO_SMT_SORT[t]
    origin = origin_of(t)
    if origin in (list, tuple, Sequence, Container):
        item_type = t.__args__[0]
        item_sort = type_to_smt_sort(item_type)
        if item_sort is None:
            item_sort = HeapRef
        return z3.SeqSort(item_sort)
    return None


def smt_var(typ: Type, name: str):
    z3type = type_to_smt_sort(typ)
    if z3type is None:
        if getattr(typ, '__origin__', None) is Tuple:
            if len(typ.__args__) == 2 and typ.__args__[1] == ...:
                z3type = z3.SeqSort(type_to_smt_sort(typ.__args__[0]))
            else:
                return tuple(smt_var(t, name+str(idx)) for (idx, t) in enumerate(typ.__args__))
    if z3type is None:
        raise Exception('unable to find smt sort for python type '+str(typ))
    return z3.Const(name, z3type)

SmtGenerator = Callable[[StateSpace, type, Union[str, z3.ExprRef]], object]

_PYTYPE_TO_WRAPPER_TYPE :Dict[type, SmtGenerator] = {} # to be populated later
_WRAPPER_TYPE_TO_PYTYPE :Dict[SmtGenerator, type] = {}

def crosshair_type_for_python_type(typ:Type) -> Optional[SmtGenerator]:
    typ = normalize_pytype(typ)
    origin = origin_of(typ)
    if origin is Union:
        return SmtUnion(frozenset(typ.__args__))
    Typ = _PYTYPE_TO_WRAPPER_TYPE.get(origin)
    if Typ:
        return Typ
    return None

def smt_bool_to_int(a: z3.ExprRef) -> z3.ExprRef:
    return z3.If(a, 1, 0)

def smt_int_to_float(a: z3.ExprRef) -> z3.ExprRef:
    if _SMT_FLOAT_SORT == z3.Float64():
        return z3.fpRealToFP(z3.RNE(), z3.ToReal(a), _SMT_FLOAT_SORT)
    elif _SMT_FLOAT_SORT == z3.RealSort():
        return z3.ToReal(a)
    else:
        raise Exception()

def smt_bool_to_float(a: z3.ExprRef) -> z3.ExprRef:
    if _SMT_FLOAT_SORT == z3.Float64():
        return z3.If(a, z3.FPVal(1.0, _SMT_FLOAT_SORT), z3.FPVal(0.0, _SMT_FLOAT_SORT))
    elif _SMT_FLOAT_SORT == z3.RealSort():
        return z3.If(a, z3.RealVal(1), z3.RealVal(0))
    else:
        raise Exception()

_NUMERIC_PROMOTION_FNS = {
    (bool, bool): lambda x,y: (smt_bool_to_int(x), smt_bool_to_int(y), int),
    (bool, int): lambda x,y: (smt_bool_to_int(x), y, int),
    (int, bool): lambda x,y: (x, smt_bool_to_int(y), int),
    (bool, float): lambda x,y: (smt_bool_to_float(x), y, float),
    (float, bool): lambda x,y: (x, smt_bool_to_float(y), float),
    (int, int): lambda x,y: (x, y, int),
    (int, float): lambda x,y: (smt_int_to_float(x), y, float),
    (float, int): lambda x,y: (x, smt_int_to_float(y), float),
    (float, float): lambda x, y: (x, y, float),
}

_LITERAL_PROMOTION_FNS = {
    bool: z3.BoolVal,
    int: z3.IntVal,
    float: z3.RealVal if _SMT_FLOAT_SORT == z3.RealSort() else (lambda v: z3.FPVal(v, _SMT_FLOAT_SORT)),
    str: z3.StringVal,
}

def smt_coerce(val:Any) -> z3.ExprRef:
    if isinstance(val, SmtBackedValue):
        return val.var
    return val

def coerce_to_smt_var(space:StateSpace, v:Any) -> Tuple[z3.ExprRef, Type]:
    if isinstance(v, SmtBackedValue):
        return (v.var, v.python_type)
    if isinstance(v, (tuple, list)):
        (vars, pytypes) = zip(*(coerce_to_smt_var(space,i) for i in v))
        if len(vars) == 0:
            return ([], list) if isinstance(v, list) else ((), tuple)
        elif len(vars) == 1:
            return (z3.Unit(vars[0]), list)
        else:
            return (z3.Concat(*map(z3.Unit,vars)), list)
    promotion_fn = _LITERAL_PROMOTION_FNS.get(type(v))
    if promotion_fn:
        return (promotion_fn(v), type(v))
    return (find_val_in_heap(space, v), type(v))

def coerce_to_ch_value(v:Any, statespace:StateSpace) -> object:
    (smt_var, py_type) = coerce_to_smt_var(statespace, v)
    Typ = crosshair_type_for_python_type(py_type)
    if Typ is None:
        raise Exception('Unable to get ch type from python type: '+str(py_type))
    return Typ(statespace, py_type, smt_var)

def smt_fork(space:StateSpace, expr:Optional[z3.ExprRef]=None):
    if expr is None:
        expr = z3.Bool('fork'+uniq())
    return SmtBool(space, bool, expr).__bool__()

class SmtBackedValue:
    def __init__(self, statespace:StateSpace, typ: Type, smtvar:object):
        self.statespace = statespace
        self.python_type = typ
        if isinstance(smtvar, str):
            self.var = self.__init_var__(typ, smtvar)
        else:
            self.var = smtvar
            # TODO test that smtvar's sort matches expected?
    def __init_var__(self, typ, varname):
        return smt_var(typ, varname)
    def __eq__(self, other):
        try:
            return SmtBool(self.statespace, bool, self.var == coerce_to_smt_var(self.statespace, other)[0])
        except z3.z3types.Z3Exception as e:
            if 'sort mismatch' in str(e):
                return False
            raise
    def __ne__(self, other):
        return not self.__eq__(other)
    def __req__(self, other):
        return coerce_to_ch_value(other, self.statespace).__eq__(self)
    def __rne__(self, other):
        return coerce_to_ch_value(other, self.statespace).__ne__(self)
    def _binary_op(self, other, op):
        left, right = self.var, coerce_to_smt_var(self.statespace, other)[0]
        return self.__class__(self.statespace, self.python_type, op(left, right))
    def _cmp_op(self, other, op):
        return SmtBool(self.statespace, bool, op(self.var, smt_coerce(other)))
    def _unary_op(self, op):
        return self.__class__(self.statespace, self.python_type, op(self.var))

class SmtNumberAble(SmtBackedValue):
    def _numeric_op(self, other, op):
        if type(self) == complex or type(other) == complex:
            return op(complex(self), complex(other))
        l_var, lpytype = self.var, self.python_type
        r_var, rpytype = coerce_to_smt_var(self.statespace, other)
        promotion_fn = _NUMERIC_PROMOTION_FNS.get((lpytype, rpytype))
        if not promotion_fn:
            return NotImplemented
        l_var, r_var, common_pytype = promotion_fn(l_var, r_var)
        cls = _PYTYPE_TO_WRAPPER_TYPE[common_pytype]
        return cls(self.statespace, common_pytype, op(l_var, r_var))

    def _numeric_unary_op(self, op):
        var, pytype = self.var, self.python_type
        if pytype is bool:
            var = smt_bool_to_int(var)
            pytype = int
        cls = _PYTYPE_TO_WRAPPER_TYPE[pytype]
        return cls(self.statespace, pytype, op(var))

    def __pos__(self):
        return self._unary_op(operator.pos)
    def __neg__(self):
        return self._unary_op(operator.neg)
    def __abs__(self):
        return self._unary_op(lambda v: z3.If(v < 0, -v, v))
    
        
    def __lt__(self, other):
        return self._cmp_op(other, operator.lt)
    def __gt__(self, other):
        return self._cmp_op(other, operator.gt)
    def __le__(self, other):
        return self._cmp_op(other, operator.le)
    def __ge__(self, other):
        return self._cmp_op(other, operator.ge)

    def __add__(self, other):
        return self._numeric_op(other, operator.add)
    def __sub__(self, other):
        return self._numeric_op(other, operator.sub)
    def __mul__(self, other):
        return self._numeric_op(other, operator.mul)
    def __pow__(self, other):
        return self._binary_op(other, operator.pow)

    
    def __rmul__(self, other):
        return coerce_to_ch_value(other, self.statespace).__mul__(self)
    def __radd__(self, other):
        return coerce_to_ch_value(other, self.statespace).__add__(self)
    def __rsub__(self, other):
        return coerce_to_ch_value(other, self.statespace).__sub__(self)
    def __rtruediv__(self, other):
        return coerce_to_ch_value(other, self.statespace).__truediv__(self)
    def __rfloordiv__(self, other):
        return coerce_to_ch_value(other, self.statespace).__floordiv__(self)
    def __rmod__(self, other):
        return coerce_to_ch_value(other, self.statespace).__mod__(self)
    def __rpow__(self, other):
        return coerce_to_ch_value(other, self.statespace).__pow__(self)
    def __rlshift__(self, other):
        return coerce_to_ch_value(other, self.statespace).__lshift__(self)
    def __rrshift__(self, other):
        return coerce_to_ch_value(other, self.statespace).__rshift__(self)
    def __rand__(self, other):
        return coerce_to_ch_value(other, self.statespace).__and__(self)
    def __rxor__(self, other):
        return coerce_to_ch_value(other, self.statespace).__xor__(self)
    def __ror__(self, other):
        return coerce_to_ch_value(other, self.statespace).__or__(self)


class SmtBool(SmtNumberAble):
    def __init__(self, statespace:StateSpace, typ: Type, smtvar:object):
        assert typ == bool
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
    def __repr__(self):
        return self.__bool__().__repr__()
    def __hash__(self):
        return self.__bool__().__hash__()
    def __xor__(self, other):
        return self._binary_op(other, z3.Xor)
    def __not__(self):
        return self._unary_op(z3.Not)

    def __bool__(self):
        return self.statespace.choose_possible(self.var)
    def __int__(self):
        return SmtInt(self.statespace, int, smt_bool_to_int(self.var))
    def __float__(self):
        return SmtFloat(self.statespace, float, smt_bool_to_float(self.var))
    def __complex__(self):
        return complex(self.__float__())

    def __add__(self, other):
        return self._numeric_op(other, operator.add)
    def __sub__(self, other):
        return self._numeric_op(other, operator.sub)

class SmtInt(SmtNumberAble):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        assert typ == int
        SmtNumberAble.__init__(self, statespace, typ, smtvar)
    def _apply_bitwise(self, op: Callable, v1: int, v2: int) -> int:
        return op(v1.__index__(), v2.__index__())
    def __repr__(self):
        return self.__index__().__repr__()
    def __hash__(self):
        return self.__index__().__hash__()

    def __float__(self):
        return SmtFloat(self.statespace, float, smt_int_to_float(self.var))
    def __complex__(self):
        return complex(self.__float__())

    def __index__(self):
        #debug('WARNING: attempting to materialize symbolic integer. Trace:')
        #traceback.print_stack()
        if self == 0:
            return 0
        ret = self.statespace.find_model_value(self.var)
        assert type(ret) is int
        return ret
    def __bool__(self):
        return SmtBool(self.statespace, bool, self.var != 0).__bool__()
    def __int__(self):
        return self.__index__()
    def __truediv__(self, other):
        return self.__float__() / other
    def __floordiv__(self, other):
        # TODO: Does this assume that other is an integer?
        return self._binary_op(other, lambda x,y:z3.If(x%y==0 or x>=0, x/y, z3.If(y>=0, x/y+1, x/y-1)))
    def __mod__(self, other):
        return self._binary_op(other, operator.mod)

    # bitwise operators
    def __invert__(self):
        return -(self + 1)
    def __lshift__(self, other):
        if other < 0:
            raise ValueError('negative shift count')
        return self * (2 ** other)
    def __rshift__(self, other):
        if other < 0:
            raise ValueError('negative shift count')
        return self // (2 ** other)
    def __and__(self, other):
        return SmtInt(self.statespace, int, self._apply_bitwise(operator.and_, self, other))
    def __or__(self, other):
        return SmtInt(self.statespace, int, self._apply_bitwise(operator.or_, self, other))
    def __xor__(self, other):
        return SmtInt(self.statespace, int, self._apply_bitwise(operator.xor, self, other))

    
_Z3_ONE_HALF = z3.RealVal("1/2")
class SmtFloat(SmtNumberAble):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        assert typ == float
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
    def __repr__(self):
        return self.statespace.find_model_value(self.var).__repr__()
    def __hash__(self):
        return self.statespace.find_model_value(self.var).__hash__()
    def __float__(self):
        return self.statespace.find_model_value(self.var).__float__()
    def __complex__(self):
        return complex(self.__float__())
    def __round__(self, ndigits=None):
        if ndigits is not None:
            factor = 10 ** ndigits
            return round(self * factor) / factor
        else:
            var, floor, nearest = self.var, z3.ToInt(self.var), z3.ToInt(self.var + _Z3_ONE_HALF)
            return SmtInt(self.statespace, int, z3.If(var != floor + _Z3_ONE_HALF, nearest, z3.If(floor % 2 == 0, floor, floor + 1)))
    def __floor__(self):
        return SmtInt(self.statespace, int, z3.ToInt(self.var))
    def __ceil__(self):
        var, floor = self.var, z3.ToInt(self.var)
        return SmtInt(self.statespace, int, z3.If(var == floor, floor, floor + 1))
    def __trunc__(self):
        var, floor = self.var, z3.ToInt(self.var)
        debug('trunc', var, floor)
        return SmtInt(self.statespace, int, z3.If(var >= 0, floor, floor + 1))

    def __truediv__(self, other):
        if not other:
            raise ZeroDivisionError('division by zero')
        return self._numeric_op(other, operator.truediv)

class SmtDictOrSet(SmtBackedValue):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        self.key_pytype = normalize_pytype(typ.__args__[0])
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        self.key_ch_type = crosshair_type_for_python_type(self.key_pytype)
        self.statespace.add(self._len() >= 0)
    def _arr(self):
        return self.var[0]
    def _len(self):
        return self.var[1]
    def __len__(self):
        return SmtInt(self.statespace, int, self._len())
    def __bool__(self):
        return SmtBool(self.statespace, bool, self._len() != 0).__bool__()


class SmtDict(SmtDictOrSet, collections.abc.MutableMapping):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        self.val_pytype = normalize_pytype(typ.__args__[1])
        SmtDictOrSet.__init__(self, statespace, typ, smtvar)
        self.val_ch_type = crosshair_type_for_python_type(self.val_pytype)
        arr_var = self._arr()
        len_var = self._len()
        self.val_missing_checker = arr_var.sort().range().recognizer(0)
        self.val_missing_constructor = arr_var.sort().range().constructor(0)
        self.val_constructor = arr_var.sort().range().constructor(1)
        self.val_accessor = arr_var.sort().range().accessor(1, 0)
        self.empty = z3.K(self._arr().sort().domain(), self.val_missing_constructor())
        self.statespace.add((arr_var == self.empty) == (len_var == 0))
    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        return (
            z3.Const(varname+'_map' + uniq(),
                     z3.ArraySort(type_to_smt_sort(self.key_pytype),
                                  possibly_missing_sort(type_to_smt_sort(self.val_pytype)))),
            z3.Const(varname + '_len' + uniq(), z3.IntSort())
        )
    def __repr__(self):
        return str(dict(self.items()))
    def __setitem__(self, k, v):
        missing = self.val_missing_constructor()
        (k,_), (v,_) = coerce_to_smt_var(self.statespace, k), coerce_to_smt_var(self.statespace, v)
        old_arr, old_len = self.var
        new_len = z3.If(z3.Select(old_arr, k) == missing, old_len + 1, old_len)
        self.var = (z3.Store(old_arr, k, self.val_constructor(v)), new_len)
    def __delitem__(self, k):
        missing = self.val_missing_constructor()
        (k,_) = coerce_to_smt_var(self.statespace, k)
        old_arr, old_len = self.var
        if SmtBool(self.statespace, bool, z3.Select(old_arr, k) == missing).__bool__():
            raise KeyError(k)
        if SmtBool(self.statespace, bool, self._len() == 0).__bool__():
            raise IgnoreAttempt('SmtDict in inconsistent state')
        self.var = (z3.Store(old_arr, k, missing), old_len - 1)
    def __getitem__(self, k):
        with self.statespace.framework():
            possibly_missing = self._arr()[coerce_to_smt_var(self.statespace, k)[0]]
            is_missing = self.val_missing_checker(possibly_missing)
            if SmtBool(self.statespace, bool, is_missing).__bool__():
                raise KeyError(k)
            if SmtBool(self.statespace, bool, self._len() == 0).__bool__():
                raise IgnoreAttempt('SmtDict in inconsistent state')
            return self.val_ch_type(self.statespace, self.val_pytype,
                                self.val_accessor(possibly_missing))
    def __iter__(self):
        arr_var, len_var = self.var
        idx = 0
        arr_sort = self._arr().sort()
        missing = self.val_missing_constructor()
        while SmtBool(self.statespace, bool, idx < len_var).__bool__():
            if SmtBool(self.statespace, bool, arr_var == self.empty).__bool__():
                raise IgnoreAttempt('SmtDict in inconsistent state')
            k = z3.Const('k'+str(idx) + uniq(), arr_sort.domain())
            v = z3.Const('v'+str(idx) + uniq(), self.val_constructor.domain(0))
            remaining = z3.Const('remaining' + str(idx) + uniq(), arr_sort)
            idx += 1
            self.statespace.add(arr_var == z3.Store(remaining, k, self.val_constructor(v)))
            self.statespace.add(z3.Select(remaining, k) == missing)
            yield self.key_ch_type(self.statespace, self.key_pytype, k)
            arr_var = remaining
        # In this conditional, we reconcile the parallel symbolic variables for length
        # and contents:
        if SmtBool(self.statespace, bool, arr_var != self.empty).__bool__():
            raise IgnoreAttempt('SmtDict in inconsistent state')


class SmtSet(SmtDictOrSet, collections.abc.Set):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        SmtDictOrSet.__init__(self, statespace, typ, smtvar)
        self.empty = z3.K(self._arr().sort().domain(), False)
        self.statespace.add((self._arr() == self.empty) == (self._len() == 0))
    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        return (
            z3.Const(varname+'_map' + uniq(),
                     z3.ArraySort(type_to_smt_sort(self.key_pytype),
                                  z3.BoolSort())),
            z3.Const(varname + '_len' + uniq(), z3.IntSort())
        )
    def __contains__(self, k):
        (k,_) = coerce_to_smt_var(self.statespace, k)
        present = self._arr()[k]
        return SmtBool(self.statespace, bool, present)
    def __iter__(self):
        arr_var, len_var = self.var
        idx = 0
        arr_sort = self._arr().sort()
        while SmtBool(self.statespace, bool, idx < len_var).__bool__():
            if SmtBool(self.statespace, bool, arr_var == self.empty).__bool__():
                raise IgnoreAttempt('SmtSet in inconsistent state')
            k = z3.Const('k' + str(idx) + uniq(), arr_sort.domain())
            remaining = z3.Const('remaining' + str(idx) + uniq(), arr_sort)
            idx += 1
            self.statespace.add(arr_var == z3.Store(remaining, k, True))
            self.statespace.add(z3.Not(z3.Select(remaining, k)))
            yield self.key_ch_type(self.statespace, self.key_pytype, k)
            arr_var = remaining
        # In this conditional, we reconcile the parallel symbolic variables for length
        # and contents:
        if SmtBool(self.statespace, bool, arr_var != self.empty).__bool__():
            raise IgnoreAttempt('SmtSet in inconsistent state')

class SmtMutableSet(SmtSet):
    def __repr__(self):
        return str(set(self))
    @classmethod 
    def _from_iterable(cls, it):
        # overrides collections.abc.Set's version
        return set(it)
    def add(self, k):
        (k,_) = coerce_to_smt_var(self.statespace, k)
        old_arr, old_len = self.var
        new_len = z3.If(z3.Select(old_arr, k), old_len, old_len + 1)
        self.var = (z3.Store(old_arr, k, True), new_len)
    def discard(self, k):
        (k,_) = coerce_to_smt_var(self.statespace, k)
        old_arr, old_len = self.var
        new_len = z3.If(z3.Select(old_arr, k), old_len - 1, old_len)
        self.var = (z3.Store(old_arr, k, False), new_len)
    
class SmtFrozenSet(SmtSet):
    def __repr__(self):
        return frozenset(self).__repr__()
    def __hash__(self):
        return frozenset(self).__hash__()
    @classmethod 
    def _from_iterable(cls, it):
        # overrides collections.abc.Set's version
        return set(it)
    def add(self, k):
        (k,_) = coerce_to_smt_var(self.statespace, k)
        old_arr, old_len = self.var
        new_len = z3.If(z3.Select(old_arr, k), old_len, old_len + 1)
        self.var = (z3.Store(old_arr, k, True), new_len)
    def discard(self, k):
        (k,_) = coerce_to_smt_var(self.statespace, k)
        old_arr, old_len = self.var
        new_len = z3.If(z3.Select(old_arr, k), old_len - 1, old_len)
        self.var = (z3.Store(old_arr, k, False), new_len)

def process_slice_vs_symbolic_len(space:StateSpace, i:slice, smt_len:z3.ExprRef) -> Union[z3.ExprRef, Tuple[z3.ExprRef, z3.ExprRef]]:
    def normalize_symbolic_index(idx):
        if isinstance(idx, int):
            return idx if idx >= 0 else smt_len + idx
        else:
            return z3.If(idx >= 0, idx, smt_len + idx)
    if isinstance(i, int) or isinstance(i, SmtInt):
        smt_i = smt_coerce(i)
        if smt_fork(space, z3.Or(smt_i >= smt_len, smt_i < -smt_len)):
            raise IndexError('index out of range')
        return normalize_symbolic_index(smt_i)
    elif isinstance(i, slice):
        smt_start, smt_stop, smt_step = map(smt_coerce, (i.start, i.stop, i.step))
        if smt_step not in (None, 1):
            raise Exception('slice steps not handled in slice: '+str(i))
        start = normalize_symbolic_index(smt_start) if i.start is not None else 0
        stop = normalize_symbolic_index(smt_stop) if i.stop is not None else smt_len
        return (start, stop)
    else:
        raise Exception('invalid slice parameter: '+str(i))

class SmtSequence(SmtBackedValue):
    def _smt_getitem(self, i):
        idx_or_pair = process_slice_vs_symbolic_len(self.statespace, i, z3.Length(self.var))
        if isinstance(idx_or_pair, tuple):
            (start, stop) = idx_or_pair
            return (z3.Extract(self.var, start, stop), True)
        else:
            return (self.var[idx_or_pair], False)
            
    def __iter__(self):
        idx = 0
        while len(self) > idx:
            yield self[idx]
            idx += 1
    def __len__(self):
        return SmtInt(self.statespace, int, z3.Length(self.var))
    def __bool__(self):
        return SmtBool(self.statespace, bool, z3.Length(self.var) > 0).__bool__()

class SmtUniformListOrTuple(SmtSequence):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        assert origin_of(typ) in (tuple, list)
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        self.item_pytype = normalize_pytype(typ.__args__[0]) # (index 0 works for both List[T] and Tuple[T, ...])
        self.item_ch_type = crosshair_type_for_python_type(self.item_pytype)
    def __add__(self, other):
        return self._binary_op(other, z3.Concat)
    def __radd__(self, other):
        other_seq, other_pytype = coerce_to_smt_var(self.statespace, other)
        return self.__class__(self.statespace, self.python_type, z3.Concat(other_seq, self.var))
    def __contains__(self, other):
        return SmtBool(self.statespace, bool, z3.Contains(self.var, z3.Unit(coerce_to_smt_var(self.statespace, other)[0])))
    def __getitem__(self, i):
        smt_result, is_slice = self._smt_getitem(i)
        if is_slice:
            return self.__class__(self.statespace, self.python_type, smt_result)
        elif smt_result.sort() == HeapRef:
            assert smt_result.sort() == HeapRef, 'smt type ({}) for {} is not a SeqSort over HeapRefs (item ch type:{})'.format(smt_result.sort(), smt_result, self.item_ch_type)
            return find_key_in_heap(self.statespace, smt_result, self.item_pytype)
        else:
            return self.item_ch_type(self.statespace, self.item_pytype, smt_result)

class SmtUniformList(SmtUniformListOrTuple, collections.abc.MutableSequence):
    def __repr__(self):
        return str(list(self))
    def extend(self, other):
        self.var = self.var + smt_coerce(other)
    def __setitem__(self, idx, obj):
        space, var = self.statespace, self.var
        varlen = z3.Length(var)
        idx_or_pair = process_slice_vs_symbolic_len(space, idx, varlen)
        if isinstance(idx_or_pair, tuple):
            (start, stop) = idx_or_pair
            to_insert = coerce_to_smt_var(space, obj)[0]
        else:
            (start, stop) = (idx_or_pair, idx_or_pair + 1)
            to_insert = z3.Unit(coerce_to_smt_var(space, obj)[0])
        self.var = z3.Concat(z3.Extract(var, 0, start),
                             to_insert,
                             z3.Extract(var, stop, varlen - stop))
    def __delitem__(self, idx):
        var = self.var
        varlen = z3.Length(var)
        idx_or_pair = process_slice_vs_symbolic_len(self.statespace, idx, varlen)
        if isinstance(idx_or_pair, tuple):
            (start, stop) = idx_or_pair
        else:
            (start, stop) = (idx_or_pair, idx_or_pair + 1)
        self.var = z3.Concat(z3.Extract(var, 0, start), z3.Extract(var, stop, varlen))
    def insert(self, idx, obj):
        space, var = self.statespace, self.var
        varlen = z3.Length(var)
        to_insert = z3.Unit(coerce_to_smt_var(space, obj)[0])
        if coerce_to_smt_var(space, idx)[0] == varlen:
            self.var = z3.Concat(var, to_insert)
        else:
            idx = process_slice_vs_symbolic_len(space, idx, varlen)
            self.var = z3.Concat(z3.Extract(var, 0, idx),
                                 to_insert,
                                 z3.Extract(var, idx, varlen - idx))
    def sort(self, **kw):
        self.var = coerce_to_smt_var(self.statespace, sorted(self, **kw))[0]


class SmtCallable(SmtBackedValue):
    def __init___(self, statespace:StateSpace, typ:Type, smtvar:object):
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
    def __init_var__(self, typ, varname):
        type_args = type_args_of(self.python_type)
        if not type_args:
            type_args = [..., Any]
        (self.arg_pytypes, self.ret_pytype) = type_args
        if self.arg_pytypes == ...:
            raise CrosshairUnsupported
        self.arg_ch_type = map(crosshair_type_for_python_type, self.arg_pytypes)
        self.ret_ch_type = crosshair_type_for_python_type(self.ret_pytype)
        all_pytypes = tuple(self.arg_pytypes) + (self.ret_pytype,)
        return z3.Function(varname + uniq(),
                           *map(type_to_smt_sort, self.arg_pytypes),
                           type_to_smt_sort(self.ret_pytype))
        #return z3.Array(tuple_sort(map(str, self.arg_pytypes),
        #                           map(type_to_smt_sort, self.arg_pytypes)),
        #                type_to_smt_sort(self.ret_pytype))
    def __call__(self, *args):
        if len(args) != len(self.arg_pytypes):
            raise TypeError('wrong number of arguments')
        args = (coerce_to_smt_var(self.statespace, a)[0] for a in args)
        smt_ret = self.var(*args)
        return self.ret_ch_type(self.statespace, self.ret_pytype, smt_ret)
    def __repr__(self):
        finterp = self.statespace.find_model_value_for_function(self.var)
        if finterp is None:
            return '<any function>' # (z3 model completion will not interpret a function for me currently)
        # 0-arg interpretations seem to be simply values:
        if type(finterp) is not z3.FuncInterp:
            return 'lambda :' + repr(model_value_to_python(finterp))
        if finterp.arity() < 10:
            arg_names = [chr(ord('a') + i) for i in range(finterp.arity())]
        else:
            arg_names = ['a' + str(i + 1) for i in range(finterp.arity())]
        entries = finterp.as_list()
        body = repr(model_value_to_python(entries[-1]))
        for entry in reversed(entries[:-1]):
            conditions = ['{} == {}'.format(arg, repr(model_value_to_python(val)))
                          for (arg, val) in zip(arg_names, entry[:-1])]
            body = '{} if (()) else (())'.format(repr(model_value_to_python(entry[-1])),
                                             ' and '.join(conditions),
                                             body)
        return 'lambda ({}): {}'.format(', '.join(arg_names), body)

class SmtUniformTuple(SmtUniformListOrTuple, collections.abc.Sequence, collections.abc.Hashable):
    def __repr__(self):
        return tuple(self).__repr__()
    def __hash__(self):
        return tuple(self).__hash__()

@functools.total_ordering
class SmtStr(SmtSequence, AbcString):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        assert typ == str
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        self.item_pytype = str
        self.item_ch_type = SmtStr
    def __str__(self):
        return self.statespace.find_model_value(self.var)
    def __copy__(self):
        return SmtStr(self.statespace, str, self.var)
    def __repr__(self):
        return self.statespace.find_model_value(self.var).__repr__()
    def __hash__(self):
        return self.statespace.find_model_value(self.var).__hash__()
    def __add__(self, other):
        return self._binary_op(other, operator.add)
    def __radd__(self, other):
        return self._binary_op(other, lambda a,b: b+a)
    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("can't multiply sequence by non-int")
        ret = ''
        idx = 0
        while idx < other:
            ret = self.__add__(ret)
            idx += 1
        return ret
    def __rmul__(self, other):
        return self.__mul__(other)
    def __lt__(self, other):
        return SmtBool(self.statespace, bool, self.var < smt_coerce(other))
    def __contains__(self, other):
        return SmtBool(self.statespace, bool, z3.Contains(self.var, smt_coerce(other)))
    def __getitem__(self, i):
        (smt_result, is_slice) = self._smt_getitem(i)
        return SmtStr(self.statespace, str, smt_result)
    def find(self, substr, start=None, end=None):
        if end is None:
            return SmtInt(self.statespace, int,
                          z3.IndexOf(self.var, smt_coerce(substr), start or 0))
        else:
            return self.__getitem__(slice(start, end, 1)).index(s)




_CACHED_TYPE_ENUMS:Dict[FrozenSet[type], z3.SortRef] = {}
def get_type_enum(types:FrozenSet[type]) -> z3.SortRef:
    ret = _CACHED_TYPE_ENUMS.get(types)
    if ret is not None:
        return ret
    datatype = z3.Datatype('typechoice('+','.join(sorted(map(str, types)))+')')
    for typ in types:
        datatype.declare(name_of_type(typ))
    datatype = datatype.create()
    _CACHED_TYPE_ENUMS[types] = datatype
    return datatype

class SmtUnion:
    def __init__(self, pytypes:FrozenSet[type]):
        self.pytypes = list(pytypes)
        self.vartype = get_type_enum(pytypes)
    def __call__(self, statespace, pytype, varname):
        var = z3.Const("type("+str(varname)+")", self.vartype)
        for typ in self.pytypes[:-1]:
            if SmtBool(statespace, bool, getattr(self.vartype, 'is_' + name_of_type(typ))(var)).__bool__():
                return proxy_for_type(typ, statespace, varname)
        return proxy_for_type(self.pytypes[-1], statespace, varname)


class ProxiedObject(object):
    def __init__(self, statespace, cls, varname):
        state = {}
        for name, typ in get_type_hints(cls).items():
            origin = getattr(typ, '__origin__', None)
            if origin is Callable:
                continue
            state[name] = proxy_for_type(typ, statespace, varname+'.'+name+uniq())

        object.__setattr__(self, "_obj", state)
        object.__setattr__(self, "_input_obj", state.copy())
        object.__setattr__(self, "_cls", cls)

        # TODO: Implement abstract methods (with uninterpreted functions?)
        for abstract_method in getattr(cls, '__abstractmethods__', ()):
            debug('abstract_method ', abstract_method)

    def __getstate__(self):
        return {k:object.__getattribute__(self, k) for k in 
                ("_cls", "_obj", "_input_obj")}
        
    def __setstate__(self, state):
        object.__setattr__(self, "_obj", state['_obj'].copy())
        object.__setattr__(self, "_input_obj", state['_input_obj'])
        object.__setattr__(self, "_cls", state['_cls'])
        
    def __getattr__(self, name):
        obj = object.__getattribute__(self, "_obj")
        cls = object.__getattribute__(self, "_cls")
        descriptor = obj[name] if name in obj else getattr(cls, name)
        if hasattr(descriptor, '__get__'):
            return descriptor.__get__(self, cls)
        return descriptor
    def __delattr__(self, name):
        obj = object.__getattribute__(self, "_obj")
        cls = object.__getattribute__(self, "_cls")
        descriptor = obj[name] if name in obj else getattr(cls, name)
        if hasattr(descriptor, '__delete__'):
            descriptor.__delete__(self)
        else:
            del obj[name]
    def __setattr__(self, name, value):
        obj = object.__getattribute__(self, "_obj")
        cls = object.__getattribute__(self, "_cls")
        descriptor = obj[name] if name in obj else getattr(cls, name)
        if hasattr(descriptor, '__set__'):
            descriptor.__set__(self, value)
        else:
            obj[name] = value
    
    _special_names = [
        '__bool__', '__str__', '__repr__',
        '__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__', 
        '__contains__', '__delitem__', '__delslice__', '__div__', '__divmod__', 
        '__eq__', '__float__', '__floordiv__', '__ge__', '__getitem__', 
        '__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__',
        '__idiv__', '__idivmod__', '__ifloordiv__', '__ilshift__', '__imod__', 
        '__imul__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', 
        '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', 
        '__long__', '__lshift__', '__lt__', '__mod__', '__mul__', '__matmul__',
        '__ne__', '__neg__', '__oct__', '__or__', '__pos__', '__pow__', '__radd__', 
        '__rand__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__', 
        '__repr__', '__reversed__', '__rfloorfiv__', '__rlshift__', '__rmod__', 
        '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', 
        '__rtruediv__', '__rxor__', '__setitem__', '__setslice__', '__sub__', 
        '__truediv__', '__xor__', 'next',
    ]
    
    @classmethod
    def _create_class_proxy(cls, proxied_class):
        """creates a proxy for the given class"""
        cls_name = name_of_type(proxied_class)
        def make_method(name: str) -> Optional[Callable]:
            fn = getattr(proxied_class, name)
            if fn is None: # sometimes used to demonstrate unsupported
                return None
            fn_name = fn.__name__
            is_wrapper_descriptor = type(fn) is type(tuple.__iter__) and fn is not getattr(object, name, None)
            def method(self, *args, **kw):
                debug('Calling', fn_name, 'on proxy of', cls_name)
                if is_wrapper_descriptor:
                    raise CrosshairUnsupported('wrapper descriptor '+fn.__name__) # TODO realize a concrete impl and call on that
                return fn(self, *args, **kw)
            functools.update_wrapper(method, fn)
            return method
        
        namespace = {}
        for name in cls._special_names:
            if not hasattr(proxied_class, name):
                continue
            namespace[name] = make_method(name)

        return type(cls_name + "_proxy", (cls,), namespace)

    def __new__(cls, statespace, proxied_class, varname):
        try:
            proxyclass = _PYTYPE_TO_WRAPPER_TYPE[proxied_class]
        except KeyError:
            proxyclass = cls._create_class_proxy(proxied_class)

        proxy = object.__new__(proxyclass)
        return proxy

def make_raiser(exc, *a) -> Callable:
    def do_raise(*ra, **rkw) -> NoReturn:
        raise exc(*a)
    return do_raise

_SIMPLE_PROXIES: MutableMapping[object, Callable] = {
    complex: lambda p: complex(p(float), p(float)),
    
    # if the target is a non-generic class, create it directly (but possibly with proxy constructor arguments?)
    Any: lambda p: p(int),
    Type: lambda p, t=Any: p(Callable[[...],t]),
    NoReturn: make_raiser(IgnoreAttempt, 'Attempted to short circuit a NoReturn function'), # type: ignore
    #Optional, (elsewhere)
    #Callable, (elsewhere)
    #ClassVar, (elsewhere)
    
    #AsyncContextManager: lambda p: p(contextlib.AbstractAsyncContextManager),
    #AsyncGenerator: ,
    #AsyncIterable,
    #AsyncIterator,
    #Awaitable,

    ContextManager: lambda p: p(contextlib.AbstractContextManager), # type: ignore
    #Coroutine: (handled via typeshed)
    #Generator: (handled via typeshed)
    
    #FrozenSet: (elsewhere)
    AbstractSet: lambda p, t=Any: p(FrozenSet[t]), # type: ignore
    
    #Dict: (elsewhere)
    # NOTE: could be symbolic (but note the default_factory is changable/stateful):
    DefaultDict: lambda p, kt=Any, vt=Any: collections.DeafultDict(p(Callable[[], vt]), p(Dict[kt, vt])), # type: ignore
    typing.ChainMap: lambda p, kt=Any, vt=Any: collections.ChainMap(*p(Tuple[Dict[kt, vt], ...])), # type: ignore
    Mapping: lambda p, t=Any: p(Dict[t]), # type: ignore
    MutableMapping: lambda p, t=Any: p(Dict[t]), # type: ignore
    typing.OrderedDict: lambda p, kt=Any, vt=Any: collections.OrderedDict(p(Dict[kt, vt])), # type: ignore
    Counter: lambda p, t=Any: collections.Counter(p(Dict[t, int])), # type: ignore
    #MappingView: (as instantiated origin)
    ItemsView: lambda p, kt=Any, vt=Any: p(Set[Tuple[kt, vt]]), # type: ignore
    KeysView: lambda p, t=Any: p(Set[t]), # type: ignore
    ValuesView: lambda p, t=Any: p(Set[t]), # type: ignore
    
    Container: lambda p, t=Any: p(Tuple[t, ...]),
    Collection: lambda p, t=Any: p(Tuple[t, ...]),
    Deque: lambda p, t=Any: collections.deque(p(Tuple[t, ...]), p(Optional[int])),
    Iterable: lambda p, t=Any: p(Tuple[t, ...]),
    Iterator: lambda p, t=Any: iter(p(Iterable[t])), # type: ignore
    #List: (elsewhere)
    
    MutableSequence: lambda p, t=Any: p(List[t]), # type: ignore
    Reversible: lambda p, t=Any: p(Tuple[t, ...]),
    Sequence: lambda p, t=Any: p(Tuple[t, ...]),
    Sized: lambda p, t=Any: p(Tuple[t, ...]),
    NamedTuple: lambda p, *t: p(Tuple.__getitem__(tuple(t))),
    
    #Set, (elsewhere)
    MutableSet: lambda p, t=Any: p(Set[t]), # type: ignore
    
    typing.Pattern: lambda p, t=None: p(re.compile), # type: ignore
    typing.Match: lambda p, t=None: p(re.match), # type: ignore
    
    # Text: (elsewhere - identical to str)
    ByteString: lambda p: bytes(b % 256 for b in p(List[int])),
    #AnyStr,  (it's a type var)
    typing.BinaryIO: io.BytesIO,
    typing.IO: lambda p: io.StringIO(p(str)),
    typing.TextIO: lambda p: io.StringIO(p(str)),
    
    Hashable: lambda p: p(int),
    SupportsAbs: lambda p: p(int),
    SupportsFloat: lambda p: p(float),
    SupportsInt: lambda p: p(int),
    SupportsRound: lambda p: p(float),
    SupportsBytes: lambda p: p(ByteString),
    SupportsComplex: lambda p: p(complex),
}

_SIMPLE_PROXIES = dict((origin_of(k), v) for (k, v) in _SIMPLE_PROXIES.items()) # type: ignore

def proxy_class_as_concrete(typ: Type, statespace: StateSpace, varname: str) -> object:
    data_members = get_type_hints(typ)
    if not data_members:
        return _MISSING
    args = {k: proxy_for_type(t, statespace, varname + '.' + k)
            for (k, t) in data_members.items()}
    init_signature = inspect.signature(typ.__init__)
    try:
        return typ(**args)
    except:
        pass
    try:
        obj = typ()
        for (k, v) in args:
            setattr(obj, k, v)
        return obj
    except:
        pass
    return _MISSING

def proxy_for_type(typ: Type, statespace: StateSpace, varname: str) -> object:
    typ = normalize_pytype(typ)
    origin = getattr(typ, '__origin__', None)
    # special cases
    if origin is tuple:
        if len(typ.__args__) == 2 and typ.__args__[1] == ...:
            return SmtUniformTuple(statespace, typ, varname)
        else:
            return tuple(proxy_for_type(t, statespace, varname +'[' + str(idx) + ']')
                         for (idx, t) in enumerate(typ.__args__))
    elif isinstance(typ, type) and issubclass(typ, enum.Enum):
        enum_values = list(typ) # type:ignore
        for enum_value in enum_values[:-1]:
            if smt_fork(statespace):
                statespace.model_additions[varname] = enum_value
                return enum_value
        statespace.model_additions[varname] = enum_values[-1]
        return enum_values[-1]
    elif typ is type(None):
        return None
    proxy_factory = _SIMPLE_PROXIES.get(origin_of(typ))
    if proxy_factory:
        recursive_proxy_factory = lambda t: proxy_for_type(t, statespace, varname+uniq())
        return proxy_factory(recursive_proxy_factory, *type_args_of(typ))
    Typ = crosshair_type_for_python_type(typ)
    if Typ is not None:
        return Typ(statespace, typ, varname)
    # if the class has data members, we attempt to create a concrete instance with
    # symbolic members; otherwise, we'll create a ProxiedObject that emulates it.
    ret = proxy_class_as_concrete(typ, statespace, varname)
    if ret is _MISSING:
        ret = ProxiedObject(statespace, typ, varname)
    class_conditions = get_class_conditions(typ)
    # symbolic custom classes may assume their invariants:
    if class_conditions is not None:
        for inv_condition in class_conditions.inv:
            if inv_condition.expr is None:
                continue
            isok = False
            with ExceptionFilter() as efilter:
                isok = eval(inv_condition.expr, {'self': ret})
            if efilter.user_exc:
                debug('Could not assume invaniant', inv_condition.expr_source, 'on proxy of', typ,
                      ' because it raised: ', str(efilter.user_exc[0]))
                # if the invarants are messed up enough to be rasing exceptions, don't bother:
                return ret
            elif efilter.ignore or not isok:
                raise IgnoreAttempt('Class proxy did not meet invariant ', inv_condition.expr_source)
    return ret

def gen_args(sig: inspect.Signature, statespace:StateSpace) -> inspect.BoundArguments:
    args = sig.bind_partial()
    for param in sig.parameters.values():
        smt_name = param.name + uniq()
        has_annotation = (param.annotation != inspect.Parameter.empty)
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            if has_annotation:
                varargs_type = List[param.annotation] # type: ignore
                value = proxy_for_type(varargs_type, statespace, smt_name)
            else:
                value = proxy_for_type(List[Any], statespace, smt_name)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            if has_annotation:
                varargs_type = Dict[str, param.annotation] # type: ignore
                value = proxy_for_type(varargs_type, statespace, smt_name)
                # Using ** on a dict requires concrete string keys. Force
                # instiantiation of keys here:
                value = {k.__str__():v for (k,v) in value.items()}
            else:
                value = proxy_for_type(Dict[str, Any], statespace, smt_name)
        else:
            if has_annotation:
                value = proxy_for_type(param.annotation, statespace, smt_name)
            else:
                value = proxy_for_type(Any, statespace, smt_name)
        debug('created proxy for', param.name, 'as type:', type(value))
        args.arguments[param.name] = value
    return args


@functools.total_ordering
class MessageType(enum.Enum):
    CANNOT_CONFIRM = 'cannot_confirm'
    PRE_UNSAT = 'pre_unsat'
    POST_ERR = 'post_err'
    EXEC_ERR = 'exec_err'
    POST_FAIL = 'post_fail'
    SYNTAX_ERR = 'syntax_err'
    def __lt__(self, other):
        return self._order[self] < self._order[other]
MessageType._order = { # type: ignore
    # This is the order that messages override each other (for the same source file line)
    MessageType.CANNOT_CONFIRM: 0,
    MessageType.PRE_UNSAT: 1,
    MessageType.POST_ERR: 2,
    MessageType.EXEC_ERR: 3,
    MessageType.POST_FAIL: 4,
    MessageType.SYNTAX_ERR: 5,
}

@dataclass(frozen=True)
class AnalysisMessage:
    state: MessageType
    message: str
    filename: str
    line: int
    column: int
    traceback: str

class MessageCollector:
    def __init__(self):
        self.by_pos = {}
    def extend(self, messages:Iterable[AnalysisMessage]) -> None:
        for message in messages:
            self.append(message)
    def append(self, message: AnalysisMessage) -> None:
        key = (message.filename, message.line, message.column)
        if key in self.by_pos:
            self.by_pos[key] = max(self.by_pos[key], message, key=lambda m:m.state)
        else:
            self.by_pos[key] = message
    def get(self) -> List[AnalysisMessage]:
        return [m for (k,m) in sorted(self.by_pos.items())]

@functools.total_ordering
class VerificationStatus(enum.Enum):
    REFUTED = 0
    UNKNOWN = 1
    CONFIRMED = 2
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

@dataclass
class AnalysisOptions:
    use_called_conditions: bool = True
    timeout: float = 30.0
    deadline: float = float('NaN')
    per_path_timeout: float = 2.0

_DEFAULT_OPTIONS = AnalysisOptions()

def analyze_module(module:types.ModuleType, options:AnalysisOptions) -> List[AnalysisMessage]:
    debug('Analyzing module ', module)
    messages = MessageCollector()
    for (name, member) in inspect.getmembers(module):
        if inspect.isclass(member) and member.__module__ == module.__name__:
            messages.extend(analyze_class(member, options))
        elif inspect.isfunction(member) and member.__module__ == module.__name__:
            messages.extend(analyze(member, options))
    message_list = messages.get()
    debug('Module', module.__name__, 'has', len(message_list), 'messages')
    return message_list

def analyze_class(cls:type, options:AnalysisOptions=_DEFAULT_OPTIONS) -> List[AnalysisMessage]:
    messages = MessageCollector()
    class_conditions = get_class_conditions(cls)
    for method, conditions in class_conditions.methods:
        if conditions.has_any():
            messages.extend(analyze(method,
                                    conditions=conditions,
                                    options=options,
                                    self_type=cls))
        
    return messages.get()

_EMULATION_TIMEOUT_FRACTION = 0.2
def analyze(fn:Callable,
            options:AnalysisOptions=_DEFAULT_OPTIONS,
            conditions:Optional[Conditions]=None,
            self_type:Optional[type]=None) -> List[AnalysisMessage]:
    debug('Analyzing ', fn.__name__)
    all_messages = MessageCollector()
    conditions = conditions or get_fn_conditions(fn, self_type=self_type)

    for cond in conditions.uncompilable_conditions():
        all_messages.append(AnalysisMessage(MessageType.SYNTAX_ERR, str(cond.compile_err), cond.filename, cond.line, 0, ''))
    conditions = conditions.compilable()
    
    for post_condition in conditions.post:
        messages = analyze_post_condition(fn, options, replace(conditions, post=[post_condition]), self_type)
        all_messages.extend(messages)
    return all_messages.get()

def analyze_post_condition(fn:Callable,
                           options:AnalysisOptions,
                           conditions:Conditions,
                           self_type:Optional[type]) -> Sequence[AnalysisMessage]:
    if options.use_called_conditions: 
        options.deadline = time.time() + options.timeout * _EMULATION_TIMEOUT_FRACTION
    else:
        options.deadline = time.time() + options.timeout
    sig = conditions.sig
    
    analysis = analyze_calltree(fn, options, conditions, sig)
    if (options.use_called_conditions and
        analysis.verification_status < VerificationStatus.CONFIRMED):
        debug('Reattempting analysis without short circuiting')
        options = replace(options,
                          use_called_conditions=False,
                          deadline=time.time() + options.timeout * (1.0 - _EMULATION_TIMEOUT_FRACTION))
        analysis = analyze_calltree(fn, options, conditions, sig)

    (condition,) = conditions.post
    if analysis.verification_status is VerificationStatus.UNKNOWN:
        addl_ctx = ' ' + condition.addl_context if condition.addl_context else ''
        message = 'I cannot confirm this' + addl_ctx
        analysis.messages = [AnalysisMessage(MessageType.CANNOT_CONFIRM, message,
                                             condition.filename, condition.line, 0, '')]
        
    return analysis.messages

def forget_contents(value:object, space:StateSpace):
    if isinstance(value, SmtBackedValue):
        clean_smt = type(value)(space, value.python_type, str(value.var)+uniq())
        value.var = clean_smt.var
    elif isinstance(value, ProxiedObject):
        # TODO test this path
        obj = object.__getattribute__(value, '_obj')
        cls = object.__getattribute__(value, '_cls')
        clean = proxy_for_type(cls, space, uniq())
        clean_obj = object.__getattribute__(clean, '_obj')
        for key, val in obj.items():
            obj[key] = clean_obj[key]
    else:
        for subvalue in value.__dict__.values():
            forget_contents(subvalue, space)

class ShortCircuitingContext:
    engaged = False
    intercepted = False
    def __init__(self, space:StateSpace):
        self.space = space
    def __enter__(self):
        assert not self.engaged
        self.engaged = True
    def __exit__(self, exc_type, exc_value, tb):
        assert self.engaged
        self.engaged = False
    def make_interceptor(self, original) -> Callable:
        subconditions = get_fn_conditions(original)
        sig = subconditions.sig
        def wrapper(*a:object, **kw:Dict[str, object]) -> object:
            #debug('intercept wrapper ', original, self.engaged)
            if not self.engaged or self.space.running_framework_code:
                return original(*a, **kw)
            try:
                self.engaged = False
                debug('intercepted a call to ', original, typing_inspect.get_parameters(sig.return_annotation))
                self.intercepted = True
                return_type = sig.return_annotation

                # Deduce type vars if necessary
                if len(typing_inspect.get_parameters(sig.return_annotation)) > 0 or typing_inspect.is_typevar(sig.return_annotation):
                    typevar_bindings :typing.ChainMap[object, type] = collections.ChainMap()
                    bound = sig.bind(*a, **kw)
                    bound.apply_defaults()
                    for param in sig.parameters.values():
                        argval = bound.arguments[param.name]
                        value_type = argval.python_type if isinstance(argval, SmtBackedValue) else type(argval)
                        if not dynamic_typing.unify(value_type, param.annotation, typevar_bindings):
                            debug('aborting intercept due to signature unification failure')
                            return original(*a, **kw)
                    return_type = dynamic_typing.realize(sig.return_annotation, typevar_bindings)
                    debug('Deduced short circuit return type was ', return_type)

                # adjust arguments that may have been mutated
                if subconditions.mutable_args:
                    bound = sig.bind(*a, **kw)
                    for mutated_arg in subconditions.mutable_args:
                        forget_contents(bound.arguments[mutated_arg], self.space)

                if return_type is type(None):
                    return None
                # note that the enforcement wrapper ensures postconditions for us, so we
                # can just return a free variable here.
                return proxy_for_type(return_type, self.space, 'proxyreturn' + uniq())
            finally:
                self.engaged = True
        functools.update_wrapper(wrapper, original)
        return wrapper

@dataclass
class CallAnalysis:
    verification_status: Optional[VerificationStatus] = None
    messages: Sequence[AnalysisMessage] = ()
    failing_precondition: Optional[ConditionExpr] = None

@dataclass
class CallTreeAnalysis:
    messages: Sequence[AnalysisMessage]
    verification_status: VerificationStatus
    num_confirmed_paths: int = 0

def analyze_calltree(fn:Callable,
                     options:AnalysisOptions,
                     conditions:Conditions,
                     sig:inspect.Signature) -> CallTreeAnalysis:
    debug('Begin analyze calltree ', fn.__name__, ' short circuit=', options.use_called_conditions)

    worst_verification_status = VerificationStatus.CONFIRMED
    all_messages = MessageCollector()
    search_history = SearchTreeNode()
    space_exhausted = False
    failing_precondition: Optional[ConditionExpr] = conditions.pre[0] if conditions.pre else None
    num_confirmed_paths = 0
    for i in range(1000):
        reset_for_iteration()
        start = time.time()
        if start > options.deadline:
            break
        debug(' ** Iteration ', i)
        space = StateSpace(search_history, execution_deadline = start + options.per_path_timeout)
        short_circuit = ShortCircuitingContext(space)
        try:
            # TODO try to patch outside the search loop
            envs = [fn_globals(fn), contracted_builtins.__dict__]
            interceptor = (short_circuit.make_interceptor if options.use_called_conditions else lambda f:f)
            with EnforcedConditions(*envs, interceptor=interceptor):
                bound_args = gen_args(sig, space)
                original_builtins = builtins.__dict__.copy()
                try:
                    builtins.__dict__.update([(k,v) for (k,v) in contracted_builtins.__dict__.items() if not k.startswith('_')])
                    call_analysis = attempt_call(conditions, space, fn, bound_args, short_circuit)
                finally:
                    builtins.__dict__.update(original_builtins)
                if failing_precondition is not None:
                    cur_precondition = call_analysis.failing_precondition
                    if cur_precondition is None:
                        if call_analysis.verification_status is not None:
                            # We escaped the all the pre conditions on this try:
                            failing_precondition = None
                    elif cur_precondition.line > failing_precondition.line:
                        failing_precondition = cur_precondition
        
        except (UnknownSatisfiability, CrosshairUnsupported):
            call_analysis = CallAnalysis(VerificationStatus.UNKNOWN)
        except IgnoreAttempt:
            call_analysis = CallAnalysis()
        exhausted = space.check_exhausted()
        debug('iter complete', call_analysis.verification_status.name if call_analysis.verification_status else 'None',
              ' (previous worst:', worst_verification_status.name, ')')
        if call_analysis.verification_status is not None:
            if call_analysis.verification_status == VerificationStatus.CONFIRMED:
                num_confirmed_paths += 1
            else:
                worst_verification_status = min(call_analysis.verification_status, worst_verification_status)
            all_messages.extend(call_analysis.messages)
        if exhausted:
            # we've searched every path
            space_exhausted = True
            break
        if worst_verification_status <= VerificationStatus.REFUTED:
            break
    if not space_exhausted:
        worst_verification_status = min(VerificationStatus.UNKNOWN, worst_verification_status)
    if failing_precondition:
        assert num_confirmed_paths == 0
        addl_ctx = ' ' + failing_precondition.addl_context if failing_precondition.addl_context else ''
        message = 'Unable to meet precondition' + addl_ctx
        all_messages.extend([AnalysisMessage(MessageType.PRE_UNSAT, message,
                                             failing_precondition.filename, failing_precondition.line, 0, '')])
        worst_verification_status = VerificationStatus.REFUTED

    debug(('Exhausted' if space_exhausted else 'Aborted') +' calltree search with',worst_verification_status.name,'. Number of iterations: ', i+1)
    return CallTreeAnalysis(messages = all_messages.get(),
                            verification_status = worst_verification_status,
                            num_confirmed_paths = num_confirmed_paths)

def python_string_for_evaluated(expr:z3.ExprRef)->str:
    return str(expr)

def get_input_description(statespace:StateSpace,
                          bound_args:inspect.BoundArguments,
                          addl_context:str = '') -> str:
    messages:List[str] = []
    for argname, argval in bound_args.arguments.items():
        messages.append(argname + ' = ' + repr(argval))
    if addl_context:
        return addl_context + ' with ' + ' and '.join(messages)
    elif messages:
        return 'when ' + ' and '.join(messages)
    else:
        return 'for any input'

def shallow_eq(old_val:object, new_val:object) -> bool:
    if old_val is new_val:
        return True
    if isinstance(old_val, ProxiedObject) and isinstance(new_val, ProxiedObject):
        od, nd = object.__getattribute__(old_val,'_obj'), object.__getattribute__(new_val,'_obj')
        # TODO: is this really shallow equality?
        for key in set(od.keys()).union(nd.keys()):
            if od.get(key,None) is not nd.get(key,None):
                return False
        return True
    elif isinstance(old_val, SmtBackedValue) and isinstance(new_val, SmtBackedValue) and old_val.var is new_val.var:
        return True
    elif isinstance(old_val, Iterable) and isinstance(new_val, Iterable):
        if isinstance(old_val, Sized) and isinstance(new_val, Sized):
            if len(old_val) != len(new_val):
                return False
        for (o,n) in itertools.zip_longest(old_val, new_val, fillvalue=object()):
            if o is not n:
                return False
        return True
    elif hasattr(old_val, '__dict__') and hasattr(new_val, '__dict__'):
        return old_val.__dict__ == new_val.__dict__
    else:
        return old_val == new_val


def rewire_inputs(fn:Callable, env):
    '''
    Turns function arguments into position parameters.
    Will this mess up line numbers?
    Makes it harer to output a repro string? like foo("foo", k=[])
    '''
    fn_source = inspect.getsource(fn)
    fndef = cast(ast.Module, ast.parse(fn_source)).body[0]
    args = cast(ast.FunctionDef, fndef).args
    allargs = args.args + args.kwonlyargs + ([args.vararg] if args.vararg else []) + ([args.kwarg] if args.kwarg else [])
    arg_names = [a.arg for a in allargs]
    
def attempt_call(conditions:Conditions,
                 statespace:StateSpace,
                 fn:Callable,
                 bound_args: inspect.BoundArguments,
                 short_circuit:ShortCircuitingContext) -> CallAnalysis:
    original_args = copy.copy(bound_args) # TODO shallow copy each param
    original_args.arguments = copy.copy(bound_args.arguments)
    for (k, v) in bound_args.arguments.items():
        try:
            vcopy = copy.copy(v)
        except:
            debug('Failed to copy input argument', k, 'of type', type(v))
            vcopy = v
        original_args.arguments[k] = vcopy

    raises = conditions.raises
    for precondition in conditions.pre:
        with ExceptionFilter() as efilter:
            eval_vars = {**fn_globals(fn), **bound_args.arguments}
            with short_circuit:
                assert precondition.expr is not None
                precondition_ok = eval(precondition.expr, eval_vars)
            if not precondition_ok:
                debug('Failed to meet precondition', precondition.expr_source)
                return CallAnalysis(failing_precondition=precondition)
        if efilter.ignore:
            return CallAnalysis()
        elif efilter.user_exc is not None:
            debug('Exception attempting to meet precondition', precondition.expr_source,':', efilter.user_exc[1].format())
            return CallAnalysis(failing_precondition=precondition)

    with ExceptionFilter() as efilter:
        a, kw = bound_args.args, bound_args.kwargs
        with short_circuit:
            assert not statespace.running_framework_code
            __return__ = fn(*a, **kw)
        lcls = {**bound_args.arguments, '__return__':__return__, fn.__name__:fn}
    if efilter.ignore:
        return CallAnalysis()
    elif efilter.user_exc is not None:
        (e, tb) = efilter.user_exc
        detail = name_of_type(type(e)) + ': ' + str(e) + ' ' + get_input_description(statespace, original_args)
        frame = frame_summary_for_fn(tb, fn)
        return CallAnalysis(VerificationStatus.REFUTED,
                            [AnalysisMessage(MessageType.EXEC_ERR, detail, frame.filename, frame.lineno, 0, ''.join(tb.format()))])

    for argname, argval in bound_args.arguments.items():
        if argname not in conditions.mutable_args:
            old_val, new_val = original_args.arguments[argname], argval
            if not shallow_eq(old_val, new_val):
                detail = 'Argument "{}" is not marked as mutable, but changed from {} to {}'.format(argname, old_val, new_val)
                return CallAnalysis(VerificationStatus.REFUTED,
                                    [AnalysisMessage(MessageType.POST_ERR, detail, fn.__code__.co_filename, fn.__code__.co_firstlineno, 0, '')])
    
    (post_condition,) = conditions.post
    with ExceptionFilter() as efilter:
        eval_vars = {**fn_globals(fn), **lcls}
        with short_circuit:
            assert post_condition.expr is not None
            isok = eval(post_condition.expr, eval_vars)
    if efilter.ignore:
        return CallAnalysis()
    elif efilter.user_exc is not None:
        (e, tb) = efilter.user_exc
        detail = str(e) + ' ' + get_input_description(statespace, original_args, post_condition.addl_context)
        failures=[AnalysisMessage(MessageType.POST_ERR, detail, post_condition.filename, post_condition.line, 0, ''.join(tb.format()))]
        return CallAnalysis(VerificationStatus.REFUTED, failures)
    if isok:
        return CallAnalysis(VerificationStatus.CONFIRMED)
    else:
        detail = 'false ' + get_input_description(statespace, original_args, post_condition.addl_context)
        failures = [AnalysisMessage(MessageType.POST_FAIL, detail, post_condition.filename, post_condition.line, 0, '')]
        return CallAnalysis(VerificationStatus.REFUTED, failures)

_PYTYPE_TO_WRAPPER_TYPE = {
    type(None): (lambda *a: None),
    bool: SmtBool,
    int: SmtInt,
    float: SmtFloat,
    str: SmtStr,
    list: SmtUniformList,
    dict: SmtDict,
    set: SmtMutableSet,
    frozenset: SmtFrozenSet,
}

# Type ignore pending https://github.com/python/mypy/issues/6864
_PYTYPE_TO_WRAPPER_TYPE[collections.abc.Callable] = SmtCallable # type:ignore

_WRAPPER_TYPE_TO_PYTYPE = dict((v,k) for (k,v) in _PYTYPE_TO_WRAPPER_TYPE.items())


# Executable concerns

import argparse
import importlib
import importlib.util

def command_line_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CrossHair Analysis Tool')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--per_path_timeout', type=float)
    parser.add_argument('--per_condition_timeout', type=float)
    subparsers = parser.add_subparsers(help='sub-command help', dest='action')
    check_parser = subparsers.add_parser('check', help='Analyze one or more files')
    check_parser.add_argument('files', metavar='F', type=str, nargs='+',
                              help='files or directories to analyze')
    watch_parser = subparsers.add_parser('watch', help='Continuously watch and analyze files')
    watch_parser.add_argument('files', metavar='F', type=str, nargs='+',
                              help='files or directories to analyze')
    return parser
    
def process_level_options(command_line_args: argparse.Namespace) -> AnalysisOptions:
    options = AnalysisOptions()
    for optname in ('per_path_timeout', 'per_condition_timeout'):
        arg_val = getattr(command_line_args, optname)
        if arg_val is not None:
            setattr(options, optname, arg_val)
    return options

def module_for_file(filepath:str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location('crosshair.examples.tic_tac_toe', filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod) # type:ignore
    return mod

if __name__ == '__main__':
    any_errors = False
    args = command_line_parser().parse_args()
    set_debug(args.verbose)
    options = process_level_options(args)
    for name in args.files:
        if '=' in name:
            continue
        if name.endswith('.py'):
            _, name = extract_module_from_file(name)
        module = importlib.import_module(name)
        debug('Analyzing module ', module.__name__)
        for message in analyze_module(module, options):
            if message.state == MessageType.CANNOT_CONFIRM:
                continue
            desc = message.message
            if message.state == MessageType.POST_ERR:
                desc = 'Error while evaluating post condition: ' + desc
            debug(message.traceback)
            print('{}:{}:{}:{}:{}'.format('error', message.filename, message.line, message.column, desc))
            any_errors = True
    sys.exit(1 if any_errors else 0)
