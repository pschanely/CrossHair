# TODO: use object() for Any?
# TODO: create a generic force() function to fully realize model
# TODO: implement enums (as immediate case split)

from dataclasses import dataclass, replace
from typing import *
from crosshair.typed_inspect import signature
import typing_inspect
from crosshair.condition_parser import get_fn_conditions, get_class_conditions, ConditionExpr, Conditions
from crosshair.enforce import EnforcedConditions, PostconditionFailed
import ast
import builtins
import collections
import copy
import enum
import inspect
import itertools
import functools
import operator
import random
import sys
import time
import traceback
import types

import z3  # type: ignore

_CHOICE_RANDOMIZATION = 4221242075

_UNIQ = 0
def uniq():
    global _UNIQ
    _UNIQ += 1
    if _UNIQ >= 1000000:
        raise Exception('Exhausted var space')
    return '{:06d}'.format(_UNIQ)

class SearchTreeNode:
    exhausted : bool = False
    positive :Optional['SearchTreeNode'] = None
    negative :Optional['SearchTreeNode'] = None
    def choose(self, seed) -> Tuple[bool, 'SearchTreeNode']:
        positive_ok = self.positive is None or not self.positive.exhausted
        negative_ok = self.negative is None or not self.negative.exhausted
        if positive_ok and negative_ok:
            choice = random.randint(0,1)#(seed % 2 == 0)
        else:
            choice = positive_ok
        if choice:
            if self.positive is None:
                self.positive = SearchTreeNode()
            return (True, self.positive)
        else:
            if self.negative is None:
                self.negative = SearchTreeNode()
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
    pass
    
class UnknownSatisfiability(Exception):
    pass
    
class StateSpace:
    def __init__(self, seed:int, previous_searches:SearchTreeNode):
        #self.solver = z3.Solver()
        #self.solver = z3.Then('simplify','smt', 'qfnra-nlsat').solver()
        #print(' -- reset -- ')
        _HEAP[:] = []
        self.solver = z3.OrElse('smt', 'qfnra-nlsat').solver()
        self.solver.set(mbqi=True)
        self.seed = seed ^ _CHOICE_RANDOMIZATION
        self.search_position = previous_searches
        self.choices_made :List[SearchTreeNode] = []
        self.model_additions :Mapping[str,object] = {}

    def add(self, expr:z3.ExprRef) -> None:
        #print('committed to ', expr)
        self.solver.add(expr)

    def check(self, expr:z3.ExprRef) -> z3.CheckSatResult:
        solver = self.solver
        solver.push()
        solver.add(expr)
        #print('CHECK ? ' + str(solver))
        ret = solver.check()
        #print('CHECK => ' + str(ret))
        if ret not in (z3.sat, z3.unsat):
            #alt_solver z3.Then('qfnra-nlsat').solver()
            print(' -- UNKOWN SAT --')
            raise UnknownSatisfiability(str(ret)+': '+str(solver))
        solver.pop()
        return ret

    def model(self):
        if len(self.solver.assertions()) == 0:
            return []
        else:
            # sometimes we introduce new variables inline
            self.solver.check()
            return self.solver.model()
    
    def choose(self, expr:z3.ExprRef) -> bool:
        choose_true = self.make_choice()
        expr = expr if choose_true else z3.Not(expr)
        #print('CHOOSE', expr)
        self.add(expr)
        return choose_true

    def find_model_value(self, expr:z3.ExprRef) -> object:
        if self.solver.check() != z3.sat:
            raise Exception('model unexpectedly became unsatisfiable')
        result = self.solver.model().evaluate(expr, model_completion=True)
        return ast.literal_eval(repr(result))

    def make_choice(self) -> bool:
        (choose_true, new_search_node) = self.search_position.choose(self.seed)
        self.choices_made.append(self.search_position)
        self.search_position = new_search_node
        self.seed = self.seed // 2
        return choose_true
    
    def check_exhausted(self) -> bool:
        return SearchTreeNode.check_exhausted(self.choices_made, self.search_position)


def could_be_instanceof(v:object, typ:Type) -> bool:
    ret = isinstance(v, origin_of(typ))
    return ret

HeapRef = z3.DeclareSort('HeapRef')
_HEAP:List[Tuple[z3.ExprRef, object]] = []
def find_key_in_heap(space:StateSpace, ref:z3.ExprRef, typ:Type) -> object:
    global _HEAP
    for (k, v) in _HEAP:
        if not could_be_instanceof(v, typ):
            continue
        if smt_fork(space, k == ref):
            return v
    ret = proxy_for_type(typ, space, 'heapref'+str(typ)+uniq())
    _HEAP.append((ref, ret))
    return ret

def find_val_in_heap(space:StateSpace, value:object) -> z3.ExprRef:
    for (k, v) in _HEAP:
        if v is value:
            return k
    ref = z3.Const('heap'+str(value)+uniq(), HeapRef)
    for (k, _) in _HEAP:
        space.add(ref != k)
    _HEAP.append((ref, value))
    return ref
            
def origin_of(typ:Type) -> Type:
    if hasattr(typ, '__origin__'):
        return typ.__origin__
    return typ

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
    
    

def type_to_smt_sort(t: Type):
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
    origin = origin_of(typ)
    if origin is Union:
        return SmtUnion(frozenset(typ.__args__))
    Typ = _PYTYPE_TO_WRAPPER_TYPE.get(origin)
    if Typ:
        return Typ
    '''
    def heaper(space:StateSpace, typ2:type, var:Union[str, z3.ExprRef]) -> object:
        assert isinstance(var, str)
        assert typ == typ2
        smt_ref = z3.Const(str(var), HeapRef)
        return find_in_heap(space, smt_ref, typ, str(var))
    return heaper
    '''
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
    #raise Exception('Unable to coerce literal '+repr(v)+' into smt var')

def coerce_to_ch_value(v:Any, statespace:StateSpace) -> object:
    (smt_var, py_type) = coerce_to_smt_var(statespace, v)
    Typ = crosshair_type_for_python_type(py_type)
    if Typ is None:
        raise Exception('Unable to get ch type from python type: '+str(py_type))
    return Typ(statespace, py_type, smt_var)

def smt_fork(space:StateSpace, expr:z3.ExprRef):
    return SmtBool(space, bool, expr).__bool__()

class SmtBackedValue:
    def __init__(self, statespace:StateSpace, typ: Type, smtvar:object):
        self.statespace = statespace
        if isinstance(smtvar, str):
            self.var = self.__init_var__(typ, smtvar)
            self.python_type = typ
        else:
            self.var = smtvar
            self.python_type = typ
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
    def __req__(self, other):
        return coerce_to_ch_value(other, self.statespace).__eq__(self)
    def _binary_op(self, other, op):
        left, right = self.var, coerce_to_smt_var(self.statespace, other)[0]
        return self.__class__(self.statespace, self.python_type, op(left, right))
    def _cmp_op(self, other, op):
        return SmtBool(self.statespace, bool, op(self.var, smt_coerce(other)))
    def _unary_op(self, op):
        return self.__class__(self.statespace, self.python_type, op(self.var))

class SmtNumberAble(SmtBackedValue):
    def _numeric_op(self, other, op):
        l_var, lpytype = self.var, self.python_type
        r_var, rpytype = coerce_to_smt_var(self.statespace, other)
        promotion_fn = _NUMERIC_PROMOTION_FNS.get((lpytype, rpytype))
        if not promotion_fn:
            return NotImplemented
        l_var, r_var, common_pytype = promotion_fn(l_var, r_var)
        cls = _PYTYPE_TO_WRAPPER_TYPE[common_pytype]
        return cls(self.statespace, common_pytype, op(l_var, r_var))

    # '__pos__',
    # '__abs__',
    # '__invert__',
    # '__round__',
    # '__ceil__',
    # '__floor__',
    # '__trunc__',
    def __ne__(self, other):
        return self._cmp_op(other, operator.ne)
    def __lt__(self, other):
        return self._cmp_op(other, operator.lt)
    def __gt__(self, other):
        return self._cmp_op(other, operator.gt)
    def __le__(self, other):
        return self._cmp_op(other, operator.le)
    def __ge__(self, other):
        return self._cmp_op(other, operator.ge)

    def __hash__(self):
        return self.__index__()

    def __neg__(self):
        return self._unary_op(operator.neg)
    
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
        return repr(self.__bool__())

    def __xor__(self, other):
        return self._binary_op(other, z3.Xor)
    def __not__(self):
        return self._unary_op(z3.Not)

    def __bool__(self):
        could_be_true = (self.statespace.check(self.var) == z3.sat)
        could_be_false = (self.statespace.check(z3.Not(self.var)) == z3.sat)
        if (not could_be_true) and (not could_be_false):
            raise Exception('Reached impossible code path')
        if could_be_true and could_be_false:
            return self.statespace.choose(self.var)
        elif not could_be_true:
            return False
        else:
            return True
    def __float__(self):
        return SmtFloat(self.statespace, float, smt_bool_to_float(self.var))
    def __int__(self):
        return SmtInt(self.statespace, int, smt_bool_to_int(self.var))

    def __add__(self, other):
        return self._numeric_op(other, operator.add)
    def __sub__(self, other):
        return self._numeric_op(other, operator.sub)

class SmtInt(SmtNumberAble):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        assert typ == int
        SmtNumberAble.__init__(self, statespace, typ, smtvar)
    def __repr__(self):
        return repr(self.__index__())

    def __float__(self):
        return SmtFloat(self.statespace, float, smt_int_to_float(self.var))

    def __index__(self):
        #print('WARNING: attempting to materialize symbolic integer. Trace:')
        #traceback.print_stack()
        if self == 0:
            return 0
        i = 1
        while True:
            if self == i:
                return i
            if self == -i:
                return -i
            i += 1
        raise Exception('unable to realize integer '+str(self.var))
    def __bool__(self):
        return SmtBool(self.statespace, bool, self.var != 0).__bool__()
    def __int__(self):
        return self

    def __truediv__(self, other):
        return self.__float__() / other
    def __floordiv__(self, other):
        # TODO: Does this assume that other is an integer?
        return self._binary_op(other, lambda x,y:z3.If(x%y==0 or x>=0, x/y, z3.If(y>=0, x/y+1, x/y-1)))
    def __mod__(self, other):
        return self._binary_op(other, operator.mod)

    # TODO: consider asking the solver for an upper bound on the value and creating
    # a bitvector value using log2(upper bound).

    def __lshift__(self, other):
        raise Exception() # TODO: z3 cannot handle arbitrary precision bitwise operations
    def __rshift__(self, other):
        raise Exception() # TODO: z3 cannot handle arbitrary precision bitwise operations
    def __and__(self, other):
        raise Exception() # z3 cannot handle arbitrary precision bitwise operations
    def __xor__(self, other):
        raise Exception() # z3 cannot handle arbitrary precision bitwise operations
    def __or__(self, other):
        raise Exception() # z3 cannot handle arbitrary precision bitwise operations

    
class SmtFloat(SmtNumberAble):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        assert typ == float
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
    def __repr__(self):
        return repr(self.statespace.find_model_value(self.var))
    def __str__(self):
        return 'SmtFloat('+str(self.var)+')'
    def __hash__(self):
        raise Exception() # TODO

    def __truediv__(self, other):
        if not other:
            raise ZeroDivisionError('division by zero')
        return self._numeric_op(other, operator.truediv)


class SmtDict(SmtBackedValue, collections.abc.MutableMapping):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        arr_var = self.__arr()
        len_var = self.__len()
        self.val_missing_checker = arr_var.sort().range().recognizer(0)
        self.val_missing_constructor = arr_var.sort().range().constructor(0)
        self.val_constructor = arr_var.sort().range().constructor(1)
        self.val_accessor = arr_var.sort().range().accessor(1, 0)
        (self.key_pytype, self.val_pytype) = typ.__args__
        (self.key_ch_type, self.val_ch_type) = map(crosshair_type_for_python_type, typ.__args__)
        # Logically bind the length to the dictionary mapping:
        empty = z3.K(self.__arr().sort().domain(), self.val_missing_constructor())
        self.statespace.add(len_var >= 0)
        self.statespace.add((arr_var == empty) == (len_var == 0))
    def __init_var__(self, typ, varname):
        key_type, val_type = typ.__args__
        return (
            z3.Const(varname+'_map'+uniq(),
                     z3.ArraySort(type_to_smt_sort(key_type),
                                  possibly_missing_sort(type_to_smt_sort(val_type)))),
            z3.Const(varname+'_len'+uniq(), z3.IntSort())
        )
    def __arr(self):
        return self.var[0]
    def __len(self):
        return self.var[1]
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
        if SmtBool(self.statespace, bool, self.__len == 0).__bool__():
            raise IgnoreAttempt()
        self.var = (z3.Store(old_arr, k, missing), old_len - 1)
    def __getitem__(self, k):
        possibly_missing = self.__arr()[coerce_to_smt_var(self.statespace, k)[0]]
        is_missing = self.val_missing_checker(possibly_missing)
        if SmtBool(self.statespace, bool, is_missing).__bool__():
            raise KeyError(k)
        if SmtBool(self.statespace, bool, self.__len == 0).__bool__():
            raise IgnoreAttempt()
        return self.val_ch_type(self.statespace, self.val_pytype,
                                self.val_accessor(possibly_missing))
    def __len__(self):
        return SmtInt(self.statespace, int, self.__len())
    def __bool__(self):
        return SmtBool(self.statespace, bool, self.__len() != 0).__bool__()
    def __iter__(self):
        empty = z3.K(self.__arr().sort().domain(), self.val_missing_constructor())
        arr_var, len_var = self.var
        idx = 0
        arr_sort = self.__arr().sort()
        missing = self.val_missing_constructor()
        while SmtBool(self.statespace, bool, idx < len_var).__bool__():
            if SmtBool(self.statespace, bool, arr_var == empty).__bool__():
                raise IgnoreAttempt()
            k = z3.Const('k'+str(idx)+uniq(), arr_sort.domain())
            v = z3.Const('v'+str(idx)+uniq(), self.val_constructor.domain(0))
            remaining = z3.Const('remaining'+str(idx)+uniq(), arr_sort)
            idx += 1
            self.statespace.add(arr_var == z3.Store(remaining, k, self.val_constructor(v)))
            self.statespace.add(z3.Select(remaining, k) == missing)
            yield self.key_ch_type(self.statespace, self.key_pytype, k)
            arr_var = remaining
        # In this conditional, we reconcile the parallel symbolic variables for length
        # and contents:
        if SmtBool(self.statespace, bool, arr_var != empty).__bool__():
            raise IgnoreAttempt()


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
        self.item_pytype = typ.__args__[0] # (works for both List[T] and Tuple[T, ...])
        self.item_ch_type = crosshair_type_for_python_type(self.item_pytype)
    def __add__(self, other):
        return self._binary_op(other, z3.Concat)
    def __radd__(self, other):
        other_seq, other_pytype = coerce_to_smt_var(self.statespace, other)
        return self.__class__(self.statespace, self.python_type, z3.Concat(other_seq, self.var))
    def __contains__(self, other):
        return SmtBool(self.statespace, bool, z3.Contains(self.var, z3.Unit(smt_coerce(other))))
    def __getitem__(self, i):
        smt_result, is_slice = self._smt_getitem(i)
        if is_slice:
            return self.__class__(self.statespace, self.python_type, smt_result)
        elif self.item_ch_type is None:
            assert smt_result.sort() == z3.SeqSort(HeapRef)
            key = z3.Const('heap'+uniq(), HeapRef)
            self.statespace.add(smt_result == z3.Unit(key))
            return find_key_in_heap(self.statespace, key, self.item_pytype)
        else:
            result = self.item_ch_type(self.statespace, self.item_pytype, str(smt_result))
            self.statespace.add(smt_result == z3.Unit(result.var))
            return result

class SmtUniformList(SmtUniformListOrTuple): # TODO , collections.abc.MutableSequence):
    def __repr__(self):
        return str(list(self))
    def extend(self, other):
        self.var = self.var + smt_coerce(other)
    def __setitem__(self, k, v):
        self.var = z3.Store(self.var, smt_coerce(k), smt_coerce(v))
    def sort(self, **kw):
        if kw:
            raise Exception('sort arguments not supported')
        raise Exception()

'''
class SmtLazySequence(SmtBackedValue):
    def __init___(self, statespace:StateSpace, typ:Type, smtvar:object):
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        (self.item_pytype,) = typ.__args__
    def __str__(self):
        return 'SmtLazySequence('+str(self.var)+')'
    def __init_var__(self, typ, varname):
        return ([], z3.Const('len('+varname+')', z3.IntSort()))
    def extend(self, other):
        ...
    def __setitem__(self, k, v):
        ...
    def __getitem__(self, i):
        materialized, smt_len = self.var
        idx_or_range = process_slice_vs_symbolic_len(self.statespace, i, smt_len)
        if not isinstance(idx_or_range, tuple):
            for (k, v) in self.materialized:
                if idx_or_range == k:
                    return v
            self.materialized.append( (idx_or_range, self._new_item()) )
        else:
            low, high = idx_or_range
            return [self[i] for i in range(*idx_or_range)]
'''

class SmtUniformTuple(SmtUniformListOrTuple, collections.abc.Sequence, collections.abc.Hashable):
    pass
    
class SmtStr(SmtSequence, collections.abc.Sequence):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        assert typ == str
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        self.item_pytype = str
        self.item_ch_type = SmtStr
    def __repr__(self):
        return repr(self.statespace.find_model_value(self.var))
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
        datatype.declare(typ.__name__)
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
            if SmtBool(statespace, bool, getattr(self.vartype, 'is_' + typ.__name__)(var)).__bool__():
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
        ret = obj[name] if name in obj else getattr(cls, name)
        if hasattr(ret, '__call__'):
            ret = types.MethodType(ret, self)
        return ret
    def __delattr__(self, name):
        obj = object.__getattribute__(self, "_obj")
        del obj[name]
    def __setattr__(self, name, value):
        obj = object.__getattribute__(self, "_obj")
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
    def _create_class_proxy(cls, theclass):
        """creates a proxy for the given class"""
        
        def make_method(name):
            fn = getattr(theclass, name)
            def method(self, *args, **kw):
                return fn(self, *args, **kw)
            functools.update_wrapper(method, fn)
            return method
        
        namespace = {}
        for name in cls._special_names:
            if hasattr(theclass, name):
                namespace[name] = make_method(name)

        return type("%s(%s)" % (cls.__name__, theclass.__name__), (cls,), namespace)

    def __new__(cls, statespace, proxied_class, varname):
        try:
            cache = cls.__dict__["_class_proxy_cache"]
        except KeyError:
            cls._class_proxy_cache = cache = {}
        try:
            theclass = cache[proxied_class]
        except KeyError:
            cache[proxied_class] = theclass = cls._create_class_proxy(proxied_class)

        proxy = object.__new__(theclass)
        return proxy
    
def proxy_for_type(typ, statespace, varname):
    #print('proxy', typ, varname)
    if typing_inspect.is_typevar(typ):
        typ = int # TODO
    origin = getattr(typ, '__origin__', None)
    # special cases
    if origin is tuple:
        if len(typ.__args__) == 2 and typ.__args__[1] == ...:
            return SmtUniformTuple(statespace, typ, varname)
        else:
            return tuple(proxy_for_type(t, statespace, varname +'[' + str(idx) + ']')
                         for (idx, t) in enumerate(typ.__args__))
    elif isinstance(typ, type) and issubclass(typ, enum.Enum):
        enum_values = list(typ)
        for enum_value in enum_values[:-1]:
            if statespace.make_choice():
                statespace.model_additions[varname] = enum_value
                return enum_value
        statespace.model_additions[varname] = enum_values[-1]
        return enum_values[-1]
    elif typ is type(None):
        return None
    Typ = crosshair_type_for_python_type(typ)
    if Typ is not None:
        return Typ(statespace, typ, varname)
    ret = ProxiedObject(statespace, typ, varname)
    return ret

class BoundArgs:
    def __init__(self, args:dict, positional_only:List[str]):
        self._args = args
        self._positional_only = positional_only
    def copy(self):
        return BoundArgs({k: copy.copy(v) for (k, v) in self._args.items()},
                         self._positional_only)
    def arguments(self):
        return self._args
    def args(self):
        return [v for (k,v) in self._args.items() if k in self._positional_only]
    def kwargs(self):
        return {k:v for (k,v) in self._args.items() if k not in self._positional_only}

def env_for_args(sig: inspect.Signature, statespace:StateSpace) -> BoundArgs:
    positional_only = []
    args = {}
    for param in sig.parameters.values():
        smt_name = param.name + uniq()
        has_annotation = (param.annotation != inspect.Parameter.empty)
        if has_annotation:
            value = proxy_for_type(param.annotation, statespace, smt_name)
        else:
            value = proxy_for_type(Any, statespace, smt_name)            
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
            else:
                value = proxy_for_type(Dict[str, Any], statespace, smt_name)
            
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional_only.append(param.name)
        args[param.name] = value
    return BoundArgs(args, positional_only)


@functools.total_ordering
class MessageType(enum.Enum):
    POST_FAIL = 'post_fail'
    EXEC_ERR = 'exec_err'
    POST_ERR = 'post_err'
    CANNOT_CONFIRM = 'cannot_confirm'
    def __lt__(self, other):
        return self._order[self] < self._order[other]
MessageType._order = { # type: ignore
    MessageType.CANNOT_CONFIRM: 0,
    MessageType.POST_ERR: 1,
    MessageType.EXEC_ERR: 2,
    MessageType.POST_FAIL: 3,
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
    REFUTED_WITH_EMULATION = 1
    UNKNOWN = 2
    CONFIRMED = 3
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

@dataclass
class AnalysisOptions:
    use_called_conditions: bool = True
    timeout: float = 20.0
    deadline: float = float('NaN')

_DEFAULT_OPTIONS = AnalysisOptions()

class PatchedBuiltins:
    def __enter__(self):
        self.originals = builtins.__dict__.copy()

        # TODO: list(x) calls x.__len__().__index__() if it can.
        # (but patching `list` changes its identity, which breaks type(_) is list)
        builtins.len = self.patched_len
        builtins.range = self.patched_range
        builtins.sorted = self.patched_sorted
        
        # We patch various typing builtins to make SmtBackedValues look like
        # native values.
        # Note this will break code that depends on the identity of the type
        # function itself ("type(type) is type") etc.

        #builtins.type = self.patched_type
        
        original_issubclass = self.originals['issubclass']
        original_isinstance = self.originals['isinstance']
        builtins.issubclass = lambda x, y: (original_issubclass(x,y) or
                                            original_issubclass(_WRAPPER_TYPE_TO_PYTYPE.get(x,x), y))
        builtins.isinstance = lambda i, c: (original_isinstance(i, c) or
                                            original_issubclass(_WRAPPER_TYPE_TO_PYTYPE.get(type(i),type(i)), c))
    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.__dict__.update(self.originals)
        return False

    # CPython's len() forces the return value to be a native integer.
    # Avoid that requirement by making it only call __len__().
    def patched_len(self, l):
        return l.__len__()
    
    # Avoid calling __index__() on min/max integers.
    def patched_range(self, arg1):
        # TODO: min value, step value
        i = 0
        while i < arg1:
            yield i
            i += 1
        return
    
    # Avoid calling __len__().__index__() on the input list.
    def patched_sorted(self, l, **kw):
        ret = list(l.__iter__())
        ret.sort()
        return ret

    # Trick the system into believing that symbolic values are
    # native types.
    def patched_type(self, *args):
        ret = self.originals['type'](*args)
        if len(args) == 1:
            ret = _WRAPPER_TYPE_TO_PYTYPE.get(ret, ret)
        for (original_type, proxied_type) in ProxiedObject.__dict__["_class_proxy_cache"].items():
            if ret is proxied_type:
                return original_type
        return ret

def analyze_module(module:types.ModuleType) -> List[AnalysisMessage]:
    messages = MessageCollector()
    for (name, member) in inspect.getmembers(module):
        if inspect.isclass(member) and member.__module__ == module.__name__:
            messages.extend(analyze_class(member))
        elif inspect.isfunction(member) and member.__module__ == module.__name__:
            messages.extend(analyze(member))
    return messages.get()

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

def resolve_signature(fn:Callable, self_type:Optional[type]=None) -> inspect.Signature:
    sig = inspect.signature(fn)
    type_hints = get_type_hints(fn, fn_globals(fn))
    params = sig.parameters.values()
    if (self_type and
        len(params) > 0 and
        next(iter(params)).name == 'self' and
        'self' not in type_hints):
        type_hints['self'] = self_type
    newparams = []
    for name, param in sig.parameters.items():
        if name in type_hints:
            param = param.replace(annotation=type_hints[name])
        newparams.append(param)
    newreturn = type_hints.get('return', sig.return_annotation)
    return inspect.Signature(newparams, return_annotation=newreturn)

_EMULATION_TIMEOUT_FRACTION = 0.25
def analyze(fn:Callable,
            options:AnalysisOptions=_DEFAULT_OPTIONS,
            conditions:Optional[Conditions]=None,
            self_type:Optional[type]=None) -> List[AnalysisMessage]:
    if options.use_called_conditions: 
        options.deadline = time.time() + options.timeout * _EMULATION_TIMEOUT_FRACTION
    else:
        options.deadline = time.time() + options.timeout
    all_messages = MessageCollector()
    conditions = conditions or get_fn_conditions(fn)
    sig = resolve_signature(fn, self_type=self_type)
    
    (messages, verification_status) = analyze_calltree(fn, options, conditions, sig)
    all_messages.extend(messages)

    if (options.use_called_conditions and
        VerificationStatus.REFUTED_WITH_EMULATION in verification_status.values()):
        print('REATTEMPTING without short circuiting')
        
        # Re-attempt the unknown postconditions without short circuiting:
        conditions.post[:] = [c for c in conditions.post if
                              verification_status.get(c) != VerificationStatus.CONFIRMED]
        options = replace(options,
                          use_called_conditions=False,
                          deadline=time.time() + options.timeout * (1.0 - _EMULATION_TIMEOUT_FRACTION))
        (messages, new_verification_status) = analyze_calltree(fn, options, conditions, sig)
        all_messages = MessageCollector()
        all_messages.extend(messages)
        verification_status.update(new_verification_status)
        
    
    for (condition, status) in verification_status.items():
        if status in (VerificationStatus.REFUTED_WITH_EMULATION, VerificationStatus.UNKNOWN):
            all_messages.extend([AnalysisMessage(MessageType.CANNOT_CONFIRM, 'I cannot confirm this '+condition.addl_context,
                                                 condition.filename, condition.line, 0, '')])
    return all_messages.get()

def forget_contents(value:object, space:StateSpace):
    if isinstance(value, SmtBackedValue):
        clean = type(value)(value.statespace, value.python_type, str(value.var)+uniq())
        value.var = clean.var
    elif isinstance(value, ProxiedObject):
        obj = object.__getattribute__(value, '_obj')
        cls = object.__getattribute__(value, '_cls')
        clean = proxy_for_type(cls, space, uniq())
        clean_obj = object.__getattribute__(clean, '_obj')
        for key, val in obj.items():
            obj[key] = clean_obj[key]

def analyze_calltree(fn:Callable,
                     options:AnalysisOptions,
                     conditions:Conditions,
                     sig:inspect.Signature) -> Tuple[List[AnalysisMessage],
                                                     MutableMapping[ConditionExpr,VerificationStatus]]:
    print('Begin analyze calltree ', fn)
    worst_verification_status = {cond:VerificationStatus.CONFIRMED for cond in conditions.post}
    all_messages = MessageCollector()
    search_history = SearchTreeNode()
    space_exhausted = False
    for i in range(1000):
        if time.time() > options.deadline:
            break
        #print(' ** Iteration ', i)
        space = StateSpace(i, search_history)
        try:
            bound_args = env_for_args(sig, space)
        except IgnoreAttempt:
            #print('iter ignored (impossible starting args)')
            continue
        intercepted_flag = [False]
        def interceptor(original):
            sig = resolve_signature(original)
            subconditions = get_fn_conditions(original)
            def wrapper(*a,**kw):
                #print('intercepted a call to ', original)
                intercepted_flag[0] = True
                if subconditions.mutable_args:
                    bound = sig.bind(*a, **kw)
                    for mutated_arg in subconditions.mutable_args:
                        forget_contents(bound.arguments[mutated_arg])
                return proxy_for_type(sig.return_annotation, space, fn.__name__+'_return'+uniq())
            functools.update_wrapper(wrapper, original)
            return wrapper
        try:
            # TODO try to patch outside the search loop
            with EnforcedConditions(fn_globals(fn) if options.use_called_conditions else {}, interceptor=interceptor):
                with PatchedBuiltins():
                    (messages, verification_status) = attempt_call(conditions, space, fn, bound_args)
        except UnknownSatisfiability:
            messages = []
            verification_status = {cond:VerificationStatus.UNKNOWN for cond in conditions.post}
        except IgnoreAttempt:
            messages, verification_status = ([], {})
        #print('iter complete ', list(verification_status.values()))
        for (condition, status) in verification_status.items():
            if status == VerificationStatus.REFUTED and intercepted_flag[0]:
                status = VerificationStatus.REFUTED_WITH_EMULATION
            worst_verification_status[condition] = min(status, worst_verification_status[condition])
        
        all_messages.extend(messages)
        if space.check_exhausted():
            # we've searched every path
            space_exhausted = True
            break 
    if not space_exhausted:
        for (condition, status) in worst_verification_status.items():
            worst_verification_status[condition] = min(VerificationStatus.UNKNOWN,
                                                       worst_verification_status[condition])
        
    print(('Exhausted' if space_exhausted else 'Aborted') +' calltree search. Number of iterations: ', i+1)
    return (all_messages.get(), worst_verification_status)

def python_string_for_evaluated(expr:z3.ExprRef)->str:
    return str(expr)

def get_input_description(statespace:StateSpace,
                          bound_args:BoundArgs,
                          addl_context:str = '') -> str:
    messages:List[str] = []
    for argname,argval in bound_args.arguments().items():
        messages.append(argname + ' = ' + str(argval))
    if addl_context:
        return addl_context + ' with ' + ' and '.join(messages)
    elif messages:
        return 'when ' + ' and '.join(messages)
    else:
        return 'for any input'

def fn_globals(fn:Callable) -> Dict[str, object]:
    ''' This function mostly exists to avoid the typing error. '''
    return fn.__globals__ # type:ignore


def shallow_eq(old_val:object, new_val:object) -> bool:
    if old_val is new_val:
        return True
    if isinstance(old_val, ProxiedObject) and isinstance(new_val, ProxiedObject):
        od, nd = object.__getattribute__(old_val,'_obj'), object.__getattribute__(new_val,'_obj')
        for key in set(od.keys()).union(nd.keys()):
            if od.get(key,None) is not nd.get(key,None):
                return False
        return True
    elif isinstance(old_val, Hashable) and isinstance(new_val, Hashable):
        return old_val == new_val
    elif isinstance(old_val, SmtBackedValue) and isinstance(new_val, SmtBackedValue) and old_val.var is new_val.var:
        return True
    elif isinstance(old_val, Iterable) and isinstance(new_val, Iterable):
        if isinstance(old_val, Sized) and isinstance(new_val, Sized):
            if len(old_val) != len(new_val):
                return False
        for (o,n) in itertools.zip_longest(old_val, new_val, fillvalue=object()):
            if o != n:
                return False
        return True
    elif hasattr(old_val, '__dict__') and hasattr(new_val, '__dict__'):
        return old_val.__dict__ == new_val.__dict__
    else:
        return old_val == new_val

def attempt_call(conditions:Conditions,
                 statespace:StateSpace,
                 fn:Callable,
                 bound_args:BoundArgs) -> Tuple[List[AnalysisMessage],
                                                Mapping[ConditionExpr,VerificationStatus]]:
    original_args = bound_args.copy()
    post_conditions = conditions.post
    raises = conditions.raises
    try:
        for precondition in conditions.pre:
            if not eval(precondition.expr, fn_globals(fn), bound_args.arguments()):
                return ([], {})
    except UnknownSatisfiability:
        raise
    except BaseException as e:
        return ([], {})

    try:
        __return__ = fn(*bound_args.args(), **bound_args.kwargs())
        lcls = {**bound_args.arguments(), '__return__':__return__}
    except PostconditionFailed as e:
        # although this indicates a problem, it's with a subroutine; not this function.
        print('skip for internal postcondition '+str(e))
        return ([], {})
    except IgnoreAttempt:
        return ([], {})
    except UnknownSatisfiability:
        raise
    except BaseException as e:
        tb = traceback.format_exc()
        detail = str(type(e).__name__) + ': ' + str(e) + ' ' + get_input_description(statespace, original_args)
        frame = traceback.extract_tb(sys.exc_info()[2])[-1]
        return ([AnalysisMessage(MessageType.EXEC_ERR, detail, frame.filename, frame.lineno, 0, tb)],
                {c:VerificationStatus.REFUTED for c in post_conditions})

    for argname, argval in bound_args.arguments().items():
        if argname not in conditions.mutable_args:
            old_val, new_val = original_args.arguments()[argname], argval
            if not shallow_eq(old_val, new_val):
                detail = 'Argument "{}" is not marked as mutable, but changed from {} to {}'.format(argname, old_val, new_val)
                return ([AnalysisMessage(MessageType.POST_ERR, detail, fn.__code__.co_filename, fn.__code__.co_firstlineno, 0, '')],
                        {c:VerificationStatus.REFUTED for c in post_conditions})
    
    failures = []
    verification_status = {}
    for condition in post_conditions:
        try:
            isok = eval(condition.expr, fn_globals(fn), lcls)
        except IgnoreAttempt:
            return ([], {})
        except UnknownSatisfiability:
            verification_status[condition] = VerificationStatus.UNKNOWN
            continue
        except BaseException as e:
            tb = traceback.format_exc()
            verification_status[condition] = VerificationStatus.REFUTED
            detail = str(e) + ' ' + get_input_description(statespace, original_args, condition.addl_context)
            failures.append(AnalysisMessage(MessageType.POST_ERR, detail, condition.filename, condition.line, 0, tb))
            continue
        if isok:
            verification_status[condition] = VerificationStatus.CONFIRMED
        else:
            verification_status[condition] = VerificationStatus.REFUTED
            detail = 'false ' + get_input_description(statespace, original_args, condition.addl_context)
            failures.append(AnalysisMessage(MessageType.POST_FAIL, detail, condition.filename, condition.line, 0, ''))
    return (failures, verification_status)

_PYTYPE_TO_WRAPPER_TYPE = {
    type(None): (lambda *a: None),
    bool: SmtBool,
    int: SmtInt,
    float: SmtFloat,
    str: SmtStr,
    list: SmtUniformList,
    dict: SmtDict,
}

_WRAPPER_TYPE_TO_PYTYPE = dict((v,k) for (k,v) in _PYTYPE_TO_WRAPPER_TYPE.items())

if __name__ == '__main__':
    import sys, importlib
    for module_name in sys.argv[1:]:
        print(module_name)
        module = importlib.import_module(module_name)
        for message in analyze_module(module):
            print(sys.stderr, message)
