from dataclasses import dataclass, replace
from typing import *
from typed_inspect import signature
import typing_inspect
from condition_parser import get_fn_conditions, get_class_conditions, ConditionExpr, Conditions
from enforce import EnforcedConditions, PostconditionFailed
import collections
import builtins
import enum
import inspect
import functools
import operator
import random
import sys
import time
import traceback
import types

import z3  # type: ignore

_CHOICE_RANDOMIZATION = 4221242075

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

class UnknownSatisfiability(Exception):
    pass
    
class StateSpace:
    def __init__(self, seed:int, previous_searches:SearchTreeNode):
        #self.solver = z3.Solver()
        #self.solver = z3.Then('simplify','smt', 'qfnra-nlsat').solver()
        self.solver = z3.OrElse('smt', 'qfnra-nlsat').solver()
        self.solver.set(mbqi=True)
        self.seed = seed ^ _CHOICE_RANDOMIZATION
        self.search_position = previous_searches
        self.choices_made :List[SearchTreeNode] = []

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
        (choose_true, new_search_node) = self.search_position.choose(self.seed)
        self.choices_made.append(self.search_position)
        self.search_position = new_search_node
        expr = expr if choose_true else z3.Not(expr)
        self.add(expr)
        self.seed = self.seed // 2
        return choose_true

    def check_exhausted(self) -> bool:
        return SearchTreeNode.check_exhausted(self.choices_made, self.search_position)

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
    if origin in (list, Sequence, Container):
        item_type = t.__args__[0]
        return z3.SeqSort(type_to_smt_sort(item_type))
    elif origin in (dict, Mapping):
        key_type, val_type = t.__args__
        return z3.ArraySort(type_to_smt_sort(key_type),
                            possibly_missing_sort(type_to_smt_sort(val_type)))
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

def coerce_to_smt_var(v:Any) -> Tuple[z3.ExprRef, Type]:
    if isinstance(v, SmtBackedValue):
        return (v.var, v.python_type)
    if isinstance(v, list):
        (vars, pytypes) = zip(*map(coerce_to_smt_var, v))
        if len(vars) == 0:
            return ([], list) # TODO: we cannot know the type here.
        elif len(vars) == 1:
            return (z3.Unit(vars[0]), list)
        else:
            return (z3.Concat(*map(z3.Unit,vars)), list)
    promotion_fn = _LITERAL_PROMOTION_FNS.get(type(v))
    if promotion_fn:
        return (promotion_fn(v), type(v))
    raise Exception()

def coerce_to_ch_value(v:Any, statespace:StateSpace) -> object:
    (smt_var, py_type) = coerce_to_smt_var(v)
    Typ = crosshair_type_for_python_type(py_type)
    if Typ is None:
        raise Exception('Unable to get ch type from python type: '+str(py_type))
    return Typ(statespace, py_type, smt_var)
    
class SmtBackedValue:
    def __init__(self, statespace:StateSpace, typ: Type, smtvar:Union[z3.ExprRef, str]):
        self.statespace = statespace
        if isinstance(smtvar, str):
            self.var = smt_var(typ, smtvar)
            self.python_type = typ
        else:
            self.var = smtvar
            self.python_type = typ
            # TODO test that smtvar's sort matches expected?
    def __eq__(self, other):
        return SmtBool(self.statespace, bool, self.var == coerce_to_smt_var(other)[0])
    def __req__(self, other):
        return coerce_to_ch_value(other, self.statespace).__eq__(self)
    def _binary_op(self, other, op):
        return self.__class__(self.statespace, self.python_type, op(self.var, smt_coerce(other)))
    def _cmp_op(self, other, op):
        return SmtBool(self.statespace, bool, op(self.var, smt_coerce(other)))
    def _unary_op(self, op):
        return self.__class__(self.statespace, self.python_type, op(self.var))

class SmtNumberAble(SmtBackedValue):
    def _numeric_op(self, other, op):
        l_var, lpytype = self.var, self.python_type
        r_var, rpytype = coerce_to_smt_var(other)
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
    def __init__(self, statespace:StateSpace, typ: Type, smtvar:Union[z3.ExprRef, str]):
        assert typ == bool
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
    def __str__(self):
        return 'SmtBool('+str(self.var)+')'

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
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:Union[z3.ExprRef, str]):
        assert typ == int
        SmtNumberAble.__init__(self, statespace, typ, smtvar)
    def __str__(self):
        return 'SmtInt('+str(self.var)+')'

    def __float__(self):
        return SmtFloat(self.statespace, float, smt_int_to_float(self.var))

    def __index__(self):
        print('WARNING: attempting to materialize symbolic integer. Trace:')
        traceback.print_stack()
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
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:Union[z3.ExprRef, str]):
        assert typ == float
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
    def __str__(self):
        return 'SmtFloat('+str(self.var)+')'

    def __truediv__(self, other):
        if not other:
            raise ZeroDivisionError('division by zero')
        return self._numeric_op(other, operator.truediv)


class SmtDict(SmtBackedValue):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:Union[z3.ExprRef, str]):
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        self.val_missing_checker = self.var.sort().range().recognizer(0)
        self.val_missing_constructor = self.var.sort().range().constructor(0)
        self.val_constructor = self.var.sort().range().constructor(1)
        self.val_accessor = self.var.sort().range().accessor(1, 0)
        (self.key_pytype, self.val_pytype) = typ.__args__
        (self.key_ch_type, self.val_ch_type) = map(crosshair_type_for_python_type, typ.__args__)
    def __str__(self):
        return 'SmtDict('+str(self.var)+')'
    def __setitem__(self, k, v):
        (k,_), (v,_) = coerce_to_smt_var(k), coerce_to_smt_var(v)
        self.var = z3.Store(self.var, k, self.val_constructor(v))
    def __getitem__(self, k):
        possibly_missing = self.var[coerce_to_smt_var(k)[0]]
        is_missing = self.val_missing_checker(possibly_missing)
        if SmtBool(self.statespace, bool, is_missing).__bool__():
            raise KeyError(str(k))
        return self.val_ch_type(self.statespace, self.val_pytype,
                                self.val_accessor(possibly_missing))
    # TODO len: trigger axioms like this (with hopefully good patterns):
    #   len([]) == 0
    #   Store(M,k,v) & new k -> len(Store(M,k,v)) = len(M)+1
    #   Store(M,k,v) & old k -> len(Store(M,k,v)) = len(M)
    #def __len__(self):
    def __bool__(self):
        empty = z3.K(self.var.sort().domain(), self.val_missing_constructor())
        return SmtBool(self.statespace, bool, self.var != empty).__bool__()
    def __iter__(self):
        missing = self.val_missing_constructor()
        empty = z3.K(self.var.sort().domain(), missing)
        cur = self.var
        arr_sort = self.var.sort()
        ii=100000 # TODO need globally unique constant names
        while SmtBool(self.statespace, bool, cur != empty).__bool__():
            k = z3.Const('k'+str(ii), arr_sort.domain())
            v = z3.Const('v'+str(ii), self.val_constructor.domain(0))
            remaining = z3.Const('remaining'+str(ii), arr_sort)
            ii+=1
            self.statespace.add(cur == z3.Store(remaining, k, self.val_constructor(v)))
            self.statespace.add(z3.Select(remaining, k) == missing)
            yield self.key_ch_type(self.statespace, self.key_pytype, k)
            cur = remaining

    
class SmtSeq(SmtBackedValue):
    def _smt_getitem(self, i):
        smt_var, smt_i = self.var, smt_coerce(i)
        def normalize_symbolic_index(idx, container):
            if isinstance(idx, int):
                return idx if idx >= 0 else z3.Length(container) + idx
            else:
                return z3.If(idx >= 0, idx, z3.Length(container) + idx)
        if isinstance(i, int) or isinstance(i, SmtInt):
            my_length = self.__len__()
            if i >= my_length or i < -my_length:
                raise IndexError('index out of range')
            return (smt_var[normalize_symbolic_index(smt_i, smt_var)], False)
        elif isinstance(i, slice):
            smt_start, smt_stop, smt_step = map(smt_coerce, (i.start, i.stop, i.step))
            if smt_step not in (None, 1):
                raise Exception('slice steps not handled in slice: '+str(i))
            start = normalize_symbolic_index(smt_start, smt_var) if i.start is not None else 0
            stop = normalize_symbolic_index(smt_stop, smt_var) if i.stop is not None else z3.Length(smt_var)
            return (z3.Extract(smt_var, start, stop), True)
        else:
            raise Exception('invalid slice parameter: '+str(i))
    def sort(self, **kw):
        if kw:
            raise Exception('sort arguments not supported')
        raise Exception()
    def __iter__(self):
        idx = 0
        while len(self) > idx:
            yield self[idx]
            idx += 1

class SmtList(SmtSeq):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:Union[z3.ExprRef, str]):
        assert origin_of(typ) in (tuple, list)
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        self.item_pytype = typ.__args__[0]
        self.item_ch_type = crosshair_type_for_python_type(self.item_pytype)
    def __str__(self):
        return 'SmtSeq('+str(self.var)+')'
    def __add__(self, other):
        return self._binary_op(other, z3.Concat)
    def __radd__(self, other):
        other_seq, other_pytype = coerce_to_smt_var(other)
        return SmtList(self.statespace, self.python_type, z3.Concat(other_seq, self.var))
    def __contains__(self, other):
        return SmtBool(self.statespace, bool, z3.Contains(self.var, z3.Unit(smt_coerce(other))))
    def __bool__(self):
        return SmtBool(self.statespace, bool, z3.Length(self.var) > 0).__bool__()
    def __len__(self):
        return SmtInt(self.statespace, int, z3.Length(self.var))
    def extend(self, other):
        self.var = self.var + smt_coerce(other)
    def __setitem__(self, k, v):
        self.var = z3.Store(self.var, smt_coerce(k), smt_coerce(v))
    def __getitem__(self, i):
        smt_result, is_slice = self._smt_getitem(i)
        if is_slice:
            return SmtList(self.statespace, self.python_type, smt_result)
        else:
            result = self.item_ch_type(self.statespace, self.item_pytype, str(smt_result))
            self.statespace.add(smt_result == z3.Unit(result.var))
            return result

class SmtStr(SmtSeq):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:Union[z3.ExprRef, str]):
        assert typ == str
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        self.item_pytype = str
        self.item_ch_type = SmtStr
    def __str__(self):
        return 'SmtStr('+str(self.var)+')'
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
    def __len__(self):
        return SmtInt(self.statespace, int, z3.Length(self.var))
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


class SmtNone:
    def __call__(self, *a):
        return None


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
    __slots__ = ["_cls", "_obj", "_varname","__weakref__"]
    def __init__(self, statespace, cls, varname):
        state = {}
        for name, typ in get_type_hints(cls).items():
            origin = getattr(typ, '__origin__', None)
            if origin is Callable:
                continue
            state[name] = proxy_for_type(typ, statespace, varname+'.'+name)

        object.__setattr__(self, "_obj", state)
        object.__setattr__(self, "_varname", state)
        object.__setattr__(self, "_cls", cls)

    def __getattribute__(self, name):
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
    if typing_inspect.is_typevar(typ):
        typ = int # TODO
    origin = getattr(typ, '__origin__', None)
    if origin is tuple:
        if len(typ.__args__) == 2 and typ.__args__[1] == ...:
            raise Exception()
        #z3type = z3.SeqSort(type_to_smt_sort(typ.__args__[0]))
        else:
            return tuple(proxy_for_type(t, statespace, varname +'[' + str(idx) + ']')
                         for (idx, t) in enumerate(typ.__args__))
    Typ = crosshair_type_for_python_type(typ)
    if Typ is None:
        #raise Exception('Cannot find crosshair type for python type: '+str(typ))
        Typ = ProxiedObject
    return Typ(statespace, typ, varname)

class BoundArgs:
    def __init__(self, args:dict, positional_only:List[str]):
        self._args = args
        self._positional_only = positional_only
    def arguments(self):
        return self._args
    def args(self):
        #print('args', [v for (k,v) in self._args.items() if k in self._positional_only])
        return [v for (k,v) in self._args.items() if k in self._positional_only]
    def kwargs(self):
        #print('kw', {k:v for (k,v) in self._args.items() if k not in self._positional_only})
        return {k:v for (k,v) in self._args.items() if k not in self._positional_only}

def env_for_args(sig: inspect.Signature, statespace:StateSpace) -> BoundArgs:
    positional_only = []
    args = {}
    for param in sig.parameters.values():
        has_annotation = (param.annotation != inspect.Parameter.empty)
        if has_annotation:
            value = proxy_for_type(param.annotation, statespace, param.name)
        else:
            value = proxy_for_type(Any, statespace, param.name)            
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            if has_annotation:
                varargs_type = List[param.annotation] # type: ignore
                value = proxy_for_type(varargs_type, statespace, param.name)
            else:
                value = proxy_for_type(List[Any], statespace, param.name)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            if has_annotation:
                varargs_type = Dict[str, param.annotation] # type: ignore
                value = proxy_for_type(varargs_type, statespace, param.name)
            else:
                value = proxy_for_type(Dict[str, Any], statespace, param.name)
            
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional_only.append(param.name)
        args[param.name] = value
    return BoundArgs(args, positional_only)


class MessageCollector:
    def __init__(self):
        self.by_pos = {}
    def extend(self, messages):
        for message in messages:
            self.by_pos[(message.filename, message.line, message.column)] = message
    def get(self):
        return [m for (k,m) in sorted(self.by_pos.items())]

@dataclass(frozen=True)
class AnalysisMessage:
    state: str
    message: str
    filename: str
    line: int
    column: int

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
    timeout: float = 5.0
    deadline: float = float('NaN')

_DEFAULT_OPTIONS = AnalysisOptions()

class PatchedBuiltins:
    def __enter__(self):
        self.originals = builtins.__dict__.copy()
        
        # CPython's len() forces the return value to be a real integer.
        # We patch it here to avoid that requirement.
        builtins.len = lambda v:v.__len__()

        # CPython's sorted creates native lists
        builtins.sorted = self.patched_sorted
        builtins.range = self.patched_range
        
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

    def patched_range(self, arg1):
        i = 0
        while i < arg1:
            yield i
            i += 1
        return
    '''
        maxval = arg1
        if isinstance(maxval, SmtInt):
            statespace = maxval.statespace
            ret = SmtList(statespace, List[int], 'range('+str(maxval.var)+')')
            i = z3.Const('i', z3.IntSort())
            statespace.add(z3.ForAll([i], z3.Implies(z3.And(0 <= i, i < maxval.var),
                                                     ret.var[i] == z3.Unit(i)),
                                     patterns=[ret.var[i]]))
            return ret
        return self.originals['range'](arg1)
    '''
    
    def patched_sorted(self, l, **kw):
        ret = list(l.__iter__())
        ret.sort()
        return ret

    def patched_type(self, *args):
        ret = self.originals['type'](*args)
        if len(args) == 1:
            ret = _WRAPPER_TYPE_TO_PYTYPE.get(ret, ret)
        return ret
        
def analyze_class(cls:type, options:AnalysisOptions=_DEFAULT_OPTIONS) -> List[AnalysisMessage]:
    messages = []
    class_conditions = get_class_conditions(cls)
    for method, conditions in class_conditions.methods:
        if conditions.has_any():
            messages.extend(analyze(method,
                                    conditions=conditions,
                                    options=options,
                                    self_type=cls))
        
    return messages

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


def analyze(fn:Callable,
            options:AnalysisOptions=_DEFAULT_OPTIONS,
            conditions:Optional[Conditions]=None,
            self_type:Optional[type]=None) -> List[AnalysisMessage]:
    options.deadline = time.time() + options.timeout
    all_messages = MessageCollector()
    conditions = conditions or get_fn_conditions(fn)
    sig = resolve_signature(fn, self_type=self_type)
    
    (messages, verification_status) = analyze_calltree(fn, options, conditions, sig)
    all_messages.extend(messages)

    if (options.use_called_conditions and
        VerificationStatus.UNKNOWN in verification_status.values()):
        print('REATTEMPTING without short circuiting')
        
        # Re-attempt the unknown postconditions without short circuiting:
        conditions.post[:] = [c for c in conditions.post if
                              verification_status.get(c) == VerificationStatus.UNKNOWN]
        (messages, verification_status) = analyze_calltree(
            fn, replace(options, use_called_conditions=False), conditions, sig)
        all_messages = MessageCollector()
        all_messages.extend(messages)
        
    
    for (condition, status) in verification_status.items():
        if status == VerificationStatus.UNKNOWN:
            all_messages.extend([AnalysisMessage('cannot_confirm', 'I cannot confirm this',
                                                 condition.filename, condition.line, 0)])
    return all_messages.get()

def analyze_calltree(fn:Callable,
                     options:AnalysisOptions,
                     conditions:Conditions,
                     sig:inspect.Signature) -> Tuple[List[AnalysisMessage],
                                                     Mapping[ConditionExpr,VerificationStatus]]:
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
        bound_args = env_for_args(sig, space)
        intercepted_flag = [False]
        def interceptor(original):
            sig = resolve_signature(original)
            def wrapper(*a,**kw):
                intercepted_flag[0] = True
                return proxy_for_type(sig.return_annotation, space, fn.__name__+'_return'+str(random.randint(1000,9999)))
            functools.update_wrapper(wrapper, original)
            return wrapper
        try:
            with EnforcedConditions(fn_globals(fn) if options.use_called_conditions else {}, interceptor=interceptor):
                with PatchedBuiltins():
                    (messages, verification_status) = attempt_call(conditions, space, fn, bound_args)
        except UnknownSatisfiability:
            messages = []
            verification_status = {cond:VerificationStatus.UNKNOWN for cond in conditions.post}
        for (condition, status) in verification_status.items():
            if status == VerificationStatus.REFUTED and intercepted_flag[0]:
                # Do not count failures if we short-circuited a subroutine call
                status = VerificationStatus.UNKNOWN
            worst_verification_status[condition] = min(status, worst_verification_status[condition])
        
        # skip checking postconditions that we've already refuted:
        #conditions.post[:] = [c for c in conditions.post if
        #                      verification_status.get(c) != VerificationStatus.REFUTED]
            
        all_messages.extend(messages)
        if not conditions.post:
            break # we broke every postcondition
        if space.check_exhausted():
            # we've searched every path
            space_exhausted = True
            break 
    if not space_exhausted:
        for (condition, status) in worst_verification_status.items():
            worst_verification_status[condition] = min(VerificationStatus.UNKNOWN,
                                                       worst_verification_status[condition])
        
    print('End analyze calltree. Number of iterations: ', i+1)
    return (all_messages.get(), worst_verification_status)

def python_string_for_evaluated(expr:z3.ExprRef)->str:
    return str(expr)

def get_input_description(statespace:StateSpace,
                          bound_args:BoundArgs,
                          addl_context:str = '') -> str:
    model = statespace.model()
    #print('model', model)
    if not model:
        return 'for any input'
    message = []
    for expr in model.decls():
        for argname in bound_args.arguments().keys():
            exprname = expr.name()
            if exprname.startswith(argname) or exprname.startswith('type('+argname+')'):
                message.append(expr.name()+' = '+python_string_for_evaluated(model.get_interp(expr)))
                break
    return 'when ' + addl_context + ' and '.join(message)

def fn_globals(fn:Callable) -> Dict[str, object]:
    ''' This function mostly exists to avoid the typing error. '''
    return fn.__globals__ # type:ignore

def attempt_call(conditions:Conditions,
                 statespace:StateSpace,
                 fn:Callable,
                 bound_args:BoundArgs) -> Tuple[List[AnalysisMessage],
                                                Mapping[ConditionExpr,VerificationStatus]]:
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
    except PostconditionFailed:
        return ([], {})
    except UnknownSatisfiability:
        raise
    except BaseException as e:
        traceback.print_exc()
        detail = str(type(e).__name__) + ': ' + str(e) + ' ' + get_input_description(statespace, bound_args)
        frame = traceback.extract_tb(sys.exc_info()[2])[-1]
        return ([AnalysisMessage('exec_err', detail, frame.filename, frame.lineno, 0)], {})

    failures = []
    verification_status = {}
    for condition in post_conditions:
        try:
            isok = eval(condition.expr, fn_globals(fn), lcls)
        except UnknownSatisfiability:
            verification_status[condition] = VerificationStatus.UNKNOWN
            continue
        except BaseException as e:
            traceback.print_exc()
            detail = str(e) + ' ' + get_input_description(statespace, bound_args, condition.addl_context)
            failures.append(AnalysisMessage('post_err', detail, condition.filename, condition.line, 0))
            continue
        if isok:
            verification_status[condition] = VerificationStatus.CONFIRMED
        else:
            verification_status[condition] = VerificationStatus.REFUTED
            detail = 'false ' + get_input_description(statespace, bound_args, condition.addl_context)
            failures.append(AnalysisMessage('post_fail', detail, condition.filename, condition.line, 0))
    return (failures, verification_status)


_PYTYPE_TO_WRAPPER_TYPE = {
    type(None): (lambda *a: None),
    bool: SmtBool,
    int: SmtInt,
    float: SmtFloat,
    str: SmtStr,
    list: SmtList,
    dict: SmtDict,
}

_WRAPPER_TYPE_TO_PYTYPE = dict((v,k) for (k,v) in _PYTYPE_TO_WRAPPER_TYPE.items())

