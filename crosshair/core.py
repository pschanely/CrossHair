# TODO: Are non-overridden supclass method conditions checked in subclasses? (they should be)
# TODO: precondition strengthening ban (Subclass constraint rule)
# TODO: Symbolic subclasses
# TODO: Object behavior and subclasses.
# TODO: Test Z3 Arrays nested inside Datastructures
# TODO: larger examples
# TODO: increase test coverage: TypeVar('T', int, str) vs bounded type vars
# TODO: first interrupt gets swallowed ("crosshair check")
# TODO: eq consistent with hash as a contract on `object`?
#       and comparison consistency elsewhere
# TODO: termination/search heuristics: deep paths suggest we need to terminate earlier.
#       overly shallow searches, though, can also miss opportunities
# TODO: slice < SmtInt shouldn't z3 error
# TODO: enforcement wrapper with preconditions that error: problematic for implies()
# TODO: if unsat preconditions, show error if errors happen +1
# TODO: class invariants cannot see other globals (possibly only on generated functions?)
#       Consider solving this by added a namespace attribute to the Conditions class.

# *** Not prioritized for v0 ***
# TODO: fully dynamic path fork reducers:
#       Worst (usual case; propagates the result closest to a REJECT)
#       ConfirmOrElse (for short-circuiting; CONFIRM on 1st branch propagates)
#       EquivChoice (for parallel proof approaches; first non-UNKNOWN result propagates)
# TODO: double-check counterexamples
# TODO: contracts for builtins
# TODO: standard library contracts
# TODO: identity-aware repr'ing for result messages
# TODO: mutating symbolic Callables?
# TODO: contracts on the contracts of function and object inputs/outputs?
# TODO: conditions on Callable arguments/return values

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
import linecache
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

from crosshair import contracted_builtins
from crosshair import dynamic_typing
from crosshair.abcstring import AbcString
from crosshair.condition_parser import get_fn_conditions, get_class_conditions, ConditionExpr, Conditions, fn_globals
from crosshair.enforce import EnforcedConditions, PostconditionFailed
from crosshair.simplestructs import SimpleDict
from crosshair.statespace import ReplayStateSpace, TrackingStateSpace, StateSpace, HeapRef, SnapshotRef, SearchTreeNode, model_value_to_python
from crosshair.util import CrosshairInternal, UnexploredPath, IdentityWrapper, AttributeHolder
from crosshair.util import debug, set_debug, extract_module_from_file, walk_qualname, get_subclass_map


def samefile(f1: Optional[str], f2: Optional[str]) -> bool:
    try:
        return f1 is not None and f2 is not None and os.path.samefile(f1, f2)
    except FileNotFoundError:
        return False

def exception_line_in_file(frames:traceback.StackSummary, filename:str) -> Optional[int]:
    for frame in reversed(frames):
        if samefile(frame.filename, filename):
            return frame.lineno
    return None

def frame_summary_for_fn(frames:traceback.StackSummary, fn:Callable) -> Tuple[str, int]:
    fn_name = fn.__name__
    fn_file = cast(str, inspect.getsourcefile(fn))
    for frame in reversed(frames):
        if (frame.name == fn_name and
            samefile(frame.filename, fn_file)):
            return (frame.filename, frame.lineno)
    (_, fn_start_line) = inspect.getsourcelines(fn)
    return fn_file, fn_start_line
    raise CrosshairInternal('Unable to find function {} in stack frames'.format(fn_name))

_MISSING = object()

class IgnoreAttempt(Exception):
    def __init__(self, *a):
        CrosshairInternal.__init__(self, *a)
        debug('IgnoreAttempt', str(self))
    
class CrosshairUnsupported(CrosshairInternal):
    def __init__(self, *a):
        CrosshairInternal.__init__(self, *a)
        debug('CrosshairUnsupported. Stack trace:\n' + ''.join(traceback.format_stack()))

class PatchedBuiltins:
    def __init__(self, patches: Mapping[str, object], enabled: Callable[[], bool]):
        self._patches = patches
        self._enabled = enabled
    def patch(self, key: str, newfn: Callable):
        orig_fn = builtins.__dict__[key]
        self._originals[key] = orig_fn
        def call_if_enabled(*a, **kw):
            return newfn(*a, **kw) if self._enabled() else orig_fn(*a, **kw)
        functools.update_wrapper(call_if_enabled, orig_fn)
        builtins.__dict__[key] = call_if_enabled
    def __enter__(self):
        patches = self._patches
        added_keys = []
        self._added_keys = added_keys
        originals = {}
        self._originals = originals
        for key, val in patches.items():
            if key.startswith('_') or not isinstance(val, Callable):
                continue
            if hasattr(builtins, key):
                self.patch(key, val)
            else:
                added_keys.append(key)
                builtins.__dict__[key] = val
        self._added_keys = added_keys
        self._originals = originals
    def __exit__(self, exc_type, exc_value, tb):
        bdict = builtins.__dict__
        bdict.update(self._originals)
        for key in self._added_keys:
            del bdict[key]
    
class ExceptionFilter:
    analysis: 'CallAnalysis'
    ignore: bool = False
    ignore_with_confirmation: bool = False
    user_exc: Optional[Tuple[BaseException, traceback.StackSummary]] = None
    def has_user_exception(self) -> bool:
        return self.user_exc is not None
    def __enter__(self) -> 'ExceptionFilter':
        return self
    def __exit__(self, exc_type, exc_value, tb):
        if isinstance(exc_value, (PostconditionFailed, IgnoreAttempt, NotImplementedError)):
            if isinstance(exc_value, PostconditionFailed):
                # Postcondition : although this indicates a problem, it's with a
                # subroutine; not this function.
                # Usualy we want to ignore this because it will be surfaced more locally
                # in the subroutine.
                debug(f'Ignoring based on internal failed post condition: {exc_value}')
            self.ignore = True
            if isinstance(exc_value, NotImplementedError):
                self.analysis = CallAnalysis(VerificationStatus.CONFIRMED)
            else:
                self.analysis = CallAnalysis()
            return True
        if isinstance(exc_value, (UnexploredPath, CrosshairInternal, z3.Z3Exception)):
            return False # internal issue: re-raise
        if isinstance(exc_value, BaseException): # TODO: should this be "Exception" instead?
            # Most other issues are assumed to be user-level exceptions:
            self.user_exc = (exc_value, traceback.extract_tb(sys.exc_info()[2]))
            self.analysis = CallAnalysis(VerificationStatus.REFUTED)
            return True # suppress user-level exception
        return False # re-raise resource and system issues


def smt_sort_has_heapref(sort: z3.SortRef) -> bool:
    return 'HeapRef' in str(sort)  # TODO: don't do this :)

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
            # instantiated differently in different places. So, this doesn't work:
            # return Union.__getitem__(tuple(map(normalize_pytype, constraints)))
        return object
    if typ is Any:
        # The distinction between any and object is for type checking, crosshair treats them the same
        return object
    return typ

def python_type(o:object) -> Type:
    if isinstance(o, SmtBackedValue):
        return o.python_type
    elif isinstance(o, SmtProxyMarker):
        return type(o).__bases__[0]
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
    datatype = z3.Datatype('optional_'+str(sort)+'_')
    datatype.declare('missing')
    datatype.declare('present', ('valueat', sort))
    ret = datatype.create()
    return ret

def type_to_smt_sort(t: Type) -> z3.SortRef:
    t = normalize_pytype(t)
    if t in _TYPE_TO_SMT_SORT:
        return _TYPE_TO_SMT_SORT[t]
    origin = origin_of(t)
    if origin in (list, Sequence, Container, tuple):
        if (origin is not tuple or
            (len(t.__args__) == 2 and t.__args__[1] == ...)):
            item_type = t.__args__[0]
            item_sort = type_to_smt_sort(item_type)
            return z3.SeqSort(item_sort)
    return HeapRef

def smt_var(typ: Type, name: str):
    z3type = type_to_smt_sort(typ)
    return z3.Const(name, z3type)

FunctionLike = Union[types.FunctionType, types.MethodType]
SmtGenerator = Callable[[StateSpace, type, Union[str, z3.ExprRef]], object]

_PYTYPE_TO_WRAPPER_TYPE :Dict[type, SmtGenerator] = {} # to be populated later
_WRAPPER_TYPE_TO_PYTYPE :Dict[SmtGenerator, type] = {}

def crosshair_type_that_inhabits_python_type(typ:Type) -> Optional[SmtGenerator]:
    typ = normalize_pytype(typ)
    origin = origin_of(typ)
    if origin is Union:
        return SmtUnion(frozenset(typ.__args__))
    return _PYTYPE_TO_WRAPPER_TYPE.get(origin)

def crosshair_type_for_python_type(typ:Type) -> Optional[SmtGenerator]:
    typ = normalize_pytype(typ)
    origin = origin_of(typ)
    return _PYTYPE_TO_WRAPPER_TYPE.get(origin)

def smt_bool_to_int(a: z3.ExprRef) -> z3.ExprRef:
    return z3.If(a, 1, 0)

def smt_int_to_float(a: z3.ExprRef) -> z3.ExprRef:
    if _SMT_FLOAT_SORT == z3.Float64():
        return z3.fpRealToFP(z3.RNE(), z3.ToReal(a), _SMT_FLOAT_SORT)
    elif _SMT_FLOAT_SORT == z3.RealSort():
        return z3.ToReal(a)
    else:
        raise CrosshairInternal()

def smt_bool_to_float(a: z3.ExprRef) -> z3.ExprRef:
    if _SMT_FLOAT_SORT == z3.Float64():
        return z3.If(a, z3.FPVal(1.0, _SMT_FLOAT_SORT), z3.FPVal(0.0, _SMT_FLOAT_SORT))
    elif _SMT_FLOAT_SORT == z3.RealSort():
        return z3.If(a, z3.RealVal(1), z3.RealVal(0))
    else:
        raise CrosshairInternal()

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

def coerce_to_smt_sort(space: StateSpace, input_value: Any, desired_sort: z3.SortRef) -> z3.ExprRef:
    natural_value = None
    promotion_fn = _LITERAL_PROMOTION_FNS.get(type(input_value))
    if isinstance(input_value, SmtBackedValue):
        natural_value = input_value.var # TODO: 'input.var' could be a tuple for dict/set
    elif promotion_fn:
        natural_value = promotion_fn(input_value)
    natural_sort = natural_value.sort() if natural_value is not None else None
    if natural_sort == desired_sort:
        return natural_value
    if desired_sort == HeapRef:
        return space.find_val_in_heap(input_value)
    if isinstance(input_value, (tuple, list)) and isinstance(desired_sort, z3.SeqSortRef):
        desired_basis = desired_sort.basis()
        converted = [coerce_to_smt_sort(space, item, desired_basis)
                     for item in input_value]
        if len(converted) == 0:
            return z3.Empty(desired_sort)
        elif len(converted) == 1:
            return z3.Unit(converted[0])
        else:
            return z3.Concat(*map(z3.Unit, converted))
    return None

def coerce_to_smt_var(space:StateSpace, v:Any) -> Tuple[z3.ExprRef, Type]:
    if isinstance(v, SmtBackedValue):
        return (v.var, v.python_type)
    promotion_fn = _LITERAL_PROMOTION_FNS.get(type(v))
    if promotion_fn:
        return (promotion_fn(v), type(v))
    return (space.find_val_in_heap(v), type(v))

def smt_to_ch_value(space: StateSpace, snapshot: SnapshotRef, smt_val: z3.ExprRef, pytype: type) -> object:
    def proxy_generator(typ: Type) -> object:
        return proxy_for_type(typ, space, 'heapval' + str(typ) + space.uniq())
    if smt_val.sort() == HeapRef:
        return space.find_key_in_heap(smt_val, pytype, proxy_generator, snapshot)
    ch_type = crosshair_type_that_inhabits_python_type(pytype)
    assert ch_type is not None
    return ch_type(space, pytype, smt_val)

def coerce_to_ch_value(v:Any, statespace:StateSpace) -> object:
    (smt_var, py_type) = coerce_to_smt_var(statespace, v)
    Typ = crosshair_type_that_inhabits_python_type(py_type)
    if Typ is None:
        raise CrosshairInternal('Unable to get ch type from python type: '+str(py_type))
    return Typ(statespace, py_type, smt_var)

class SmtBackedValue:
    def __init__(self, statespace:StateSpace, typ: Type, smtvar:object):
        self.statespace = statespace
        self.snapshot = SnapshotRef(-1)
        self.python_type = typ
        if isinstance(smtvar, str):
            self.var = self.__init_var__(typ, smtvar)
        else:
            self.var = smtvar
            # TODO test that smtvar's sort matches expected?
    def __init_var__(self, typ, varname):
        return smt_var(typ, varname)
    def __deepcopy__(self, memo):
        shallow = copy.copy(self)
        shallow.snapshot = self.statespace.current_snapshot()
        return shallow
    def __eq__(self, other):
        coerced = coerce_to_smt_sort(self.statespace, other, self.var.sort())
        if coerced is None:
            return False
        return SmtBool(self.statespace, bool, self.var == coerced)
    def _coerce_into_compatible(self, other):
        raise TypeError(f'Unexpected type "{type(other)}"')
    def __ne__(self, other):
        return not self.__eq__(other)
    def __req__(self, other):
        return self.__eq__(other)
    def __rne__(self, other):
        return coerce_to_ch_value(other, self.statespace).__ne__(self)

    def __lt__(self, other):
        raise TypeError
    def __gt__(self, other):
        raise TypeError
    def __le__(self, other):
        raise TypeError
    def __ge__(self, other):
        raise TypeError

    def __add__(self, other):
        raise TypeError
    def __sub__(self, other):
        raise TypeError
    def __mul__(self, other):
        raise TypeError
    def __pow__(self, other):
        raise TypeError
    def __truediv__(self, other):
        raise TypeError
    def __floordiv__(self, other):
        raise TypeError
    def __mod__(self, other):
        raise TypeError

    def __and__(self, other):
        raise TypeError
    def __or__(self, other):
        raise TypeError
    def __xor__(self, other):
        raise TypeError

    def _binary_op(self, other, smt_op, py_op=None, expected_sort=None):
        #debug(f'binary op ({smt_op}) on value of type {type(other)}')
        left = self.var
        if expected_sort is None:
            right = coerce_to_smt_var(self.statespace, other)[0]
        else:
            right = coerce_to_smt_sort(self.statespace, other, expected_sort)
            if right is None:
                return py_op(self._coerce_into_compatible(self), self._coerce_into_compatible(other))
        return self.__class__(self.statespace, self.python_type, smt_op(left, right))
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
            raise TypeError
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
        self.empty = z3.K(arr_var.sort().domain(), self.val_missing_constructor())
        self.statespace.add((arr_var == self.empty) == (len_var == 0))
    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        smt_key_type = type_to_smt_sort(self.key_pytype)
        smt_val_type = type_to_smt_sort(self.val_pytype)
        arr_smt_type = z3.ArraySort(smt_key_type, possibly_missing_sort(smt_val_type))
        return (
            z3.Const(varname+'_map' + self.statespace.uniq(), arr_smt_type),
            z3.Const(varname + '_len' + self.statespace.uniq(), z3.IntSort())
        )
    def __eq__(self, other):
        (self_arr, self_len) = self.var
        has_heapref = smt_sort_has_heapref(self.var[1].sort()) or smt_sort_has_heapref(self.var[0].sort())
        if not has_heapref:
            if isinstance(other, SmtDict):
                (other_arr, other_len) = other.var
                return SmtBool(self.statespace, bool, z3.And(self_len == other_len, self_arr == other_arr))
        # Manually check equality. Drive the loop from the (likely) concrete value 'other':
        if len(self) != len(other):
            return False
        for k, v in other.items():
            if k not in self or self[k] != v:
                return False
        return True
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
            smt_key, _ = coerce_to_smt_var(self.statespace, k)
            debug('lookup', self._arr().sort(), smt_key)
            possibly_missing = self._arr()[smt_key]
            is_missing = self.val_missing_checker(possibly_missing)
            if SmtBool(self.statespace, bool, is_missing).__bool__():
                raise KeyError(k)
            if SmtBool(self.statespace, bool, self._len() == 0).__bool__():
                raise IgnoreAttempt('SmtDict in inconsistent state')
            return smt_to_ch_value(self.statespace,
                                   self.snapshot,
                                   self.val_accessor(possibly_missing),
                                   self.val_pytype)
    def __iter__(self):
        arr_var, len_var = self.var
        idx = 0
        arr_sort = self._arr().sort()
        missing = self.val_missing_constructor()
        while SmtBool(self.statespace, bool, idx < len_var).__bool__():
            if SmtBool(self.statespace, bool, arr_var == self.empty).__bool__():
                raise IgnoreAttempt('SmtDict in inconsistent state')
            k = z3.Const('k'+str(idx) + self.statespace.uniq(), arr_sort.domain())
            v = z3.Const('v'+str(idx) + self.statespace.uniq(), self.val_constructor.domain(0))
            remaining = z3.Const('remaining' + str(idx) + self.statespace.uniq(), arr_sort)
            idx += 1
            self.statespace.add(arr_var == z3.Store(remaining, k, self.val_constructor(v)))
            self.statespace.add(z3.Select(remaining, k) == missing)
            yield smt_to_ch_value(self.statespace,
                                  self.snapshot,
                                  k,
                                  self.key_pytype)
            arr_var = remaining
        # In this conditional, we reconcile the parallel symbolic variables for length
        # and contents:
        if SmtBool(self.statespace, bool, arr_var != self.empty).__bool__():
            raise IgnoreAttempt('SmtDict in inconsistent state')
    def copy(self):
        return SmtDict(self.statespace, self.python_type, self.var)

    # TODO: investigate this approach for type masquerading:
    #@property
    #def __class__(self):
    #    return dict


class SmtSet(SmtDictOrSet, collections.abc.Set):
    def __init__(self, statespace:StateSpace, typ:Type, smtvar:object):
        SmtDictOrSet.__init__(self, statespace, typ, smtvar)
        self.empty = z3.K(self._arr().sort().domain(), False)
        self.statespace.add((self._arr() == self.empty) == (self._len() == 0))
    def __eq__(self, other):
        (self_arr, self_len) = self.var
        if isinstance(other, SmtSet):
            (other_arr, other_len) = other.var
            return SmtBool(self.statespace, bool, z3.And(self_len == other_len, self_arr == other_arr))
        if not isinstance(other, (set, frozenset)):
            return False
        # Manually check equality. Drive the loop from the (likely) concrete value 'other':
        if len(self) != len(other):
            return False
        for item in other:
            if k not in self:
                return False
        return True
    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        return (
            z3.Const(varname+'_map' + self.statespace.uniq(),
                     z3.ArraySort(type_to_smt_sort(self.key_pytype),
                                  z3.BoolSort())),
            z3.Const(varname + '_len' + self.statespace.uniq(), z3.IntSort())
        )
    def __contains__(self, k):
        k = coerce_to_smt_sort(self.statespace, k, self._arr().sort().domain())
        # TODO: test k for nullness (it's a key type error)
        present = self._arr()[k]
        return SmtBool(self.statespace, bool, present)
    def __iter__(self):
        arr_var, len_var = self.var
        idx = 0
        arr_sort = self._arr().sort()
        while SmtBool(self.statespace, bool, idx < len_var).__bool__():
            if SmtBool(self.statespace, bool, arr_var == self.empty).__bool__():
                raise IgnoreAttempt('SmtSet in inconsistent state')
            k = z3.Const('k' + str(idx) + self.statespace.uniq(), arr_sort.domain())
            remaining = z3.Const('remaining' + str(idx) + self.statespace.uniq(), arr_sort)
            idx += 1
            self.statespace.add(arr_var == z3.Store(remaining, k, True))
            self.statespace.add(z3.Not(z3.Select(remaining, k)))
            yield smt_to_ch_value(self.statespace, self.snapshot, k, self.key_pytype)
            arr_var = remaining
        # In this conditional, we reconcile the parallel symbolic variables for length
        # and contents:
        if SmtBool(self.statespace, bool, arr_var != self.empty).__bool__():
            raise IgnoreAttempt('SmtSet in inconsistent state')

    # Hardwire boolean operators into abc methods
    # (SmtBackedValue defaults these operations into
    # TypeErrors, but must appear first in the mro)
    def __and__(self, other):
        return collections.abc.Set.__and__(self, other)
    def __or__(self, other):
        return collections.abc.Set.__or__(self, other)
    def __xor__(self, other):
        return collections.abc.Set.__xor__(self, other)


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
            # In theory, we could do this without the fork. But it's heavy for the solver, and
            # it breaks z3 sometimes? (unreachable @ crosshair.examples.showcase.duplicate_list)
            # return z3.If(idx >= 0, idx, smt_len + idx)
            if space.smt_fork(idx >= 0):
                return idx
            else:
                return smt_len + idx
    if isinstance(i, int) or isinstance(i, SmtInt):
        smt_i = smt_coerce(i)
        if space.smt_fork(z3.Or(smt_i >= smt_len, smt_i < -smt_len)):
            raise IndexError(f'index "{i}" is out of range')
        return normalize_symbolic_index(smt_i)
    elif isinstance(i, slice):
        smt_start, smt_stop, smt_step = map(smt_coerce, (i.start, i.stop, i.step))
        if smt_step not in (None, 1):
            raise CrosshairUnsupported('slice steps not handled')
        start = normalize_symbolic_index(smt_start) if i.start is not None else 0
        stop = normalize_symbolic_index(smt_stop) if i.stop is not None else smt_len
        return (start, stop)
    else:
        raise TypeError('indices must be integers or slices, not '+str(type(i)))

class SmtSequence(SmtBackedValue):
    def _smt_getitem(self, i):
        idx_or_pair = process_slice_vs_symbolic_len(self.statespace, i, z3.Length(self.var))
        if isinstance(idx_or_pair, tuple):
            (start, stop) = idx_or_pair
            return (z3.Extract(self.var, start, stop - start), True)
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
    def __init__(self, space:StateSpace, typ:Type, smtvar:object):
        assert origin_of(typ) in (tuple, list)
        SmtBackedValue.__init__(self, space, typ, smtvar)
        self.item_pytype = normalize_pytype(typ.__args__[0]) # (index 0 works for both List[T] and Tuple[T, ...])
        self.item_ch_type = crosshair_type_for_python_type(self.item_pytype)
        self.item_smt_sort = type_to_smt_sort(self.item_pytype)
    def __add__(self, other: Any) -> Any:
        return self._binary_op(other, lambda x, y: z3.Concat(x, y),
                               py_op=operator.add,
                               expected_sort=self.var.sort())
    def __radd__(self, other: Any) -> Any:
        return self._binary_op(other, lambda x, y: z3.Concat(y, x),
                               py_op=(lambda x, y: y + x),
                               expected_sort=self.var.sort())
    def __contains__(self, other):
        if smt_sort_has_heapref(self.item_smt_sort):
            # Fall back to standard equality and iteration
            for self_item in self:
                if self_item == other:
                    return True
            return False
        else:
            smt_item = coerce_to_smt_sort(self.statespace, other, self.item_smt_sort)
            # TODO: test smt_item nullness (incorrect type)
            return SmtBool(self.statespace, bool, z3.Contains(self.var, z3.Unit(smt_item)))
    def __getitem__(self, i):
        smt_result, is_slice = self._smt_getitem(i)
        if is_slice:
            return self.__class__(self.statespace, self.python_type, smt_result)
        else:
            return smt_to_ch_value(self.statespace, self.snapshot, smt_result, self.item_pytype)

class SmtList(SmtUniformListOrTuple, collections.abc.MutableSequence):
    def __repr__(self):
        return str(list(self))
    def extend(self, other):
        self.var = self.var + smt_coerce(other)
    def _coerce_into_compatible(self, other):
        debug(f'coercing value of type {type(other)}')
        if isinstance(other, collections.abc.MutableSequence):
            return [v for v in other]
        else:
            return super()._coerce_into_compatible(other)
    def __setitem__(self, idx, obj):
        space, var = self.statespace, self.var
        varlen = z3.Length(var)
        idx_or_pair = process_slice_vs_symbolic_len(space, idx, varlen)
        if isinstance(idx_or_pair, tuple):
            (start, stop) = idx_or_pair
            to_insert = coerce_to_smt_sort(space, obj, z3.SeqSort(self.item_smt_sort))
        else:
            (start, stop) = (idx_or_pair, idx_or_pair + 1)
            to_insert = z3.Unit(coerce_to_smt_sort(space, obj, self.item_smt_sort))
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
        self.var = z3.Concat(z3.Extract(var, 0, start), z3.Extract(var, stop, varlen - stop))
    def insert(self, idx, obj):
        space, var = self.statespace, self.var
        varlen = z3.Length(var)
        to_insert = z3.Unit(coerce_to_smt_sort(space, obj, self.item_smt_sort))
        if coerce_to_smt_var(space, idx)[0] == varlen:
            self.var = z3.Concat(var, to_insert)
        else:
            idx = process_slice_vs_symbolic_len(space, idx, varlen)
            self.var = z3.Concat(z3.Extract(var, 0, idx),
                                 to_insert,
                                 z3.Extract(var, idx, varlen - idx))
    def sort(self, **kw):
        self.var = coerce_to_smt_sort(self.statespace, sorted(self, **kw), self.var.sort())


class SmtCallable(SmtBackedValue):
    __closure__ = None
    def __init___(self, statespace:StateSpace, typ:Type, smtvar:object):
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
    def __eq__(self, other):
        return (self.var is other.var) if isinstance(other, SmtCallable) else False
    def __hash__(self):
        return id(self.var)
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
        return z3.Function(varname + self.statespace.uniq(),
                           *map(type_to_smt_sort, self.arg_pytypes),
                           type_to_smt_sort(self.ret_pytype))
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
            body = '{} if ({}) else ({})'.format(repr(model_value_to_python(entry[-1])),
                                                 ' and '.join(conditions),
                                                 body)
        return 'lambda ({}): {}'.format(', '.join(arg_names), body)

class SmtUniformTuple(SmtUniformListOrTuple, collections.abc.Sequence, collections.abc.Hashable):
    def _coerce_into_compatible(self, other):
        if isinstance(other, tuple):
            return tuple(other)
        else:
            return super()._coerce_into_compatible(other)
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


def dt_recognizer(dt: z3.z3.DatatypeSortRef, name: str) -> z3.z3.FuncDeclRef:
    for i in range(dt.num_constructors()):
        if name == dt.constructor(i).name():
            return dt.recognizer(i)
    raise CrosshairInternal(f'cannot find recognizer for {dt}')

_CACHED_TYPE_ENUMS:Dict[FrozenSet[type], z3.SortRef] = {}
def get_type_enum(types:FrozenSet[type]) -> z3.SortRef:
    ret = _CACHED_TYPE_ENUMS.get(types)
    if ret is not None:
        return ret
    datatype = z3.Datatype('typechoice_'+'_'.join(sorted(map(str, types))))
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
        var = z3.Const("type_"+str(varname), self.vartype)
        for typ in self.pytypes[:-1]:
            if SmtBool(statespace, bool, dt_recognizer(self.vartype, name_of_type(typ))(var)).__bool__():
                return proxy_for_type(typ, statespace, varname)
        return proxy_for_type(self.pytypes[-1], statespace, varname)


class SmtProxyMarker:
    pass

_SUBTYPES: Dict[type, type] = {}
def get_subtype(cls: type) -> type:
    if isinstance(cls, SmtProxyMarker):
        return cls
    global _SUBTYPES
    cls_name = name_of_type(cls)
    if cls not in _SUBTYPES:
        def symbolic_init(self):
            pass
        _SUBTYPES[cls] = type(cls_name + '_proxy', (SmtProxyMarker, cls), {
            '__init__': symbolic_init,
        })
    return _SUBTYPES[cls]

def SmtObject(statespace: StateSpace, cls: type, varname: str) -> object:
    proxy = get_subtype(cls)()
    for name, typ in get_type_hints(cls).items():
        origin = getattr(typ, '__origin__', None)
        if origin is Callable:
            continue
        value = proxy_for_type(typ, statespace, varname + '.' + name + statespace.uniq())
        object.__setattr__(proxy, name, value)
    return proxy

def make_raiser(exc, *a) -> Callable:
    def do_raise(*ra, **rkw) -> NoReturn:
        raise exc(*a)
    return do_raise

def choose_type(space: StateSpace, from_type: Type) -> Type:
    subtypes = get_subclass_map()[from_type]
    for subtype in subtypes:
        if space.smt_fork():
            choose_type(space, subtype)
    return from_type


_SIMPLE_PROXIES: MutableMapping[object, Callable] = {
    complex: lambda p: complex(p(float), p(float)),
    type(None): lambda p: None,
    slice: lambda p: slice(p(Optional[int]), p(Optional[int]), p(Optional[int])),

    # if the target is a non-generic class, create it directly (but possibly with proxy constructor arguments?)
    Any: lambda p: p(int),
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
    Deque: lambda p, t=Any: collections.deque(p(Tuple[t, ...])), # TODO: a custom impl in simplestructs.py
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

    # random.Random proxies can't be deepcopied.
    # Would be nice to have a natively implemented version.
    random.Random: lambda p: random.Random(p(int)),
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
    origin = origin_of(typ)
    # special cases
    if origin is tuple:
        if len(typ.__args__) == 2 and typ.__args__[1] == ...:
            return SmtUniformTuple(statespace, typ, varname)
        else:
            return tuple(proxy_for_type(t, statespace, varname +'[' + str(idx) + ']')
                         for (idx, t) in enumerate(typ.__args__))
    elif origin is type:
        base = typ.__args__[0] if hasattr(typ, '__args__') else object
        return choose_type(statespace, normalize_pytype(base))
    elif isinstance(typ, type) and issubclass(typ, enum.Enum):
        enum_values = list(typ) # type:ignore
        for enum_value in enum_values[:-1]:
            if statespace.smt_fork():
                return enum_value
        return enum_values[-1]
    elif isinstance(origin, type) and issubclass(origin, Mapping):
        if hasattr(typ, '__args__'):
            args = typ.__args__
            if smt_sort_has_heapref(type_to_smt_sort(args[0])):
                return SimpleDict(proxy_for_type(List[Tuple[args[0], args[1]]], statespace, varname)) # type: ignore
    proxy_factory = _SIMPLE_PROXIES.get(origin)
    if proxy_factory:
        recursive_proxy_factory = lambda t: proxy_for_type(t, statespace, varname + statespace.uniq())
        return proxy_factory(recursive_proxy_factory, *type_args_of(typ))
    Typ = crosshair_type_that_inhabits_python_type(typ)
    if Typ is not None:
        return Typ(statespace, typ, varname)
    # if the class has data members, we attempt to create a concrete instance with
    # symbolic members; otherwise, we'll create an object proxy that emulates it.
    ret = proxy_class_as_concrete(typ, statespace, varname)
    if ret is _MISSING:
        ret = SmtObject(statespace, typ, varname)
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
                debug('Could not assume invariant', inv_condition.expr_source, 'on proxy of', typ,
                      ' because it raised: ', repr(efilter.user_exc[0]))
                # if the invarants are messed up enough to be rasing exceptions, don't bother:
                return ret
            elif efilter.ignore or not isok:
                raise IgnoreAttempt('Class proxy did not meet invariant ', inv_condition.expr_source)
    return ret

def gen_args(sig: inspect.Signature, statespace:StateSpace) -> inspect.BoundArguments:
    args = sig.bind_partial()
    for param in sig.parameters.values():
        smt_name = param.name + statespace.uniq()
        has_annotation = (param.annotation != inspect.Parameter.empty)
        value: object
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            if has_annotation:
                varargs_type = List[param.annotation] # type: ignore
                value = proxy_for_type(varargs_type, statespace, smt_name)
            else:
                value = proxy_for_type(List[Any], statespace, smt_name)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            if has_annotation:
                varargs_type = Dict[str, param.annotation] # type: ignore
                value = cast(dict, proxy_for_type(varargs_type, statespace, smt_name))
                # Using ** on a dict requires concrete string keys. Force
                # instiantiation of keys here:
                value = {k.__str__(): v for (k,v) in value.items()}
            else:
                value = proxy_for_type(Dict[str, Any], statespace, smt_name)
        else:
            if has_annotation:
                value = proxy_for_type(param.annotation, statespace, smt_name)
            else:
                value = proxy_for_type(cast(type, Any), statespace, smt_name)
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
    IMPORT_ERR = 'import_err'
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
    MessageType.IMPORT_ERR: 6,
}

@dataclass(frozen=True)
class AnalysisMessage:
    state: MessageType
    message: str
    filename: str
    line: int
    column: int
    traceback: str
    execution_log: Optional[str] = None
    test_fn: Optional[str] = None
    condition_src: Optional[str] = None

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
    per_condition_timeout: float = 4.0
    deadline: float = float('NaN')
    per_path_timeout: float = 2.0
    stats: Optional[collections.Counter] = None
    def incr(self, key: str):
        if self.stats is not None:
            self.stats[key] += 1

_DEFAULT_OPTIONS = AnalysisOptions()

def analyzable_members(module:types.ModuleType) -> Iterator[Tuple[str, Union[Type, types.FunctionType]]]:
    module_name = module.__name__
    for name, member in inspect.getmembers(module):
        if not (inspect.isclass(member) or inspect.isfunction(member)):
            continue
        if member.__module__ != module_name:
            continue
        yield (name, member)

def analyze_any(entity:object, options:AnalysisOptions) -> List[AnalysisMessage]:
    if inspect.isclass(entity):
        return analyze_class(cast(Type,entity), options)
    elif inspect.isfunction(entity):
        self_class: Optional[type] = None
        fn = cast(types.FunctionType, entity)
        if fn.__name__ != fn.__qualname__:
            self_thing = walk_qualname(sys.modules[fn.__module__],
                                       fn.__qualname__.split('.')[-2])
            assert isinstance(self_thing, type)
            self_class = self_thing
        return analyze_function(fn, options, self_type = self_class)
    elif inspect.ismodule(entity):
        return analyze_module(cast(types.ModuleType, entity), options)
    else:
        raise CrosshairInternal('Entity type not analyzable: ' + str(type(entity)))

def analyze_module(module:types.ModuleType, options:AnalysisOptions) -> List[AnalysisMessage]:
    debug('Analyzing module ', module)
    messages = MessageCollector()
    for (name, member) in analyzable_members(module):
        messages.extend(analyze_any(member, options))
    message_list = messages.get()
    debug('Module', module.__name__, 'has', len(message_list), 'messages')
    return message_list

def message_class_clamper(cls: type):
    '''
    We clamp messages for a clesses method to appear on the class itself.
    So, even if the method is defined on a superclass, or defined dynamically (via
    decorator etc), we report it on the class definition instead.
    '''
    cls_file = inspect.getsourcefile(cls)
    (lines, cls_start_line) = inspect.getsourcelines(cls)
    def clamp(message: AnalysisMessage):
        if not samefile(message.filename, cls_file):
            return replace(message, filename = cls_file, line = cls_start_line)
        else:
            return message
    return clamp

def analyze_class(cls:type, options:AnalysisOptions=_DEFAULT_OPTIONS) -> List[AnalysisMessage]:
    messages = MessageCollector()
    class_conditions = get_class_conditions(cls)
    for method, conditions in class_conditions.methods.items():
        if conditions.has_any():
            cur_messages = analyze_function(getattr(cls, method),
                                             options=options,
                                             self_type=cls)
            clamper = message_class_clamper(cls)
            messages.extend(map(clamper, cur_messages))
        
    return messages.get()

_EMULATION_TIMEOUT_FRACTION = 0.2
def analyze_function(fn:FunctionLike,
                     options:AnalysisOptions=_DEFAULT_OPTIONS,
                     self_type:Optional[type]=None) -> List[AnalysisMessage]:
    debug('Analyzing ', fn.__name__)
    all_messages = MessageCollector()

    if self_type is not None:
        class_conditions = get_class_conditions(self_type)
        conditions = class_conditions.methods[fn.__name__]
    else:
        conditions = get_fn_conditions(fn, self_type=self_type)
        if conditions is None:
            debug('Skipping ', str(fn), ': Unable to determine the function signature.')
            return []

    for syntax_message in conditions.syntax_messages():
        all_messages.append(AnalysisMessage(MessageType.SYNTAX_ERR,
                                            syntax_message.message,
                                            syntax_message.filename,
                                            syntax_message.line_num, 0, ''))
    conditions = conditions.compilable()
    for post_condition in conditions.post:
        messages = analyze_single_condition(fn, options, replace(conditions, post=[post_condition]), self_type)
        all_messages.extend(messages)
    return all_messages.get()

def analyze_single_condition(fn:FunctionLike,
                             options:AnalysisOptions,
                             conditions:Conditions,
                             self_type:Optional[type]) -> Sequence[AnalysisMessage]:
    debug('Analyzing postcondition: "', conditions.post[0].expr_source, '"')
    debug('assuming preconditions: ', ','.join([p.expr_source for p in conditions.pre]))
    if options.use_called_conditions: 
        options.deadline = time.time() + options.per_condition_timeout * _EMULATION_TIMEOUT_FRACTION
    else:
        options.deadline = time.time() + options.per_condition_timeout
    
    analysis = analyze_calltree(fn, options, conditions)
    if (options.use_called_conditions and
        analysis.verification_status < VerificationStatus.CONFIRMED):
        debug('Reattempting analysis without short circuiting')
        options = replace(options,
                          use_called_conditions=False,
                          deadline=time.time() + options.per_condition_timeout * (1.0 - _EMULATION_TIMEOUT_FRACTION))
        analysis = analyze_calltree(fn, options, conditions)

    (condition,) = conditions.post
    if analysis.verification_status is VerificationStatus.UNKNOWN:
        addl_ctx = ' ' + condition.addl_context if condition.addl_context else ''
        message = 'I cannot confirm this' + addl_ctx
        analysis.messages = [AnalysisMessage(MessageType.CANNOT_CONFIRM, message,
                                             condition.filename, condition.line, 0, '')]
        
    return analysis.messages

def forget_contents(value:object, space:StateSpace):
    if isinstance(value, SmtBackedValue):
        clean_smt = type(value)(space, value.python_type, str(value.var) + space.uniq())
        value.var = clean_smt.var
    elif isinstance(value, SmtProxyMarker):
        cls = type(value).__bases__[0]
        clean = proxy_for_type(cls, space, space.uniq())
        for name, val in value.__dict__.items():
            value.__dict__[name] = clean.__dict__[name]
    else:
        for subvalue in value.__dict__.values():
            forget_contents(subvalue, space)

class ShortCircuitingContext:
    engaged = False
    intercepted = False
    def __init__(self, space_getter: Callable[[], StateSpace]):
        self.space_getter = space_getter
    def __enter__(self):
        assert not self.engaged
        self.engaged = True
    def __exit__(self, exc_type, exc_value, tb):
        assert self.engaged
        self.engaged = False
    def make_interceptor(self, original:Callable) -> Callable:
        subconditions = get_fn_conditions(original)
        if subconditions is None:
            return original
        sig = subconditions.sig
        def wrapper(*a:object, **kw:Dict[str, object]) -> object:
            #debug('intercept wrapper ', original, self.engaged)
            if not self.engaged or self.space_getter().running_framework_code:
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
                assert subconditions is not None
                if subconditions.mutable_args:
                    bound = sig.bind(*a, **kw)
                    for mutated_arg in subconditions.mutable_args:
                        forget_contents(bound.arguments[mutated_arg], self.space_getter())

                if return_type is type(None):
                    return None
                # note that the enforcement wrapper ensures postconditions for us, so we
                # can just return a free variable here.
                return proxy_for_type(return_type, self.space_getter(), 'proxyreturn' + self.space_getter().uniq())
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

def replay(fn: FunctionLike,
           message: AnalysisMessage,
           conditions: Conditions) -> CallAnalysis:
    debug('replay log', message.test_fn, message.execution_log)
    assert message.execution_log is not None
    assert fn.__qualname__ == message.test_fn
    conditions = replace(conditions, post=[c for c in conditions.post
                                           if c.expr_source == message.condition_src])
    space = ReplayStateSpace(message.execution_log)
    short_circuit = ShortCircuitingContext(lambda : space)
    envs = [fn_globals(fn), contracted_builtins.__dict__]
    enforced_conditions = EnforcedConditions(*envs)
    in_symbolic_mode = lambda : not space.running_framework_code
    patched_builtins = PatchedBuiltins(contracted_builtins.__dict__, in_symbolic_mode)
    with enforced_conditions, patched_builtins:
        return attempt_call(conditions, space, fn, short_circuit, enforced_conditions)

def analyze_calltree(fn: FunctionLike,
                     options: AnalysisOptions,
                     conditions: Conditions) -> CallTreeAnalysis:
    debug('Begin analyze calltree ', fn.__name__,
          ' short circuit=', options.use_called_conditions)

    worst_verification_status = VerificationStatus.CONFIRMED
    all_messages = MessageCollector()
    search_history = SearchTreeNode()
    space_exhausted = False
    failing_precondition: Optional[ConditionExpr] = conditions.pre[0] if conditions.pre else None
    num_confirmed_paths = 0

    cur_space :List[StateSpace] = [cast(StateSpace, None)]
    short_circuit = ShortCircuitingContext(lambda : cur_space[0])
    envs = [fn_globals(fn), contracted_builtins.__dict__]
    interceptor = (short_circuit.make_interceptor if options.use_called_conditions else lambda f:f)
    _ = get_subclass_map() # ensure loaded
    enforced_conditions = EnforcedConditions(*envs, interceptor=interceptor)
    in_symbolic_mode = lambda : cur_space[0] is not None and not cur_space[0].running_framework_code
    patched_builtins = PatchedBuiltins(contracted_builtins.__dict__, in_symbolic_mode)
    with enforced_conditions, patched_builtins:
        for i in itertools.count(1):
            start = time.time()
            if start > options.deadline:
                break
            options.incr('num_paths')
            debug('iteration ', i)
            space = TrackingStateSpace(execution_deadline = start + options.per_path_timeout,
                                       model_check_timeout = options.per_path_timeout / 2,
                                       previous_searches = search_history)
            cur_space[0] = space
            try:
                # The real work happens here!:
                call_analysis = attempt_call(conditions, space, fn, short_circuit, enforced_conditions)

                if failing_precondition is not None:
                    cur_precondition = call_analysis.failing_precondition
                    if cur_precondition is None:
                        if call_analysis.verification_status is not None:
                            # We escaped the all the pre conditions on this try:
                            failing_precondition = None
                    elif cur_precondition.line > failing_precondition.line:
                        failing_precondition = cur_precondition
        
            except (UnexploredPath, CrosshairUnsupported):
                call_analysis = CallAnalysis(VerificationStatus.UNKNOWN)
            except IgnoreAttempt:
                call_analysis = CallAnalysis()
            exhausted = space.check_exhausted()
            debug('iter complete', call_analysis.verification_status.name
                  if call_analysis.verification_status else 'None',
                  ' (previous worst:', worst_verification_status.name, ')')
            if call_analysis.verification_status is not None:
                if call_analysis.verification_status == VerificationStatus.CONFIRMED:
                    num_confirmed_paths += 1
                else:
                    worst_verification_status = min(call_analysis.verification_status, worst_verification_status)
                if call_analysis.messages:
                    log = space.execution_log()
                    all_messages.extend(
                        replace(m,
                                execution_log=log,
                                test_fn=fn.__qualname__,
                                condition_src=conditions.post[0].expr_source)
                        for m in call_analysis.messages)
            if exhausted:
                # we've searched every path
                space_exhausted = True
                break
            if worst_verification_status <= VerificationStatus.REFUTED: #type: ignore
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

    debug(('Exhausted' if space_exhausted else 'Aborted'),
          ' calltree search with', worst_verification_status.name,
          '. Number of iterations: ', i+1)
    return CallTreeAnalysis(messages = all_messages.get(),
                            verification_status = worst_verification_status,
                            num_confirmed_paths = num_confirmed_paths)

def python_string_for_evaluated(expr:z3.ExprRef)->str:
    return str(expr)

def get_input_description(statespace:StateSpace,
                          bound_args:inspect.BoundArguments,
                          addl_context:str = '') -> str:
    messages:List[str] = []
    for argname, argval in list(bound_args.arguments.items()):
        try:
            repr_str = repr(argval)
        except Exception as e:
            debug(f'Exception attempting to repr input "{argname}": {e}')
            repr_str = '<unable to repr>'
        messages.append(argname + ' = ' + repr_str)
    if addl_context:
        return addl_context + ' with ' + ' and '.join(messages)
    elif messages:
        return 'when ' + ' and '.join(messages)
    else:
        return 'for any input'

class UnEqual:
    pass
_UNEQUAL = UnEqual()

def deep_eq(old_val: object, new_val: object, visiting: Set[Tuple[int, int]]) -> bool:
    # TODO: test just about all of this
    if old_val is new_val:
        return True
    if type(old_val) != type(new_val):
        return False
    visit_key = (id(old_val), id(new_val))
    if visit_key in visiting:
        return True
    visiting.add(visit_key)
    try:
        if isinstance(old_val, SmtBackedValue):
            return old_val == new_val
        elif hasattr(old_val, '__dict__') and hasattr(new_val, '__dict__'):
            return deep_eq(old_val.__dict__, new_val.__dict__, visiting)
        elif isinstance(old_val, dict):
            assert isinstance(new_val, dict)
            for key in set(itertools.chain(old_val.keys(), *new_val.keys())):
                if (key in old_val) ^ (key in new_val):
                    return False
                if not deep_eq(old_val.get(key, _UNEQUAL), new_val.get(key, _UNEQUAL), visiting):
                    return False
            return True
        elif isinstance(old_val, Iterable):
            assert isinstance(new_val, Iterable)
            if isinstance(old_val, Sized):
                if len(old_val) != len(new_val):
                    return False
            return all(deep_eq(o, n, visiting) for (o, n) in
                       itertools.zip_longest(old_val, new_val, fillvalue=_UNEQUAL))
        else:  # hopefully just ints, bools, etc
            return old_val == new_val
    finally:
        visiting.remove(visit_key)

def attempt_call(conditions: Conditions,
                 space: StateSpace,
                 fn: FunctionLike,
                 short_circuit: ShortCircuitingContext,
                 enforced_conditions: EnforcedConditions) -> CallAnalysis:
    bound_args = gen_args(conditions.sig, space)

    code_obj = cast(types.FunctionType, fn).__code__
    fn_filename, fn_start_lineno = (code_obj.co_filename, code_obj.co_firstlineno)
    try:
        (lines, _) = inspect.getsourcelines(fn)
    except OSError:
        lines = []
    fn_end_lineno = fn_start_lineno + len(lines)
    def locate_msg(detail: str, suggested_filename: str, suggested_lineno: int) -> Tuple[str, str, int, int]:
        if ((os.path.abspath(suggested_filename) == os.path.abspath(fn_filename)) and
            (fn_start_lineno <= suggested_lineno <= fn_end_lineno)):
            return (detail, suggested_filename, suggested_lineno, 0)
        else:
            try:
                exprline = linecache.getlines(suggested_filename)[suggested_lineno - 1].strip()
            except IndexError:
                exprline = '<unknown>'
            detail = f'Line "{exprline}" yields {detail}'
            return (detail, fn_filename, fn_start_lineno, 0)
        
    original_args = copy.deepcopy(bound_args)
    space.checkpoint()

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
            return efilter.analysis
        elif efilter.user_exc is not None:
            debug('Exception attempting to meet precondition',
                  precondition.expr_source, ':',
                  efilter.user_exc[0],
                  efilter.user_exc[1].format())
            return CallAnalysis(failing_precondition=precondition)

    with ExceptionFilter() as efilter:
        a, kw = bound_args.args, bound_args.kwargs
        with short_circuit:
            assert not space.running_framework_code
            __return__ = fn(*a, **kw)
        lcls = {**bound_args.arguments,
                '__return__': __return__,
                '_': __return__,
                '__old__': AttributeHolder(original_args.arguments),
                fn.__name__: fn}
    if efilter.ignore:
        return efilter.analysis
    elif efilter.user_exc is not None:
        (e, tb) = efilter.user_exc
        detail = name_of_type(type(e)) + ': ' + str(e) + ' ' + get_input_description(space, original_args)
        frame_filename, frame_lineno = frame_summary_for_fn(tb, fn)
        return CallAnalysis(VerificationStatus.REFUTED,
                            [AnalysisMessage(MessageType.EXEC_ERR,
                                             *locate_msg(detail, frame_filename, frame_lineno),
                                             ''.join(tb.format()))])

    for argname, argval in bound_args.arguments.items():
        if argname not in conditions.mutable_args:
            old_val, new_val = original_args.arguments[argname], argval
            if not deep_eq(old_val, new_val, set()):
                detail = 'Argument "{}" is not marked as mutable, but changed from {} to {}'.format(argname, old_val, new_val)
                return CallAnalysis(VerificationStatus.REFUTED,
                                    [AnalysisMessage(MessageType.POST_ERR, detail, fn_filename, fn_start_lineno, 0, '')])

    (post_condition,) = conditions.post
    with ExceptionFilter() as efilter:
        with enforced_conditions.currently_enforcing(fn):
            eval_vars = {**fn_globals(fn), **lcls}
            with short_circuit:
                assert post_condition.expr is not None
                isok = bool(eval(post_condition.expr, eval_vars))
    with enforced_conditions.disabled_enforcement():
        if efilter.ignore:
            return efilter.analysis
        elif efilter.user_exc is not None:
            (e, tb) = efilter.user_exc
            detail = repr(e) + ' ' + get_input_description(space, original_args, post_condition.addl_context)
            failures=[AnalysisMessage(MessageType.POST_ERR,
                                      *locate_msg(detail, post_condition.filename, post_condition.line),
                                      ''.join(tb.format()))]
            return CallAnalysis(VerificationStatus.REFUTED, failures)
        if isok:
            return CallAnalysis(VerificationStatus.CONFIRMED)
        else:
            detail = 'false ' + get_input_description(space, original_args, post_condition.addl_context)
            failures = [AnalysisMessage(MessageType.POST_FAIL,
                                        *locate_msg(detail, post_condition.filename, post_condition.line), '')]
            return CallAnalysis(VerificationStatus.REFUTED, failures)

_PYTYPE_TO_WRAPPER_TYPE = {
    type(None): (lambda *a: None),
    bool: SmtBool,
    int: SmtInt,
    float: SmtFloat,
    str: SmtStr,
    list: SmtList,
    dict: SmtDict,
    set: SmtMutableSet,
    frozenset: SmtFrozenSet,
}

# Type ignore pending https://github.com/python/mypy/issues/6864
_PYTYPE_TO_WRAPPER_TYPE[collections.abc.Callable] = SmtCallable # type:ignore

_WRAPPER_TYPE_TO_PYTYPE = dict((v,k) for (k,v) in _PYTYPE_TO_WRAPPER_TYPE.items())
