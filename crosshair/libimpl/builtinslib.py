import collections
import contextlib
import copy
import enum
import functools
import io
import math
from numbers import Number, Complex, Real, Rational, Integral
import operator as ops
import re
import typing
from typing import *
import builtins as orig_builtins

from crosshair.abcstring import AbcString
from crosshair.core import register_patch
from crosshair.core import register_type
from crosshair.core import realize
from crosshair.core import proxy_for_type
from crosshair.core import python_type
from crosshair.core import normalize_pytype
from crosshair.core import choose_type
from crosshair.core import CrossHairValue
from crosshair.core import SmtProxyMarker
from crosshair.core import type_arg_of
from crosshair.core import type_args_of
from crosshair.core import name_of_type
from crosshair.core import with_realized_args
from crosshair.objectproxy import ObjectProxy
from crosshair.simplestructs import SimpleDict
from crosshair.simplestructs import SequenceConcatenation
from crosshair.simplestructs import SliceView
from crosshair.simplestructs import ShellMutableSequence
from crosshair.statespace import StateSpace
from crosshair.statespace import HeapRef
from crosshair.statespace import SnapshotRef
from crosshair.statespace import model_value_to_python
from crosshair.type_repo import PYTYPE_SORT
from crosshair.util import debug
from crosshair.util import CrosshairInternal
from crosshair.util import CrosshairUnsupported
from crosshair.util import IgnoreAttempt
from crosshair.util import is_iterable
from crosshair.util import is_hashable

import typing_inspect # type: ignore
import z3 # type: ignore

class _Missing(enum.Enum):
    value = 0

_MISSING = _Missing.value


def smt_min(x, y):
    if x is y:
        return x
    return z3.If(x <= y, x, y)

def smt_sort_has_heapref(sort: z3.SortRef) -> bool:
    return 'HeapRef' in str(sort)  # TODO: don't do this :)

_HEAPABLE_PYTYPES = set([int, float, str, bool, type(None), complex])

def pytype_uses_heap(typ: Type) -> bool:
    return not (typ in _HEAPABLE_PYTYPES)

def typeable_value(val: object) -> object:
    '''
    Foces values of unknown type (SmtObject) into a typed (but possibly still symbolic) value.
    '''
    while type(val) is SmtObject:
        val = cast(SmtObject, val)._wrapped()
    return val

_SMT_FLOAT_SORT = z3.RealSort()  # difficulty getting the solver to use z3.Float64()

_TYPE_TO_SMT_SORT = {
    bool: z3.BoolSort(),
    str: z3.StringSort(),
    int: z3.IntSort(),
    float: _SMT_FLOAT_SORT,
}


def possibly_missing_sort(sort):
    datatype = z3.Datatype('optional_' + str(sort) + '_')
    datatype.declare('missing')
    datatype.declare('present', ('valueat', sort))
    ret = datatype.create()
    return ret


def type_to_smt_sort(t: Type) -> z3.SortRef:
    t = normalize_pytype(t)
    if t in _TYPE_TO_SMT_SORT:
        return _TYPE_TO_SMT_SORT[t]
    origin = origin_of(t)
    if origin is type:
        return PYTYPE_SORT
    return HeapRef

SmtGenerator = Callable[[StateSpace, type, Union[str, z3.ExprRef]], object]

_PYTYPE_TO_WRAPPER_TYPE: Dict[type, SmtGenerator] = {}  # to be populated later
_WRAPPER_TYPE_TO_PYTYPE: Dict[SmtGenerator, type] = {}

def origin_of(typ: Type) -> Type:
    typ = _WRAPPER_TYPE_TO_PYTYPE.get(typ, typ)
    if hasattr(typ, '__origin__'):
        return typ.__origin__
    return typ

def crosshair_type_for_python_type(typ: Type) -> Optional[SmtGenerator]:
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

_IMPLICIT_SORT_CONVERSIONS: Dict[Tuple[z3.SortRef, z3.SortRef], Callable[[z3.ExprRef], z3.ExprRef]] = {
    (z3.BoolSort(), z3.IntSort()): smt_bool_to_int,
    (z3.BoolSort(), _SMT_FLOAT_SORT): smt_bool_to_float,
    (z3.IntSort(), _SMT_FLOAT_SORT): smt_int_to_float,
}

_LITERAL_PROMOTION_FNS = {
    bool: z3.BoolVal,
    int: z3.IntVal,
    float: z3.RealVal if _SMT_FLOAT_SORT == z3.RealSort() else (lambda v: z3.FPVal(v, _SMT_FLOAT_SORT)),
    str: z3.StringVal,
}

def smt_coerce(val: Any) -> z3.ExprRef:
    if isinstance(val, SmtBackedValue):
        return val.var
    return val

def force_to_smt_sort(space: StateSpace, input_value: Any, desired_sort: z3.SortRef) -> z3.ExprRef:
    ret = coerce_to_smt_sort(space, input_value, desired_sort)
    if ret is None:
        raise TypeError('Could not derive smt sort ' + str(desired_sort))
    return ret

def coerce_to_smt_sort(space: StateSpace, input_value: Any, desired_sort: z3.SortRef) -> Optional[z3.ExprRef]:
    natural_value = None
    input_value = typeable_value(input_value)
    promotion_fn = _LITERAL_PROMOTION_FNS.get(type(input_value))
    if isinstance(input_value, SmtBackedValue):
        natural_value = input_value.var
        if type(natural_value) is tuple:
            # Many container types aren't described by a single z3 value:
            return None
    elif promotion_fn:
        natural_value = promotion_fn(input_value)
    elif isinstance(input_value, z3.ExprRef):
        natural_value = input_value
    natural_sort = natural_value.sort() if natural_value is not None else None
    conversion_fn = _IMPLICIT_SORT_CONVERSIONS.get((natural_sort, desired_sort))
    if conversion_fn:
        return conversion_fn(natural_value)
    if natural_sort == desired_sort:
        return natural_value
    if desired_sort == HeapRef:
        return space.find_val_in_heap(input_value)
    if desired_sort == PYTYPE_SORT and isinstance(input_value, type):
        return space.type_repo.get_type(input_value)
    return None


def coerce_to_smt_var(space: StateSpace, v: Any) -> z3.ExprRef:
    v = typeable_value(v)
    if isinstance(v, SmtBackedValue):
        return v.var
    promotion_fn = _LITERAL_PROMOTION_FNS.get(type(v))
    if promotion_fn:
        return promotion_fn(v)
    return space.find_val_in_heap(v)


def smt_to_ch_value(space: StateSpace, snapshot: SnapshotRef, smt_val: z3.ExprRef, pytype: type) -> object:
    def proxy_generator(typ: Type) -> object:
        return proxy_for_type(typ, space, 'heapval' + str(typ) + space.uniq())
    if smt_val.sort() == HeapRef:
        return space.find_key_in_heap(smt_val, pytype, proxy_generator, snapshot)
    ch_type = crosshair_type_for_python_type(pytype)
    assert ch_type is not None
    return ch_type(space, pytype, smt_val)


def attr_on_ch_value(other: Any, statespace: StateSpace, attr: str) -> object:
    if not isinstance(other, CrossHairValue):
        smt_var = coerce_to_smt_var(statespace, other)
        py_type = python_type(other)
        Typ = crosshair_type_for_python_type(py_type)
        if Typ is None:
            raise TypeError
        other = Typ(statespace, py_type, smt_var)
    if not hasattr(other, attr):
        raise TypeError
    return getattr(other, attr)

class SmtBackedValue(CrossHairValue):
    def __init__(self, statespace: StateSpace, typ: Type, smtvar: object):
        self.statespace = statespace
        self.snapshot = SnapshotRef(-1)
        self.python_type = typ
        if isinstance(smtvar, str):
            self.var = self.__init_var__(typ, smtvar)
        else:
            self.var = smtvar
            # TODO test that smtvar's sort matches expected?

    def __init_var__(self, typ, varname):
        z3type = type_to_smt_sort(typ)
        return z3.Const(varname, z3type)

    def __deepcopy__(self, memo):
        shallow = copy.copy(self)
        shallow.snapshot = self.statespace.current_snapshot()
        return shallow

    def __bool__(self):
        return NotImplemented

    def __eq__(self, other):
        coerced = coerce_to_smt_sort(self.statespace, other, self.var.sort())
        if coerced is None:
            return False
        return SmtBool(self.statespace, bool, self.var == coerced)

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

    def __pow__(self, other, mod=None):
        raise TypeError

    def __truediv__(self, other):
        return numeric_binop(ops.truediv, self, other)

    def __floordiv__(self, other):
        raise TypeError

    def __mod__(self, other):
        raise TypeError

    def __ch_pytype__(self):
        return self.python_type

    def __ch_realize__(self):
        return origin_of(self.python_type)(self)

    def __ch_forget_contents__(self, space: StateSpace):
        clean_smt = type(self)(space, self.python_type,
                               str(self.var) + space.uniq())
        self.var = clean_smt.var

    def _binary_op(self, other, smt_op, py_op=None, expected_sort=None):
        #debug(f'binary op ({smt_op}) on value of type {type(other)}')
        left = self.var
        if expected_sort is None:
            expected_sort = type_to_smt_sort(self.python_type)
        right = coerce_to_smt_sort(self.statespace, other, expected_sort)
        if right is None:
            return py_op(realize(self), realize(other))
        try:
            ret = smt_op(left, right)
        except z3.z3types.Z3Exception as e:
            debug('Raising z3 error as Python TypeError: ', str(e))
            raise TypeError
        return self.__class__(self.statespace, self.python_type, ret)

    def _unary_op(self, op):
        return self.__class__(self.statespace, self.python_type, op(self.var))







# The Python numeric tower is (at least to me) fairly confusing.
# A summary here, with some example implementations:
#
# Number
# |
# Complex
# | \- complex
# Real
# | \- float
# Rational
# | \- Fraction
# Integral
# |
# int
# |
# bool   (yes, bool is a subtype of int!)
#

TypePair = Tuple[type, type]
BinFn = Callable[[Number, Number], Number]
OpHandler = Union[_Missing, Callable[[BinFn, Number, Number], Number]]

_BIN_OPS: Dict[Tuple[BinFn, type, type], OpHandler] = {}
_BIN_OPS_SEARCH_ORDER: List[Tuple[BinFn, type, type, OpHandler]] = []

class FiniteFloat(float):
    pass

def numeric_binop(op: BinFn, a: Number, b: Number):
    a_type, b_type = type(a), type(b)
    binfn = _BIN_OPS.get((op, a_type, b_type))
    if binfn is None:
        for curop, cur_a_type, cur_b_type, curfn in reversed(_BIN_OPS_SEARCH_ORDER):
            if op != curop:
                continue
            if (issubclass(a_type, cur_a_type) and
                issubclass(b_type, cur_b_type)):
                _BIN_OPS[(op, a_type, b_type)] = curfn  # cache concrete types for later
                binfn = curfn
                break
        if binfn is None:
            binfn = _MISSING
            _BIN_OPS[(op, a_type, b_type)] = _MISSING
    if binfn is _MISSING:
        return NotImplemented
    return binfn(op, a, b)

def _binop_type_hints(fn: Callable):
    hints = get_type_hints(fn)
    a, b = hints['a'], hints['b']
    if typing_inspect.get_origin(a) == Union:
        a = typing_inspect.get_args(a)
    else:
        a = [a]
    if typing_inspect.get_origin(b) == Union:
        b = typing_inspect.get_args(b)
    else:
        b = [b]
    return (a, b)

def setup_promotion(fn: Callable[[Number, Number], Tuple[Number, Number]], reg_ops: Set[BinFn]):
    a, b = _binop_type_hints(fn)
    for a_type in a:
        for b_type in b:
            for op in reg_ops:
                _BIN_OPS_SEARCH_ORDER.append((op, a_type, b_type, lambda o, x, y: o(*fn(x, y))))
                def symmetric(o, x, y):
                    y2, x2 = fn(y, x)
                    return o(x2, y2)
                _BIN_OPS_SEARCH_ORDER.append((op, b_type, a_type, symmetric))

_FLIPPED_OPS = {ops.ge: ops.le, ops.gt: ops.lt, ops.le: ops.ge, ops.lt: ops.gt}
def setup_binop(fn: Callable[[BinFn, Number, Number], Number], reg_ops: Set[BinFn]):
    a, b = _binop_type_hints(fn)
    for a_type in a:
        for b_type in b:
            for op in reg_ops:
                _BIN_OPS_SEARCH_ORDER.append((op, a_type, b_type, fn))

                # Also, handle flipped comparisons transparently:
                ## (a >= b)   <==>   (b <= a)
                if op in (ops.ge, ops.gt, ops.le, ops.lt):
                    def flipped(o, x, y):
                        return fn(_FLIPPED_OPS[o], y, x)
                    _BIN_OPS_SEARCH_ORDER.append((_FLIPPED_OPS[op], b_type, a_type, flipped))


_COMPARISON_OPS: Set[BinFn] = {
    ops.eq,
    ops.ne,
    ops.ge,
    ops.gt,
    ops.le,
    ops.lt,
}
_ARITHMETIC_OPS: Set[BinFn] = {
    ops.add,
    ops.sub,
    ops.mul,
    ops.truediv,
    ops.floordiv,
    ops.mod,
    ops.pow,
}
_BITWISE_OPS: Set[BinFn] = {
    ops.and_,
    ops.or_,
    ops.xor,
    ops.rshift,
    ops.lshift,
}

def apply_smt(op: BinFn, x: z3.ExprRef, y: z3.ExprRef, space: StateSpace) -> z3.ExprRef:
    # Mostly, z3 overloads operators and things just work.
    # But some special cases need to be checked first.
    if op in _ARITHMETIC_OPS:
        if op in(ops.truediv, ops.floordiv, ops.mod):
            if space.smt_fork(y == 0):
                raise ZeroDivisionError('division by zero')
            if op == ops.floordiv:
                return z3.If(
                    x % y == 0 or x >= 0, x / y,
                    z3.If(y >= 0, x / y + 1, x / y - 1))
        elif op == ops.pow:
            if space.smt_fork(z3.And(x == 0, y < 0)):
                raise ZeroDivisionError('zero cannot be raised to a negative power')
    elif op in _BITWISE_OPS:
        if op in (ops.lshift, ops.rshift):
            if space.smt_fork(y < 0):
                raise ValueError('negative shift count')
            return x * (2 ** y) if op == ops.lshift else x / (2 ** y)
    return op(x, y)


_ARITHMETIC_AND_BITWISE_OPS = _ARITHMETIC_OPS.union(_BITWISE_OPS)
_ARITHMETIC_AND_COMPARISON_OPS = _ARITHMETIC_OPS.union(_COMPARISON_OPS)
_ALL_OPS = _ARITHMETIC_AND_COMPARISON_OPS.union(_BITWISE_OPS)

def setup_binops():

    # We check NaN and infitity immediately; not all
    # symbolic floats support these cases.
    def _(a: Real, b: float):
        if math.isfinite(b):
            return (a, FiniteFloat(b))  # type: ignore
        if a > 0:  # type: ignore
            return (1, b)  # type: ignore
        elif a < 0:
            return (-1, b)  # type: ignore
        else:
            return (0, b)  # type: ignore
    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)


    # Implicitly upconvert symbolic bools to integers.
    # Note that we don't want this when `other` is a boolean, but that
    # case will be overridden in the booleans section below.
    def _(a: SmtBool, b: Number):
        return (SmtInt(a.statespace, int, z3.If(a.var, 1, 0)), b)
    setup_promotion(_, _ALL_OPS)

    # Implicitly upconvert symbolic ints to floats.
    def _(a: SmtInt, b: Union[float, FiniteFloat, SmtFloat, complex]):
        return (SmtFloat(a.statespace, float, z3.ToReal(a.var)), b)
    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # Implicitly upconvert symbolic numbers into complex values.
    def _(a: SmtNumberAble, b: complex):
        return (complex(a), b)
    setup_promotion(_, _ALL_OPS)

    # Implicitly upconvert native ints to floats.
    def _(a: int, b: SmtFloat):
        return (float(a), b)
    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # Implicitly upconvert native bools to ints.
    def _(a: bool, b: Union[SmtInt, SmtFloat]):
        return (int(a), b)
    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)


    # float
    def _(op: BinFn, a: SmtFloat, b: SmtFloat):
        return SmtFloat(a.statespace, float, apply_smt(op, a.var, b.var, a.statespace))
    setup_binop(_, _ARITHMETIC_OPS)
    def _(op: BinFn, a: SmtFloat, b: SmtFloat):
        return SmtBool(a.statespace, bool, apply_smt(op, a.var, b.var, a.statespace))
    setup_binop(_, _COMPARISON_OPS)
    def _(op: BinFn, a: SmtFloat, b: FiniteFloat):
        return SmtFloat(a.statespace, float, apply_smt(op, a.var, z3.RealVal(b), a.statespace))
    setup_binop(_, _ARITHMETIC_OPS)
    def _(op: BinFn, a: FiniteFloat, b: SmtFloat):
        return SmtFloat(b.statespace, float, apply_smt(op, z3.RealVal(a), b.var, b.statespace))
    setup_binop(_, _ARITHMETIC_OPS)
    def _(op: BinFn, a: SmtFloat, b: FiniteFloat):
        return SmtBool(a.statespace, bool, apply_smt(op, a.var, z3.RealVal(b), a.statespace))
    setup_binop(_, _COMPARISON_OPS)

    # int
    def _(op: BinFn, a: SmtInt, b: SmtInt):
        return SmtInt(a.statespace, int, apply_smt(op, a.var, b.var, a.statespace))
    setup_binop(_, _ARITHMETIC_AND_BITWISE_OPS)
    def _(op: BinFn, a: SmtInt, b: SmtInt):
        return SmtBool(a.statespace, bool, apply_smt(op, a.var, b.var, a.statespace))
    setup_binop(_, _COMPARISON_OPS)
    def _(op: BinFn, a: SmtInt, b: int):
        return SmtInt(a.statespace, int, apply_smt(op, a.var, z3.IntVal(b), a.statespace))
    setup_binop(_, _ARITHMETIC_AND_BITWISE_OPS)
    def _(op: BinFn, a: int, b: SmtInt):
        return SmtInt(b.statespace, int, apply_smt(op, z3.IntVal(a), b.var, b.statespace))
    setup_binop(_, _ARITHMETIC_AND_BITWISE_OPS)
    def _(op: BinFn, a: SmtInt, b: int):
        return SmtBool(a.statespace, bool, apply_smt(op, a.var, z3.IntVal(b), a.statespace))
    setup_binop(_, _COMPARISON_OPS)
    def _(op: BinFn, a: Integral, b: Integral):  # Most bitwise operators require realization
        return op(a.__index__(), b.__index__())  # type: ignore
    setup_binop(_, {ops.and_, ops.or_, ops.xor})
    def _(op: BinFn, a: Integral, b: Integral):  # Floor division over ints requires realization, at present
        return op(a.__index__(), b.__index__())  # type: ignore
    setup_binop(_, {ops.truediv})
    def _(a: SmtInt, b: Number):  # Division over ints must produce float
        return (a.__float__(), b)
    setup_promotion(_, {ops.truediv})

    # bool
    def _(op: BinFn, a: SmtBool, b: SmtBool):
        return SmtBool(a.statespace, bool, apply_smt(op, a.var, b.var, a.statespace))
    setup_binop(_, {ops.eq, ops.ne})
    def _(op: BinFn, a: SmtBool, b: bool):
        return SmtInt(a.statespace, int, apply_smt(op, a.var, z3.BoolVal(b), a.statespace))
    setup_binop(_, _ARITHMETIC_OPS)
    def _(op: BinFn, a: bool, b: SmtBool):
        return SmtInt(b.statespace, int, apply_smt(op, z3.BoolVal(a), b.var, b.statespace))
    setup_binop(_, _ARITHMETIC_OPS)


#
#  END new numbers
#


class SmtNumberAble(SmtBackedValue, Real):
    def __pos__(self):
        return self

    def __neg__(self):
        return self._unary_op(ops.neg)

    def __abs__(self):
        return self._unary_op(lambda v: z3.If(v < 0, -v, v))

    def __lt__(self, other):
        return numeric_binop(ops.lt, self, other)

    def __gt__(self, other):
        return numeric_binop(ops.gt, self, other)

    def __le__(self, other):
        return numeric_binop(ops.le, self, other)

    def __ge__(self, other):
        return numeric_binop(ops.ge, self, other)

    def __eq__(self, other):
        return numeric_binop(ops.eq, self, other)

    def __add__(self, other):
        return numeric_binop(ops.add, self, other)
    def __radd__(self, other):
        return numeric_binop(ops.add, other, self)

    def __sub__(self, other):
        return numeric_binop(ops.sub, self, other)
    def __rsub__(self, other):
        return numeric_binop(ops.sub, other, self)

    def __mul__(self, other):
        return numeric_binop(ops.mul, self, other)
    def __rmul__(self, other):
        return numeric_binop(ops.mul, other, self)

    def __pow__(self, other, mod=None):
        if mod is not None:
            return pow(realize(self), pow, mod)
        return numeric_binop(ops.pow, self, other)
    def __rpow__(self, other, mod=None):
        if mod is not None:
            return pow(other, realize(self), mod)
        return numeric_binop(ops.pow, other, self)

    def __lshift__(self, other):
        return numeric_binop(ops.lshift, self, other)
    def __rlshift__(self, other):
        return numeric_binop(ops.lshift, other, self)

    def __rshift__(self, other):
        return numeric_binop(ops.rshift, self, other)
    def __rrshift__(self, other):
        return numeric_binop(ops.rshift, other, self)


    def __and__(self, other):
        return numeric_binop(ops.and_, self, other)
    def __rand__(self, other):
        return numeric_binop(ops.and_, other, self)

    def __or__(self, other):
        return numeric_binop(ops.or_, self, other)
    def __ror__(self, other):
        return numeric_binop(ops.or_, other, self)

    def __xor__(self, other):
        return numeric_binop(ops.xor, self, other)
    def __rxor__(self, other):
        return numeric_binop(ops.xor, other, self)

    def __rtruediv__(self, other):
        return numeric_binop(ops.truediv, other, self)

    def __floordiv__(self, other):
        return numeric_binop(ops.floordiv, self, other)
    def __rfloordiv__(self, other):
        return numeric_binop(ops.floordiv, other, self)

    def __mod__(self, other):
        return numeric_binop(ops.mod, self, other)
    def __rmod__(self, other):
        return numeric_binop(ops.mod, other, self)

    def __divmod__(self, other):
        return (self // other, self % other)
    def __rdivmod__(self, other):
        return (other // self, other % self)


class SmtIntable(SmtNumberAble, Integral):
    # bitwise operators
    def __invert__(self):
        return -(self + 1)

    def __floor__(self):
        return self
    def __ceil__(self):
        return self
    def __trunc__(self):
        return self

    def __mul__(self, other):
        if isinstance(other, str):
            # Create a symbolic string that regex-matches as a repetition.
            space = self.statespace
            count = self.var # z3.If(self.var >= 0, self.var, 0))
            result = SmtStr(space, str, f'{self.var}_str{space.uniq()}')
            space.add(z3.InRe(result.var, z3.Star(z3.Re(other))))
            space.add(z3.Length(result.var) == len(other) * count)
            return result
        return numeric_binop(ops.mul, self, other)
    __rmul__ = __mul__


class SmtBool(SmtIntable):
    def __init__(self, statespace: StateSpace, typ: Type, smtvar: object):
        assert typ == bool
        SmtBackedValue.__init__(self, statespace, typ, smtvar)

    def __neg__(self):
        return SmtInt(self.statespace, int, -smt_bool_to_int(self.var))

    def __repr__(self):
        return self.__bool__().__repr__()

    def __hash__(self):
        return self.__bool__().__hash__()

    def __index__(self):
        return SmtInt(self.statespace, int, smt_bool_to_int(self.var))

    def __bool__(self):
        return self.statespace.choose_possible(self.var)

    def __int__(self):
        return SmtInt(self.statespace, int, smt_bool_to_int(self.var))

    def __float__(self):
        return SmtFloat(self.statespace, float, smt_bool_to_float(self.var))

    def __complex__(self):
        return complex(self.__float__())

    def __round__(self, ndigits=None):
        return round(int(self), ndigits)


class SmtInt(SmtIntable):
    def __init__(self, statespace: StateSpace, typ: Type, smtvar: Union[str, z3.ArithRef]):
        assert typ == int
        assert type(smtvar) != int
        SmtIntable.__init__(self, statespace, typ, smtvar)

    def __repr__(self):
        return self.__index__().__repr__()
        # TODO: do a symbolic conversion!:
        #return SmtStr(self.statespace, str, z3.IntToStr(self.var))

    def __hash__(self):
        return self.__index__().__hash__()

    def __float__(self):
        return SmtFloat(self.statespace, float, smt_int_to_float(self.var))

    def __complex__(self):
        return complex(self.__float__())

    def __index__(self):
        #debug('WARNING: attempting to materialize symbolic integer. Trace:')
        # traceback.print_stack()
        if self == 0:
            return 0
        ret = self.statespace.find_model_value(self.var)
        assert type(ret) is int, f'SmtInt with wrong SMT var type ({type(ret)})'
        return ret

    def __bool__(self):
        return SmtBool(self.statespace, bool, self.var != 0).__bool__()

    def __int__(self):
        return self.__index__()

    def __round__(self, ndigits=None):
        if ndigits is None or ndigits >= 0:
            return self # TODO: test
        return round(self.__index__(), ndigits) # TODO: could do this symbolically


_Z3_ONE_HALF = z3.RealVal("1/2")


class SmtFloat(SmtNumberAble):
    def __init__(self, statespace: StateSpace, typ: Type, smtvar: object):
        assert typ == float, f'SmtFloat created with unexpected python type ({type(typ)})'
        SmtBackedValue.__init__(self, statespace, typ, smtvar)

    def __repr__(self):
        return self.statespace.find_model_value(self.var).__repr__()

    def __hash__(self):
        return self.statespace.find_model_value(self.var).__hash__()

    def __bool__(self):
        return SmtBool(self.statespace, bool, self.var != 0).__bool__()
    
    def __float__(self):
        return self.statespace.find_model_value(self.var).__float__()

    def __complex__(self):
        return complex(self.__float__())

    def __round__(self, ndigits=None):
        if ndigits is not None:
            factor = 10 ** realize(ndigits)  # realize to avoid exponentation-to-variable
            return round(self * factor) / factor
        else:
            var, floor, nearest = self.var, z3.ToInt(
                self.var), z3.ToInt(self.var + _Z3_ONE_HALF)
            return SmtInt(self.statespace, int, z3.If(var != floor + _Z3_ONE_HALF, nearest, z3.If(floor % 2 == 0, floor, floor + 1)))

    def __floor__(self):
        return SmtInt(self.statespace, int, z3.ToInt(self.var))

    def __ceil__(self):
        var, floor = self.var, z3.ToInt(self.var)
        return SmtInt(self.statespace, int, z3.If(var == floor, floor, floor + 1))

    def __mod__(self, other):
        return realize(self) % realize(other) # TODO: z3 does not support modulo on reals

    def __trunc__(self):
        var, floor = self.var, z3.ToInt(self.var)
        return SmtInt(self.statespace, int, z3.If(var >= 0, floor, floor + 1))


class SmtDictOrSet(SmtBackedValue):
    '''
    TODO: Ordering is a challenging issue here.
    Modern pythons have in-order iteration for dictionaries but not sets.
    '''
    def __init__(self, statespace: StateSpace, typ: Type, smtvar: object):
        self.key_pytype = normalize_pytype(type_arg_of(typ, 0))
        self.smt_key_sort = type_to_smt_sort(self.key_pytype)
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
    def __init__(self, space: StateSpace, typ: Type, smtvar: object):
        self.val_pytype = normalize_pytype(type_arg_of(typ, 1))
        self.smt_val_sort = type_to_smt_sort(self.val_pytype)
        SmtDictOrSet.__init__(self, space, typ, smtvar)
        self.val_ch_type = crosshair_type_for_python_type(self.val_pytype)
        arr_var = self._arr()
        len_var = self._len()
        self.val_missing_checker = arr_var.sort().range().recognizer(0)
        self.val_missing_constructor = arr_var.sort().range().constructor(0)
        self.val_constructor = arr_var.sort().range().constructor(1)
        self.val_accessor = arr_var.sort().range().accessor(1, 0)
        self.empty = z3.K(arr_var.sort().domain(),
                          self.val_missing_constructor())
        self._iter_cache: List[z3.Const] = []
        space.add((arr_var == self.empty) == (len_var == 0))
        def list_can_be_iterated():
            list(self)
            return True
        space.defer_assumption('dict iteration is consistent with items', list_can_be_iterated)

    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        arr_smt_sort = z3.ArraySort(
            self.smt_key_sort, possibly_missing_sort(self.smt_val_sort))
        return (
            z3.Const(varname + '_map' + self.statespace.uniq(), arr_smt_sort),
            z3.Const(varname + '_len' + self.statespace.uniq(), z3.IntSort())
        )

    def __eq__(self, other):
        (self_arr, self_len) = self.var
        has_heapref = smt_sort_has_heapref(
            self.var[1].sort()) or smt_sort_has_heapref(self.var[0].sort())
        if not has_heapref:
            if isinstance(other, SmtDict):
                (other_arr, other_len) = other.var
                return SmtBool(self.statespace, bool, z3.And(self_len == other_len, self_arr == other_arr))
        # Manually check equality. Drive the loop from the (likely) concrete value 'other':
        if not isinstance(other, collections.abc.Mapping):
            return False
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
        k = coerce_to_smt_sort(self.statespace, k, self.smt_key_sort)
        v = coerce_to_smt_sort(self.statespace, v, self.smt_val_sort)
        if k is None or v is None:
            # TODO: dictionaries can become more losely typed as items are
            # assigned. Dictionary is invariant, though, so we expect such cases
            # to have been already caught by the type checker.
            raise CrosshairUnsupported('dictionary assignment with conflicting types')
        old_arr, old_len = self.var
        new_len = z3.If(z3.Select(old_arr, k) == missing, old_len + 1, old_len)
        self.var = (z3.Store(old_arr, k, self.val_constructor(v)), new_len)

    def __delitem__(self, pykey):
        missing = self.val_missing_constructor()
        k = force_to_smt_sort(self.statespace, pykey, self.smt_key_sort)
        old_arr, old_len = self.var
        if SmtBool(self.statespace, bool, z3.Select(old_arr, k) == missing).__bool__():
            raise KeyError(pykey)
        if SmtBool(self.statespace, bool, self._len() == 0).__bool__():
            raise IgnoreAttempt('SmtDict in inconsistent state')
        self.var = (z3.Store(old_arr, k, missing), old_len - 1)

    def __getitem__(self, k):
        with self.statespace.framework():
            smt_key = coerce_to_smt_sort(self.statespace, k, self.smt_key_sort)
            if smt_key is None:
                # A key of the wrong type cannot be present.
                # Try to raise the right exception:
                if getattr(k, '__hash__', None) is None:
                    raise TypeError("unhashable type")
                else:
                    raise KeyError(k)
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
        iter_cache = self._iter_cache
        space = self.statespace
        idx = 0
        arr_sort = self._arr().sort()
        is_missing = self.val_missing_checker
        while SmtBool(space, bool, idx < len_var).__bool__():
            if not space.choose_possible(arr_var != self.empty, favor_true=True):
                raise IgnoreAttempt('SmtDict in inconsistent state')
            k = z3.Const('k' + str(idx) + space.uniq(),
                         arr_sort.domain())
            v = z3.Const('v' + str(idx) + space.uniq(),
                         self.val_constructor.domain(0))
            remaining = z3.Const('remaining' + str(idx) +
                                 space.uniq(), arr_sort)
            space.add(arr_var == z3.Store(
                remaining, k, self.val_constructor(v)))
            space.add(is_missing(z3.Select(remaining, k)))

            # our iter_cache might contain old keys that were removed;
            # check to make sure the current key is still present:
            while idx < len(iter_cache):
                still_present = z3.Not(is_missing(z3.Select(arr_var, iter_cache[idx])))
                if space.choose_possible(still_present, favor_true=True):
                    break
                del iter_cache[idx]
            if idx > len(iter_cache):
                raise CrosshairInternal()
            if idx == len(iter_cache):
                iter_cache.append(k)
            else:
                space.add(k == iter_cache[idx])
            idx += 1
            yield smt_to_ch_value(space, self.snapshot, k, self.key_pytype)
            arr_var = remaining
        # In this conditional, we reconcile the parallel symbolic variables for length
        # and contents:
        if not space.choose_possible(arr_var == self.empty, favor_true=True):
            raise IgnoreAttempt('SmtDict in inconsistent state')

    def copy(self):
        return SmtDict(self.statespace, self.python_type, self.var)

    # TODO: investigate this approach for type masquerading:
    # @property
    # def __class__(self):
    #    return dict


class SmtSet(SmtDictOrSet, collections.abc.Set):
    def __init__(self, statespace: StateSpace, typ: Type, smtvar: object):
        SmtDictOrSet.__init__(self, statespace, typ, smtvar)
        self.empty = z3.K(self._arr().sort().domain(), False)
        self.statespace.add((self._arr() == self.empty) == (self._len() == 0))

    def __eq__(self, other):
        (self_arr, self_len) = self.var
        if isinstance(other, SmtSet):
            (other_arr, other_len) = other.var
            if other_arr.sort() == self_arr.sort():
                return SmtBool(self.statespace, bool, z3.And(self_len == other_len, self_arr == other_arr))
        if not isinstance(other, (set, frozenset, SmtSet)):
            return False
        # Manually check equality. Drive size from the (likely) concrete value 'other':
        if len(self) != len(other):
            return False
        # Then iterate on self (iteration will create a lot of good symbolic constraints):
        for item in self:
            # We iterate over other instead of just checking "if item in other:" because we
            # don't want to hash our symbolic item, which would materialize it.
            found = False
            for oitem in other:
                if item == oitem:
                    found = True
                    break
            if not found:
                return False
        return True

    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        return (
            z3.Const(varname + '_map' + self.statespace.uniq(),
                     z3.ArraySort(type_to_smt_sort(self.key_pytype),
                                  z3.BoolSort())),
            z3.Const(varname + '_len' + self.statespace.uniq(), z3.IntSort())
        )

    def __contains__(self, key):
        if getattr(key, '__hash__', None) is None:
            raise TypeError("unhashable type")
        k = coerce_to_smt_sort(self.statespace, key, self._arr().sort().domain())
        if k is not None:
            present = self._arr()[k]
            return SmtBool(self.statespace, bool, present)
        # Fall back to standard equality and iteration
        for self_item in self:
            if self_item == key:
                return True
        return False

    def __iter__(self):
        arr_var, len_var = self.var
        idx = 0
        arr_sort = self._arr().sort()
        keys_on_heap = smt_sort_has_heapref(arr_sort.domain())
        already_yielded = []
        while SmtBool(self.statespace, bool, idx < len_var).__bool__():
            if SmtBool(self.statespace, bool, arr_var == self.empty).__bool__():
                raise IgnoreAttempt('SmtSet in inconsistent state')
            k = z3.Const('k' + str(idx) + self.statespace.uniq(),
                         arr_sort.domain())
            remaining = z3.Const('remaining' + str(idx) +
                                 self.statespace.uniq(), arr_sort)
            idx += 1
            self.statespace.add(arr_var == z3.Store(remaining, k, True))
            self.statespace.add(z3.Not(z3.Select(remaining, k)))
            ch_value = smt_to_ch_value(self.statespace, self.snapshot, k, self.key_pytype)
            if keys_on_heap:
                # need to confirm that we do not yield two keys that are __eq__
                for previous_value in already_yielded:
                    if ch_value == previous_value:
                        raise IgnoreAttempt('Duplicate items in set')
                already_yielded.append(ch_value)
            yield ch_value
            arr_var = remaining
        # In this conditional, we reconcile the parallel symbolic variables for length
        # and contents:
        if SmtBool(self.statespace, bool, arr_var != self.empty).__bool__():
            raise IgnoreAttempt('SmtSet in inconsistent state')

    def _set_op(self, attr, other):
        # We need to check the type of other here, because builtin sets
        # do not accept iterable args (but the abc Set does)
        if isinstance(other, collections.abc.Set):
            return getattr(collections.abc.Set, attr)(self, other)
        else:
            raise TypeError

    # Hardwire some operations into abc methods
    # (SmtBackedValue defaults these operations into
    # TypeErrors, but must appear first in the mro)
    def __ge__(self, other):
        return self._set_op('__ge__', other)

    def __gt__(self, other):
        return self._set_op('__gt__', other)

    def __le__(self, other):
        return self._set_op('__le__', other)

    def __lt__(self, other):
        return self._set_op('__lt__', other)

    def __and__(self, other):
        return self._set_op('__and__', other)
    __rand__ = __and__

    def __or__(self, other):
        return self._set_op('__or__', other)
    __ror__ = __or__

    def __xor__(self, other):
        return self._set_op('__xor__', other)
    __rxor__ = __xor__

    def __sub__(self, other):
        return self._set_op('__sub__', other)


class SmtMutableSet(SmtSet):
    def __repr__(self):
        return str(set(self))

    @classmethod
    def _from_iterable(cls, it):
        # overrides collections.abc.Set's version
        return set(it)

    def add(self, k):
        k = coerce_to_smt_var(self.statespace, k)
        old_arr, old_len = self.var
        new_len = z3.If(z3.Select(old_arr, k), old_len, old_len + 1)
        self.var = (z3.Store(old_arr, k, True), new_len)

    def discard(self, k):
        k = coerce_to_smt_var(self.statespace, k)
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
        return frozenset(it)


def process_slice_vs_symbolic_len(
        space: StateSpace,
        i: slice,
        smt_len: z3.ExprRef
) -> Union[z3.ExprRef, Tuple[z3.ExprRef, z3.ExprRef]]:
    def normalize_symbolic_index(idx):
        if isinstance(idx, int):
            return idx if idx >= 0 else smt_len + idx
        else:
            idx = force_to_smt_sort(space, idx, z3.IntSort())
            return z3.If(idx >= 0, idx, smt_len + idx)
    if isinstance(i, int) or isinstance(i, SmtInt):
        smt_i = smt_coerce(i)
        if space.smt_fork(z3.Or(smt_i >= smt_len, smt_i < -smt_len)):
            raise IndexError(f'index "{i}" is out of range')
        smt_i = normalize_symbolic_index(smt_i)
        return force_to_smt_sort(space, smt_i, z3.IntSort())
    elif isinstance(i, slice):
        smt_start, smt_stop, smt_step = (i.start, i.stop, i.step)
        if smt_step not in (None, 1):
            raise CrosshairUnsupported('slice steps not handled')
        start = normalize_symbolic_index(
            smt_start) if i.start is not None else 0
        stop = normalize_symbolic_index(
            smt_stop) if i.stop is not None else smt_len
        return (force_to_smt_sort(space, start, z3.IntSort()),
                force_to_smt_sort(space, stop, z3.IntSort()))
    else:
        raise TypeError(
            'indices must be integers or slices, not ' + str(type(i)))


class SmtSequence(SmtBackedValue, collections.abc.Sequence):
    def __iter__(self):
        idx = 0
        while len(self) > idx:
            yield self[idx]
            idx += 1

    def __len__(self):
        return SmtInt(self.statespace, int, z3.Length(self.var))

    def __bool__(self):
        return SmtBool(self.statespace, bool, z3.Length(self.var) > 0).__bool__()

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("can't multiply by non-int")
        if other <= 0:
            return self[0:0]
        ret = self
        for idx in range(1, other):
            ret = self.__add__(ret)
        return ret

    def __rmul__(self, other):
        return self.__mul__(other)


class SmtArrayBasedUniformTuple(SmtSequence):
    def __init__(self, statespace: StateSpace, typ: Type, smtvar: Union[str, Tuple]):
        if type(smtvar) == str:
            pass
        else:
            assert type(smtvar) is tuple, f'incorrect type {type(smtvar)}'
            assert len(smtvar) == 2
        self.val_pytype = normalize_pytype(type_arg_of(typ, 0))
        self.item_smt_sort = (HeapRef if pytype_uses_heap(self.val_pytype)
                              else type_to_smt_sort(self.val_pytype))
        self.key_pytype = int
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        arr_var = self._arr()
        len_var = self._len()
        self.statespace.add(len_var >= 0)
        
        self.val_ch_type = crosshair_type_for_python_type(self.val_pytype)
        

    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        arr_smt_type = z3.ArraySort(z3.IntSort(), self.item_smt_sort)
        return (
            z3.Const(varname + '_map' + self.statespace.uniq(), arr_smt_type),
            z3.Const(varname + '_len' + self.statespace.uniq(), z3.IntSort())
        )

    def _arr(self):
        return self.var[0]

    def _len(self):
        return self.var[1]

    def __len__(self):
        return SmtInt(self.statespace, int, self._len())

    def __bool__(self):
        return SmtBool(self.statespace, bool, self._len() != 0).__bool__()
    
    def __eq__(self, other):
        if self is other:
            return True
        (self_arr, self_len) = self.var
        if not is_iterable(other):
            return False
        if len(self) != len(other):
            return False
        for idx, v in enumerate(other):
            if self[idx] is v:
                continue
            if self[idx] != v:
                return False
        return True

    def __repr__(self):
        return str(list(self))

    def __setitem__(self, k, v):
        raise CrosshairInternal()

    def __delitem__(self, k):
        raise CrosshairInternal()

    def __iter__(self):
        arr_var, len_var = self.var
        idx = 0
        while SmtBool(self.statespace, bool, idx < len_var).__bool__():
            yield smt_to_ch_value(self.statespace,
                                  self.snapshot,
                                  z3.Select(arr_var, idx),
                                  self.val_pytype)
            idx += 1

    def __add__(self, other: object):
        if isinstance(other, collections.abc.Sequence):
            return SequenceConcatenation(self, other)
        return NotImplemented

    def __radd__(self, other: object):
        if isinstance(other, collections.abc.Sequence):
            return SequenceConcatenation(other, self)
        return NotImplemented

    def __contains__(self, other):
        space = self.statespace
        with space.framework():
            if not smt_sort_has_heapref(self.item_smt_sort):
                smt_other = coerce_to_smt_sort(space, other, self.item_smt_sort)
                if smt_other is not None:
                    # OK to perform a symbolic comparison
                    idx = z3.Const('possible_idx' + space.uniq(), z3.IntSort())
                    idx_in_range = z3.Exists(idx, z3.And(0 <= idx,
                                                         idx < self._len(),
                                                         z3.Select(self._arr(), idx) == smt_other))
                    return SmtBool(space, bool, idx_in_range)
            # Fall back to standard equality and iteration
            for self_item in self:
                if self_item == other:
                    return True
            return False

    def __getitem__(self, i):
        space = self.statespace
        with space.framework():
            if i == slice(None, None, None):
                return self
            idx_or_pair = process_slice_vs_symbolic_len(space, i, self._len())
            if isinstance(idx_or_pair, tuple):
                (start, stop) = idx_or_pair
                (myarr, mylen) = self.var
                start = SmtInt(space, int, start)
                stop = SmtInt(space, int, smt_min(mylen, smt_coerce(stop)))
                return SliceView(self, start, stop)
            else:
                smt_result = z3.Select(self._arr(), idx_or_pair)
                return smt_to_ch_value(space, self.snapshot, smt_result, self.val_pytype)

    def insert(self, idx, obj):
        raise CrosshairUnsupported


class SmtList(ShellMutableSequence, collections.abc.MutableSequence, CrossHairValue):
    def __init__(self, *a):
        ShellMutableSequence.__init__(self, SmtArrayBasedUniformTuple(*a))
    def __ch_pytype__(self):
        return python_type(self.inner)
    def __ch_realize__(self):
        return list(self)
    def _is_subclass_of_(cls, other):
        return other is list
    def __mod__(self, *a):
        raise TypeError
    def index(self, value: object, start: int = 0, stop: int = 9223372036854775807) -> int:
        '''
        Return first index of value.
        Raises ValueError if the value is not present.
        '''
        try:
            start, stop = start.__index__(), stop.__index__()
        except AttributeError:
            # Re-create the error that list.index would give on bad start/stop values:
            raise TypeError('slice indices must be integers or have an __index__ method')
        self_len = self.__len__()
        if self_len < stop:
            stop = self_len
        i = start
        while i < self_len and i < stop:
            cur = self[i]
            if cur == value:
                return cur
            i += 1
        raise ValueError(f'{value} is not in list')


class SmtType(SmtBackedValue):
    _realization : Optional[Type] = None
    def __init__(self, statespace: StateSpace, typ: Type, smtvar: object):
        assert origin_of(typ) is type
        self.pytype_cap = origin_of(typ.__args__[0]) if hasattr(typ, '__args__') else object
        assert type(self.pytype_cap) is type
        smt_cap = statespace.type_repo.get_type(self.pytype_cap)
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        statespace.add(statespace.type_repo.smt_issubclass(self.var, smt_cap))
    def _is_superclass_of_(self, other):
        if self is SmtType:
            return False
        if type(other) is SmtType:
            # Prefer it this way because only _is_subcless_of_ does the type cap lowering.
            return other._is_subclass_of_(self)
        space = self.statespace
        with space.framework():
            coerced = coerce_to_smt_sort(space, other, self.var.sort())
            if coerced is None:
                return False
            return SmtBool(space, bool, space.type_repo.smt_issubclass(coerced, self.var))
    def _is_subclass_of_(self, other):
        if self is SmtType:
            return False
        space = self.statespace
        with space.framework():
            coerced = coerce_to_smt_sort(space, other, self.var.sort())
            if coerced is None:
                return False
            ret = SmtBool(space, bool, space.type_repo.smt_issubclass(self.var, coerced))
            other_pytype = other.pytype_cap if type(other) is SmtType else other
            # consider lowering the type cap
            if other_pytype is not self.pytype_cap and issubclass(other_pytype, self.pytype_cap) and ret:
                self.pytype_cap = other_pytype
            return ret
    def __ch_realize__(self):
        return self._realized()
    def _realized(self):
        if self._realization is None:
            self._realization = self._realize()
        return self._realization
    def _realize(self) -> Type:
        cap = self.pytype_cap
        space = self.statespace
        if cap is object:
            pytype_to_smt = space.type_repo.pytype_to_smt
            for pytype, smt_type in pytype_to_smt.items():
                if not issubclass(pytype, cap):
                    continue
                if space.smt_fork(self.var == smt_type):
                    return pytype
            raise CrosshairUnsupported('Will not exhaustively attempt `object` types')
        else:
            subtype = choose_type(space, cap)
            smt_type = space.type_repo.get_type(subtype)
            if space.smt_fork(self.var != smt_type):
                raise IgnoreAttempt
            return subtype
    def __bool__(self):
        return True
    def __copy__(self):
        return self if self._realization is None else self._realization
    def __repr__(self):
        return repr(self._realized())
    def __hash__(self):
        return hash(self._realized())


class LazyObject(ObjectProxy):
    _inner: object = _MISSING

    def _realize(self):
        raise NotImplementedError

    def _wrapped(self):
        inner = object.__getattribute__(self, '_inner')
        if inner is _MISSING:
            inner = self._realize()
            object.__setattr__(self, '_inner', inner)
        return inner

    def __deepcopy__(self, memo):
        inner = object.__getattribute__(self, '_inner')
        if inner is _MISSING:
            # CrossHair will deepcopy for mutation checking.
            # That's usually bad for LazyObjects, which want to defer their
            # realization, so we simply don't do mutation checking for these
            # kinds of values right now.
            return self
        else:
            return copy.deepcopy(self.wrapped())


class SmtObject(LazyObject, CrossHairValue):
    '''
    An object with an unknown type.
    We lazily create a more specific smt-based value in hopes that an
    isinstance() check will be called before something is accessed on us.
    Note that this class is not an SmtBackedValue, but its _typ and _inner
    members can be.
    '''
    def __init__(self, space: StateSpace, typ: Type, varname: object):
        object.__setattr__(self, '_typ', SmtType(space, type, varname))
        object.__setattr__(self, '_space', space)
        object.__setattr__(self, '_varname', varname)

    def _realize(self):
        space = object.__getattribute__(self, '_space')
        varname = object.__getattribute__(self, '_varname')

        typ = object.__getattribute__(self, '_typ')
        pytype = realize(typ)
        debug('materializing symbolic object as an instance of', pytype)
        if pytype is object:
            return object()
        return proxy_for_type(pytype, space, varname, allow_subtypes=False)

    @property
    def python_type(self):
        return object.__getattribute__(self, '_typ')

    @property
    def __class__(self):
        return SmtObject

    @__class__.setter
    def __class__(self, value):
        raise CrosshairUnsupported


class SmtCallable(SmtBackedValue):
    __closure__ = None

    def __init___(self, statespace: StateSpace, typ: Type, smtvar: object):
        SmtBackedValue.__init__(self, statespace, typ, smtvar)

    def __bool__(self):
        return True

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
        self.arg_ch_type = map(
            crosshair_type_for_python_type, self.arg_pytypes)
        self.ret_ch_type = crosshair_type_for_python_type(self.ret_pytype)
        all_pytypes = tuple(self.arg_pytypes) + (self.ret_pytype,)
        return z3.Function(varname + self.statespace.uniq(),
                           *map(type_to_smt_sort, self.arg_pytypes),
                           type_to_smt_sort(self.ret_pytype))

    def __ch_realize__(self):
        return self  # we don't realize callables right now

    def __call__(self, *args):
        if len(args) != len(self.arg_pytypes):
            raise TypeError('wrong number of arguments')
        args = (coerce_to_smt_var(self.statespace, a) for a in args)
        smt_ret = self.var(*args)
        # TODO: detect that `smt_ret` might be a HeapRef here
        return self.ret_ch_type(self.statespace, self.ret_pytype, smt_ret)

    def __repr__(self):
        finterp = self.statespace.find_model_value_for_function(self.var)
        if finterp is None:
            # (z3 model completion will not interpret a function for me currently)
            return 'lambda *a: None'
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


class SmtUniformTuple(SmtArrayBasedUniformTuple, collections.abc.Sequence, collections.abc.Hashable):
    def __repr__(self):
        return tuple(self).__repr__()

    def __hash__(self):
        return tuple(self).__hash__()


class SmtStr(SmtSequence, AbcString):
    def __init__(self, statespace: StateSpace, typ: Type, smtvar: object):
        assert typ == str
        SmtBackedValue.__init__(self, statespace, typ, smtvar)
        self.item_pytype = str
        self.item_ch_type = SmtStr

    def __str__(self):
        return self.statespace.find_model_value(self.var)

    def __copy__(self):
        return SmtStr(self.statespace, str, self.var)

    def __repr__(self):
        return repr(self.__str__())

    def __hash__(self):
        return hash(self.__str__())

    def __add__(self, other):
        return self._binary_op(other, ops.add)

    def __radd__(self, other):
        return self._binary_op(other, lambda a, b: b + a)

    def __mul__(self, other):
        space = self.statespace
        # If repetition count is a literal, use that first:
        if isinstance(other, Integral):
            if other <= 1:
                return self if other == 1 else ''
            # Note that in SmtInt, we attempt string multiplication via regex.
            # Z3 cannot do much with a symbolic regex, so we case-split on
            # the repetition count.
            return SmtStr(space, str, z3.Concat(*[self.var for _ in range(other)]))
        return NotImplemented
    __rmul__ = __mul__

    def __mod__(self, other):
        return self.__str__() % realize(other)

    def _cmp_op(self, other, op):
        forced = force_to_smt_sort(self.statespace, other, self.var.sort())
        return SmtBool(self.statespace, bool, op(self.var, forced))

    def __lt__(self, other):
        return self._cmp_op(other, ops.lt)

    def __le__(self, other):
        return self._cmp_op(other, ops.le)

    def __gt__(self, other):
        return self._cmp_op(other, ops.gt)

    def __ge__(self, other):
        return self._cmp_op(other, ops.ge)

    def __contains__(self, other):
        forced = force_to_smt_sort(self.statespace, other, self.var.sort())
        return SmtBool(self.statespace, bool, z3.Contains(self.var, forced))

    def __getitem__(self, i):
        idx_or_pair = process_slice_vs_symbolic_len(
            self.statespace, i, z3.Length(self.var))
        if isinstance(idx_or_pair, tuple):
            (start, stop) = idx_or_pair
            smt_result = z3.Extract(self.var, start, stop - start)
        else:
            smt_result = z3.Extract(self.var, idx_or_pair, 1)
        return SmtStr(self.statespace, str, smt_result)

    def find(self, substr, start=None, end=None):
        if end is None:
            return SmtInt(self.statespace, int,
                          z3.IndexOf(self.var, smt_coerce(substr), start or 0))
        else:
            return self.__getitem__(slice(start, end, 1)).index(s)


_CACHED_TYPE_ENUMS: Dict[FrozenSet[type], z3.SortRef] = {}


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
    type: SmtType,
}

# Type ignore pending https://github.com/python/mypy/issues/6864
_PYTYPE_TO_WRAPPER_TYPE[collections.abc.Callable] = SmtCallable  # type:ignore

_WRAPPER_TYPE_TO_PYTYPE = dict((v, k)
                               for (k, v) in _PYTYPE_TO_WRAPPER_TYPE.items())


#
# Proxy making helpers
#

def make_union_choice(creator, *pytypes):
    for typ in pytypes[:-1]:
        if creator.space.smt_fork():
            return creator(typ)
    return creator(pytypes[-1])

def make_optional_smt(smt_type):
    def make(creator, *type_args):
        ret = smt_type(creator.space, creator.pytype, creator.varname)
        if creator.space.fork_parallel(false_probability=0.98):
            debug('Prematurely realizing', creator.pytype, 'value')
            ret = realize(ret)
        return ret
    return make

def make_dictionary(creator, key_type = Any, value_type = Any):
    space, varname = creator.space, creator.varname
    if smt_sort_has_heapref(type_to_smt_sort(key_type)):
        kv = proxy_for_type(List[Tuple[key_type, value_type]], # type: ignore
                            space, varname + 'items', allow_subtypes=False)
        orig_kv = kv[:]
        def ensure_keys_are_unique() -> bool:
            return len(set(k for k, _ in orig_kv)) == len(orig_kv)
        space.defer_assumption('dict keys are unique', ensure_keys_are_unique)
        return SimpleDict(kv)
    return SmtDict(space, creator.pytype, varname)

def make_tuple(creator, *type_args):
    if not type_args:
        type_args = (object, ...)  # type: ignore
    if len(type_args) == 2 and type_args[1] == ...:
        return SmtUniformTuple(creator.space, creator.pytype, creator.varname)
    else:
        return tuple(proxy_for_type(t, creator.space, creator.varname + '_at_' + str(idx), allow_subtypes=True)
                     for (idx, t) in enumerate(type_args))

def make_raiser(exc, *a) -> Callable:
    def do_raise(*ra, **rkw) -> NoReturn:
        raise exc(*a)
    return do_raise

#
# Monkey Patches
#

_T = TypeVar('_T')
_VT = TypeVar('_VT')

class _BuiltinsCopy:
    pass

_TRUE_BUILTINS: Any = _BuiltinsCopy()
_TRUE_BUILTINS.__dict__.update(orig_builtins.__dict__)


# CPython's len() forces the return value to be a native integer.
# Avoid that requirement by making it only call __len__().
def _len(l):
    return l.__len__() if hasattr(l, '__len__') else [x for x in l].__len__()

# Avoid calling __len__().__index__() on the input list.


def _sorted(l, **kw):
    ret = list(l.__iter__())
    ret.sort()
    return ret

# Trick the system into believing that symbolic values are
# native types.

def _issubclass(subclass, superclasses):
    subclass_is_special = hasattr(subclass, '_is_subclass_of_')
    if not subclass_is_special:
        # We could also check superclass(es) for a special method, but
        # the native function won't return True in those cases anyway.
        try:
            ret = _TRUE_BUILTINS.issubclass(subclass, superclasses)
            if ret:
                return True
        except TypeError:
            pass
    if type(superclasses) is not tuple:
        superclasses = (superclasses,)
    for superclass in superclasses:
        if hasattr(superclass, '_is_superclass_of_'):
            method = superclass._is_superclass_of_
            if method(subclass) if hasattr(method, '__self__') else method(subclass, superclass):
                return True
        if subclass_is_special:
            method = subclass._is_subclass_of_
            if method(superclass) if hasattr(method, '__self__') else method(subclass, superclass):
                return True
    return False

def _isinstance(obj, types):
    try:
        ret = _TRUE_BUILTINS.isinstance(obj, types)
        if ret:
            return True
    except TypeError:
        pass
    if hasattr(obj, 'python_type'):
        obj_type = obj.python_type
        if hasattr(obj_type, '__origin__'):
            obj_type = obj_type.__origin__
    else:
        obj_type = type(obj)
    return issubclass(obj_type, types)

#    # TODO: consider tricking the system into believing that symbolic values are
#    # native types.
#    def patched_type(self, *args):
#        ret = self.originals['type'](*args)
#        if len(args) == 1:
#            ret = _WRAPPER_TYPE_TO_PYTYPE.get(ret, ret)
#        for (original_type, proxied_type) in ProxiedObject.__dict__["_class_proxy_cache"].items():
#            if ret is proxied_type:
#                return original_type
#        return ret


def _hash(obj: Hashable) -> int:
    '''
    post[]: -2**63 <= _ < 2**63
    '''
    # Skip the built-in hash if possible, because it requires the output
    # to be a native int:
    if is_hashable(obj):
        # You might think we'd say "return obj.__hash__()" here, but we need some
        # special gymnastics to avoid "metaclass confusion".
        # See: https://docs.python.org/3/reference/datamodel.html#special-method-lookup
        return type(obj).__hash__(obj)
    else:
        return _TRUE_BUILTINS.hash(obj)

#def sum(i: Iterable[_T]) -> Union[_T, int]:
#    '''
#    post[]: _ == 0 or len(i) > 0
#    '''
#    return _TRUE_BUILTINS.sum(i)

# def print(*a: object, **kw: Any) -> None:
#    '''
#    post: True
#    '''
#    _TRUE_BUILTINS.print(*a, **kw)


def _repr(arg: object) -> str:
    '''
    post[]: True
    '''
    # Skip the built-in repr if possible, because it requires the output
    # to be a native string:
    if hasattr(arg, '__repr__'):
        # You might think we'd say "return obj.__repr__()" here, but we need some
        # special gymnastics to avoid "metaclass confusion".
        # See: https://docs.python.org/3/reference/datamodel.html#special-method-lookup
        return type(arg).__repr__(arg)
    else:
        return _TRUE_BUILTINS.repr(arg)

_orig_list_index = orig_builtins.list.index
def _list_index(self, value, start=0, stop=9223372036854775807):
    return _orig_list_index(self, value, realize(start), realize(stop))

def _list_repr(self):
    # A pure python implementation so that we get the monkey-patched
    # version of repr when appropriate:
    return '[' + ', '.join(repr(x) for x in self) + ']'

@functools.singledispatch
def _max(*values, key=lambda x: x, default=_MISSING):
    return _max_iter(values, key=key, default=default)


@_max.register(collections.Iterable)  # TODO: I think this explodes: max([1,2], [3], key=len)
def _max_iter(values: Iterable[_T], *, key: Callable = lambda x: x, default: Union[_Missing, _VT] = _MISSING) -> _T:
    '''
    pre: bool(values) or default is not _MISSING
    post[]::
      (_ in values) if default is _MISSING else True
      ((_ in values) or (_ is default)) if default is not _MISSING else True
    '''
    kw = {} if default is _MISSING else {'default': default}
    return _TRUE_BUILTINS.max(values, key=key, **kw)


@functools.singledispatch
def _min(*values, key=lambda x: x, default=_MISSING):
    return _min_iter(values, key=key, default=default)


@_min.register(collections.Iterable)
def _min_iter(values: Iterable[_T], *, key: Callable = lambda x: x, default: Union[_Missing, _VT] = _MISSING) -> _T:
    '''
    pre: bool(values) or default is not _MISSING
    post[]::
      (_ in values) if default is _MISSING else True
      ((_ in values) or (_ is default)) if default is not _MISSING else True
    '''
    kw = {} if default is _MISSING else {'default': default}
    return _TRUE_BUILTINS.min(values, key=key, **kw)


#
# Registrations
#

def make_registrations():

    register_type(Union, make_union_choice)

    # Types modeled in the SMT solver:

    register_type(type(None), lambda *a: None)
    register_type(bool, make_optional_smt(SmtBool))
    register_type(int, make_optional_smt(SmtInt))
    register_type(float, make_optional_smt(SmtFloat))
    register_type(str, make_optional_smt(SmtStr))
    register_type(list, make_optional_smt(SmtList))
    register_type(dict, make_dictionary)
    register_type(tuple, make_tuple)
    register_type(set, make_optional_smt(SmtMutableSet))
    register_type(frozenset, make_optional_smt(SmtFrozenSet))
    register_type(type, make_optional_smt(SmtType))
    register_type(collections.abc.Callable, make_optional_smt(SmtCallable))

    # Most types are not directly modeled in the solver, rather they are built
    # on top of the modeled types. Such types are enumerated here:
    
    register_type(object, lambda p: SmtObject(p.space, p.pytype, p.varname))
    register_type(complex, lambda p: complex(p(float), p(float)))
    register_type(slice, lambda p: slice(p(Optional[int]), p(Optional[int]), p(Optional[int])))
    register_type(NoReturn, make_raiser(IgnoreAttempt, 'Attempted to short circuit a NoReturn function')) # type: ignore
    
    # AsyncContextManager, lambda p: p(contextlib.AbstractAsyncContextManager),
    # AsyncGenerator: ,
    # AsyncIterable,
    # AsyncIterator,
    # Awaitable,
    # Coroutine: (handled via typeshed)
    # Generator: (handled via typeshed)
    
    register_type(NamedTuple, lambda p, *t: p(Tuple.__getitem__(tuple(t))))
    
    register_type(re.Pattern, lambda p, t=None: p(re.compile))  # type: ignore
    register_type(re.Match, lambda p, t=None: p(re.match))  # type: ignore
    
    # Text: (elsewhere - identical to str)
    register_type(bytes, lambda p: p(ByteString))
    register_type(bytearray, lambda p: p(ByteString))
    register_type(memoryview, lambda p: p(ByteString))
    # AnyStr,  (it's a type var)
    
    register_type(typing.BinaryIO, lambda p: io.BytesIO(p(ByteString)))
    # TODO: handle Any/AnyStr with a custom class that accepts str/bytes interchangably?:
    register_type(typing.IO, lambda p, t=Any: p(BinaryIO) if t == 'bytes' else p(TextIO))
    # TODO: StringIO (and BytesIO) won't accept SmtStr writes.
    # Consider clean symbolic implementations of these.
    register_type(typing.TextIO, lambda p: io.StringIO(str(p(str))))
    
    register_type(SupportsAbs, lambda p: p(int))
    register_type(SupportsFloat, lambda p: p(float))
    register_type(SupportsInt, lambda p: p(int))
    register_type(SupportsRound, lambda p: p(float))
    register_type(SupportsBytes, lambda p: p(ByteString))
    register_type(SupportsComplex, lambda p: p(complex))


    # Patches

    register_patch(orig_builtins, _len, 'len')
    register_patch(orig_builtins, _sorted, 'sorted')
    register_patch(orig_builtins, _issubclass, 'issubclass')
    register_patch(orig_builtins, _isinstance, 'isinstance')
    register_patch(orig_builtins, _hash, 'hash')
    register_patch(orig_builtins, _repr, 'repr')
    register_patch(orig_builtins, _max, 'max')
    register_patch(orig_builtins, _min, 'min')

    # Patches on str
    for name in [
            'center',
            'count',
            'encode',
            'endswith',
            'expandtabs',
            'find',
            'format', # TODO: shallow realization likely isn't sufficient
            'format_map',
            'index',
            'ljust',
            'lstrip',
            'partition',
            'replace',
            'rfind',
            'rindex',
            'rjust',
            'rpartition',
            'rsplit',
            'rstrip',
            'split',
            'splitlines',
            'startswith',
            'strip',
            'translate',
            'zfill',
    ]:
        orig_impl = getattr(orig_builtins.str, name)
        register_patch(orig_builtins.str, with_realized_args(orig_impl), name)

    # TODO: do a symbolic string concatenation
    orig_join = orig_builtins.str.join
    register_patch(orig_builtins.str, lambda s, l: orig_join(s, map(realize, l)), 'join')

    # TODO: override str.__new__ to make symbolic strings

    # Patches on list
    register_patch(orig_builtins.list, _list_index, 'index')
    # TODO: forbiddenfruit can't patch __repr__ yet:
    # register_patch(orig_builtins.list, _list_repr, '__repr__')

    setup_binops()
