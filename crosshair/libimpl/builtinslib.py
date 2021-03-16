import builtins as orig_builtins
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
import sys

from crosshair.abcstring import AbcString
from crosshair.core import inside_realization
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
from crosshair.simplestructs import ShellMutableMap
from crosshair.simplestructs import ShellMutableSequence
from crosshair.simplestructs import ShellMutableSet
from crosshair.statespace import context_statespace
from crosshair.statespace import StateSpace
from crosshair.statespace import HeapRef
from crosshair.statespace import prefer_true
from crosshair.statespace import SnapshotRef
from crosshair.statespace import model_value_to_python
from crosshair.statespace import VerificationStatus
from crosshair.type_repo import PYTYPE_SORT
from crosshair.util import debug
from crosshair.util import memo
from crosshair.util import CrosshairInternal
from crosshair.util import CrosshairUnsupported
from crosshair.util import IgnoreAttempt
from crosshair.util import is_iterable
from crosshair.util import is_hashable

import typing_inspect  # type: ignore
import z3  # type: ignore


class _Missing(enum.Enum):
    value = 0


_MISSING = _Missing.value
NoneType = type(None)


def smt_min(x, y):
    if x is y:
        return x
    return z3.If(x <= y, x, y)


def smt_and(a: bool, b: bool, *more: bool) -> bool:
    if isinstance(a, SmtBool) and isinstance(b, SmtBool):
        ret = SmtBool(z3.And(a.var, b.var))
    else:
        ret = a and b
    if not more:
        return ret
    return smt_and(ret, *more)


_HEAPABLE_PYTYPES = set([int, float, str, bool, NoneType, complex])


def pytype_uses_heap(typ: Type) -> bool:
    return not (typ in _HEAPABLE_PYTYPES)


def typeable_value(val: object) -> object:
    """
    Foces values of unknown type (SmtObject) into a typed (but possibly still symbolic) value.
    """
    while type(val) is SmtObject:
        val = cast(SmtObject, val)._wrapped()
    return val


_SMT_FLOAT_SORT = z3.RealSort()  # difficulty getting the solver to use z3.Float64()


@memo
def possibly_missing_sort(sort):
    datatype = z3.Datatype("optional_" + str(sort) + "_")
    datatype.declare("missing")
    datatype.declare("present", ("valueat", sort))
    ret = datatype.create()
    return ret


_MAYBE_HEAPREF = possibly_missing_sort(HeapRef)


def is_heapref_sort(sort: z3.SortRef) -> bool:
    return sort == HeapRef or sort == _MAYBE_HEAPREF


SmtGenerator = Callable[[Union[str, z3.ExprRef], type], object]


def origin_of(typ: Type) -> Type:
    typ = _WRAPPER_TYPE_TO_PYTYPE.get(typ, typ)
    if hasattr(typ, "__origin__"):
        return typ.__origin__
    return typ


# TODO: refactor away casting in SMT-sapce:
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


def smt_coerce(val: Any) -> z3.ExprRef:
    if isinstance(val, SmtBackedValue):
        return val.var
    return val


class SmtBackedValue(CrossHairValue):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        self.statespace = context_statespace()
        self.snapshot = SnapshotRef(-1)
        self.python_type = typ
        if type(smtvar) is str:
            self.var = self.__init_var__(typ, smtvar)
        else:
            self.var = smtvar
            # TODO test that smtvar's sort matches expected?

    def __init_var__(self, typ, varname):
        raise CrosshairInternal(f"__init_var__ not implemented in {type(self)}")

    def __deepcopy__(self, memo):
        if inside_realization():
            return self.__ch_realize__()
        shallow = copy.copy(self)
        shallow.snapshot = self.statespace.current_snapshot()
        return shallow

    def __bool__(self):
        return NotImplemented

    # TODO: do we need these comparison rejections?:
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

    def __ch_realize__(self) -> object:
        raise NotImplementedError(
            f"Realization not supported for {name_of_type(type(self))} instances"
        )

    def _unary_op(self, op):
        # ...
        return self.__class__(op(self.var), self.python_type)


class AtomicSmtValue(SmtBackedValue):
    def __init_var__(self, typ, varname):
        z3type = type(self)._ch_smt_sort()
        return z3.Const(varname, z3type)

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        raise CrosshairInternal(f"_ch_smt_sort not implemented in {cls}")

    @classmethod
    def _pytype(cls) -> Type:
        raise CrosshairInternal(f"_pytype not implemented in {cls}")

    @classmethod
    def _smt_promote_literal(cls, val: object) -> Optional[z3.SortRef]:
        raise CrosshairInternal(f"_smt_promote_literal not implemented in {cls}")

    @classmethod
    def _coerce_to_smt_sort(cls, input_value: Any) -> Optional[z3.ExprRef]:
        input_value = typeable_value(input_value)
        target_pytype = cls._pytype()

        # check the likely cases first
        if isinstance(input_value, cls):
            return input_value.var
        elif isinstance(input_value, target_pytype):
            return cls._smt_promote_literal(input_value)

        # see whether we can safely cast and retry
        if isinstance(input_value, Number) and issubclass(cls, Number):
            if isinstance(input_value, SmtBackedValue):
                casting_fn_name = "__" + target_pytype.__name__ + "__"
                converted = getattr(input_value, casting_fn_name)()
                return cls._coerce_to_smt_sort(converted)
            else:  # non-symbolic
                casted = target_pytype(input_value)
                if casted == input_value:
                    return cls._coerce_to_smt_sort(casted)

        return None

    def __eq__(self, other):
        with self.statespace.framework():
            coerced = type(self)._coerce_to_smt_sort(other)
            if coerced is None:
                return False
            return SmtBool(self.var == coerced)


_PYTYPE_TO_WRAPPER_TYPE: Dict[
    type, Tuple[Type[AtomicSmtValue], ...]
] = {}  # to be populated later
_WRAPPER_TYPE_TO_PYTYPE: Dict[SmtGenerator, type] = {}


def crosshair_types_for_python_type(typ: Type) -> Tuple[Type[AtomicSmtValue], ...]:
    typ = normalize_pytype(typ)
    origin = origin_of(typ)
    return _PYTYPE_TO_WRAPPER_TYPE.get(origin, ())


def smt_to_ch_value(
    space: StateSpace, snapshot: SnapshotRef, smt_val: z3.ExprRef, pytype: type
) -> object:
    def proxy_generator(typ: Type) -> object:
        return proxy_for_type(typ, name_of_type(typ) + "_inheap" + space.uniq())

    if smt_val.sort() == HeapRef:
        return space.find_key_in_heap(smt_val, pytype, proxy_generator, snapshot)
    ch_type = crosshair_types_for_python_type(pytype)
    assert ch_type
    return ch_type[0](smt_val, pytype)


def force_to_smt_sort(
    input_value: Any, desired_ch_type: Type[AtomicSmtValue]
) -> z3.ExprRef:
    ret = desired_ch_type._coerce_to_smt_sort(input_value)
    if ret is None:
        raise TypeError(f"Could not derive smt from {input_value}:{type(input_value)}")
    return ret


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
            if issubclass(a_type, cur_a_type) and issubclass(b_type, cur_b_type):
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
    a, b = hints["a"], hints["b"]
    if typing_inspect.get_origin(a) == Union:
        a = typing_inspect.get_args(a)
    else:
        a = [a]
    if typing_inspect.get_origin(b) == Union:
        b = typing_inspect.get_args(b)
    else:
        b = [b]
    return (a, b)


def setup_promotion(
    fn: Callable[[Number, Number], Tuple[Number, Number]], reg_ops: Set[BinFn]
):
    a, b = _binop_type_hints(fn)
    for a_type in a:
        for b_type in b:
            for op in reg_ops:

                def forward(o, x, y):
                    x2, y2 = fn(x, y)
                    return numeric_binop(o, x2, y2)

                def backward(o, x, y):
                    y2, x2 = fn(y, x)
                    return numeric_binop(o, x2, y2)

                _BIN_OPS_SEARCH_ORDER.append((op, a_type, b_type, forward))
                _BIN_OPS_SEARCH_ORDER.append((op, b_type, a_type, backward))


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

                    _BIN_OPS_SEARCH_ORDER.append(
                        (_FLIPPED_OPS[op], b_type, a_type, flipped)
                    )


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


def apply_smt(op: BinFn, x: z3.ExprRef, y: z3.ExprRef) -> z3.ExprRef:
    # Mostly, z3 overloads operators and things just work.
    # But some special cases need to be checked first.
    space = context_statespace()
    if op in _ARITHMETIC_OPS:
        if op in (ops.truediv, ops.floordiv, ops.mod):
            if space.smt_fork(y == 0):
                raise ZeroDivisionError("division by zero")
            if op == ops.floordiv:
                if space.smt_fork(y >= 0):
                    if space.smt_fork(x >= 0):
                        return x / y
                    else:
                        return -((y - x - 1) / y)
                else:
                    if space.smt_fork(x >= 0):
                        return -((x - y - 1) / -y)
                    else:
                        return -x / -y
            if op == ops.mod:
                if space.smt_fork(y >= 0):
                    return x % y
                elif space.smt_fork(x % y == 0):
                    return 0
                else:
                    return (x % y) + y
        elif op == ops.pow:
            if space.smt_fork(z3.And(x == 0, y < 0)):
                raise ZeroDivisionError("zero cannot be raised to a negative power")
    elif op in _BITWISE_OPS:
        if op in (ops.lshift, ops.rshift):
            if space.smt_fork(y < 0):
                raise ValueError("negative shift count")
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
        return (SmtInt(z3.If(a.var, 1, 0)), b)

    setup_promotion(_, _ALL_OPS)

    # Implicitly upconvert symbolic ints to floats.
    def _(a: SmtInt, b: Union[float, FiniteFloat, SmtFloat, complex]):
        return (SmtFloat(z3.ToReal(a.var)), b)

    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # Implicitly upconvert native ints to floats.
    def _(a: int, b: SmtFloat):
        return (float(a), b)

    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # Implicitly upconvert native bools to ints.
    def _(a: bool, b: Union[SmtInt, SmtFloat]):
        return (int(a), b)

    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # complex
    def _(op: BinFn, a: SmtNumberAble, b: complex):
        return op(complex(a), b)  # type: ignore

    setup_binop(_, _ALL_OPS)

    # float
    def _(op: BinFn, a: SmtFloat, b: SmtFloat):
        return SmtFloat(apply_smt(op, a.var, b.var))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: SmtFloat, b: SmtFloat):
        return SmtBool(apply_smt(op, a.var, b.var))

    setup_binop(_, _COMPARISON_OPS)

    def _(op: BinFn, a: SmtFloat, b: FiniteFloat):
        return SmtFloat(apply_smt(op, a.var, z3.RealVal(b)))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: FiniteFloat, b: SmtFloat):
        return SmtFloat(apply_smt(op, z3.RealVal(a), b.var))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: SmtFloat, b: FiniteFloat):
        return SmtBool(apply_smt(op, a.var, z3.RealVal(b)))

    setup_binop(_, _COMPARISON_OPS)

    # int
    def _(op: BinFn, a: SmtInt, b: SmtInt):
        return SmtInt(apply_smt(op, a.var, b.var))

    setup_binop(_, _ARITHMETIC_AND_BITWISE_OPS)

    def _(op: BinFn, a: SmtInt, b: SmtInt):
        return SmtBool(apply_smt(op, a.var, b.var))

    setup_binop(_, _COMPARISON_OPS)

    def _(op: BinFn, a: SmtInt, b: int):
        return SmtInt(apply_smt(op, a.var, z3.IntVal(b)))

    setup_binop(_, _ARITHMETIC_AND_BITWISE_OPS)

    def _(op: BinFn, a: int, b: SmtInt):
        return SmtInt(apply_smt(op, z3.IntVal(a), b.var))

    setup_binop(_, _ARITHMETIC_AND_BITWISE_OPS)

    def _(op: BinFn, a: SmtInt, b: int):
        return SmtBool(apply_smt(op, a.var, z3.IntVal(b)))

    setup_binop(_, _COMPARISON_OPS)

    def _(
        op: BinFn, a: Integral, b: Integral
    ):  # Most bitwise operators require realization
        return op(a.__index__(), b.__index__())  # type: ignore

    setup_binop(_, {ops.and_, ops.or_, ops.xor})

    def _(
        op: BinFn, a: Integral, b: Integral
    ):  # Floor division over ints requires realization, at present
        return op(a.__index__(), b.__index__())  # type: ignore

    setup_binop(_, {ops.truediv})

    def _(a: SmtInt, b: Number):  # Division over ints must produce float
        return (a.__float__(), b)

    setup_promotion(_, {ops.truediv})

    # bool
    def _(op: BinFn, a: SmtBool, b: SmtBool):
        return SmtBool(apply_smt(op, a.var, b.var))

    setup_binop(_, {ops.eq, ops.ne})

    def _(op: BinFn, a: SmtBool, b: bool):
        return SmtInt(apply_smt(op, a.var, z3.BoolVal(b)))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: bool, b: SmtBool):
        return SmtInt(apply_smt(op, z3.BoolVal(a), b.var))

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
            count = self.var  # z3.If(self.var >= 0, self.var, 0))
            result = SmtStr(f"{self.var}_str{space.uniq()}")
            space.add(z3.InRe(result.var, z3.Star(z3.Re(other))))
            space.add(z3.Length(result.var) == len(other) * count)
            return result
        return numeric_binop(ops.mul, self, other)

    __rmul__ = __mul__


class SmtBool(AtomicSmtValue, SmtIntable):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = bool):
        assert typ == bool
        SmtBackedValue.__init__(self, smtvar, typ)

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return z3.BoolSort()

    @classmethod
    def _pytype(cls) -> Type:
        return bool

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, bool):
            return z3.BoolVal(literal)
        return None

    def __ch_realize__(self) -> object:
        return self.statespace.choose_possible(self.var)

    def __neg__(self):
        return SmtInt(z3.If(self.var, -1, 0))

    def __repr__(self):
        return self.__bool__().__repr__()

    def __hash__(self):
        return self.__bool__().__hash__()

    def __index__(self):
        return SmtInt(z3.If(self.var, 1, 0))

    def __bool__(self):
        return self.statespace.choose_possible(self.var)

    def __int__(self):
        return SmtInt(z3.If(self.var, 1, 0))

    def __float__(self):
        return SmtFloat(smt_bool_to_float(self.var))

    def __complex__(self):
        return complex(self.__float__())

    def __round__(self, ndigits=None):
        # This could be smarter, but nobody rounds a bool right?:
        return round(realize(self), realize(ndigits))


class SmtInt(AtomicSmtValue, SmtIntable):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = int):
        assert typ == int
        SmtIntable.__init__(self, smtvar, typ)

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return z3.IntSort()

    @classmethod
    def _pytype(cls) -> Type:
        return int

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, int):
            return z3.IntVal(literal)
        return None

    @classmethod
    def from_bytes(cls, b: bytes, byteorder: str, signed=False) -> int:
        return int.from_bytes(b, byteorder, signed=signed)

    def __ch_realize__(self) -> object:
        return self.statespace.find_model_value(self.var)

    def __repr__(self):
        return self.__index__().__repr__()
        # TODO: do a symbolic conversion!:
        # return SmtStr(z3.IntToStr(self.var))

    def __hash__(self):
        return self.__index__().__hash__()

    def __float__(self):
        return SmtFloat(smt_int_to_float(self.var))

    def __complex__(self):
        return complex(self.__float__())

    def __index__(self):
        if self == 0:
            return 0
        ret = self.statespace.find_model_value(self.var)
        assert type(ret) is int, f"SmtInt with wrong SMT var type ({type(ret)})"
        return ret

    def __bool__(self):
        return SmtBool(self.var != 0).__bool__()

    def __int__(self):
        return self.__index__()

    def __round__(self, ndigits=None):
        if ndigits is None or ndigits >= 0:
            return self  # TODO: test
        return round(self.__index__(), ndigits)  # TODO: could do this symbolically

    def bit_length(self) -> int:
        return realize(self).bit_length()

    def to_bytes(self, length, byteorder, *, signed=False):
        return realize(self).to_bytes(length, byteorder, signed=signed)

    def as_integer_ratio(self) -> Tuple["SmtInt", int]:
        return (self, 1)


_Z3_ONE_HALF = z3.RealVal("1/2")


class SmtFloat(AtomicSmtValue, SmtNumberAble):
    def __new__(
        mytype, firstarg: Union[None, str, z3.ExprRef] = None, pytype: Type = float
    ):
        if not isinstance(firstarg, (str, z3.ExprRef, NoneType)):  # type: ignore
            # The Python staticstics module pulls types of values and assumes it can
            # re-create those types by calling the type.
            # See https://github.com/pschanely/CrossHair/issues/94
            return float(firstarg)  # type: ignore
        return object.__new__(mytype)

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = float):
        assert typ is float, f"SmtFloat with unexpected python type ({type(typ)})"
        SmtBackedValue.__init__(self, smtvar, typ)

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return z3.RealSort()

    @classmethod
    def _pytype(cls) -> Type:
        return float

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, float):
            # return z3.FPVal(literal, _SMT_FLOAT_SORT)
            return z3.RealVal(literal)
        return None

    def __ch_realize__(self) -> object:
        return self.statespace.find_model_value(self.var).__float__()  # type: ignore

    def __repr__(self):
        return self.statespace.find_model_value(self.var).__repr__()

    def __hash__(self):
        return self.statespace.find_model_value(self.var).__hash__()

    def __bool__(self):
        return SmtBool(self.var != 0).__bool__()

    def __int__(self):
        var = self.var
        return SmtInt(z3.If(var >= 0, z3.ToInt(var), -z3.ToInt(-var)))

    def __float__(self):
        return self.__ch_realize__()

    def __complex__(self):
        return complex(self.__float__())

    def __round__(self, ndigits=None):
        if ndigits is not None:
            factor = 10 ** realize(
                ndigits
            )  # realize to avoid exponentation-to-variable
            return round(self * factor) / factor
        else:
            var, floor, nearest = (
                self.var,
                z3.ToInt(self.var),
                z3.ToInt(self.var + _Z3_ONE_HALF),
            )
            return SmtInt(
                z3.If(
                    var != floor + _Z3_ONE_HALF,
                    nearest,
                    z3.If(floor % 2 == 0, floor, floor + 1),
                )
            )

    def __floor__(self):
        return SmtInt(z3.ToInt(self.var))

    def __ceil__(self):
        var, floor = self.var, z3.ToInt(self.var)
        return SmtInt(z3.If(var == floor, floor, floor + 1))

    def __mod__(self, other):
        return realize(self) % realize(
            other
        )  # TODO: z3 does not support modulo on reals

    def __trunc__(self):
        var, floor = self.var, z3.ToInt(self.var)
        return SmtInt(z3.If(var >= 0, floor, floor + 1))

    def as_integer_ratio(self) -> Tuple[Integral, Integral]:
        space = context_statespace()
        numerator = SmtInt("numerator" + space.uniq())
        denominator = SmtInt("denominator" + space.uniq())
        space.add(denominator.var > 0)
        space.add(numerator.var == denominator.var * self.var)
        # There are many valid integer ratios to return. Experimentally, both
        # z3 and CPython tend to pick the same ones. But verify this, while
        # deferring materialization:
        def ratio_is_chosen_by_cpython() -> bool:
            return realize(self).as_integer_ratio() == (numerator, denominator)

        space.defer_assumption(
            "float.as_integer_ratio gets the right ratio", ratio_is_chosen_by_cpython
        )
        return (numerator, denominator)

    def is_integer(self) -> SmtBool:
        return SmtBool(z3.IsInt(self.var))

    def hex(self) -> str:
        return realize(self).hex()


class SmtDictOrSet(SmtBackedValue):
    """
    TODO: Ordering is a challenging issue here.
    Modern pythons have in-order iteration for dictionaries but not sets.
    """

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        self.key_pytype = normalize_pytype(type_arg_of(typ, 0))
        ch_types = crosshair_types_for_python_type(self.key_pytype)
        if ch_types:
            self.ch_key_type: Optional[Type[AtomicSmtValue]] = ch_types[0]
            self.smt_key_sort = self.ch_key_type._ch_smt_sort()
        else:
            self.ch_key_type = None
            self.smt_key_sort = HeapRef
        SmtBackedValue.__init__(self, smtvar, typ)
        self.statespace.add(self._len() >= 0)

    def __ch_realize__(self):
        return origin_of(self.python_type)(self)  # TODO: make this a deep-realization

    def _arr(self):
        return self.var[0]

    def _len(self):
        return self.var[1]

    def __len__(self):
        return SmtInt(self._len())

    def __bool__(self):
        return SmtBool(self._len() != 0).__bool__()


class SmtDict(SmtDictOrSet, collections.abc.Mapping):
    """ An immutable symbolic dictionary. """

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        space = context_statespace()
        self.val_pytype = normalize_pytype(type_arg_of(typ, 1))
        val_ch_types = crosshair_types_for_python_type(self.val_pytype)
        if val_ch_types:
            self.ch_val_type: Optional[Type[AtomicSmtValue]] = val_ch_types[0]
            self.smt_val_sort = self.ch_val_type._ch_smt_sort()
        else:
            self.ch_val_type = None
            self.smt_val_sort = HeapRef
        SmtDictOrSet.__init__(self, smtvar, typ)
        arr_var = self._arr()
        len_var = self._len()
        self.val_missing_checker = arr_var.sort().range().recognizer(0)
        self.val_missing_constructor = arr_var.sort().range().constructor(0)
        self.val_constructor = arr_var.sort().range().constructor(1)
        self.val_accessor = arr_var.sort().range().accessor(1, 0)
        self.empty = z3.K(arr_var.sort().domain(), self.val_missing_constructor())
        self._iter_cache: List[z3.Const] = []
        space.add((arr_var == self.empty) == (len_var == 0))

        def list_can_be_iterated():
            list(self)
            return True

        space.defer_assumption(
            "dict iteration is consistent with items", list_can_be_iterated
        )

    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        arr_smt_sort = z3.ArraySort(
            self.smt_key_sort, possibly_missing_sort(self.smt_val_sort)
        )
        return (
            z3.Const(varname + "_map" + self.statespace.uniq(), arr_smt_sort),
            z3.Const(varname + "_len" + self.statespace.uniq(), z3.IntSort()),
        )

    def __eq__(self, other):
        (self_arr, self_len) = self.var
        has_heapref = is_heapref_sort(self.var[0].sort().domain()) or is_heapref_sort(
            self.var[0].sort().range()
        )
        if not has_heapref:
            if isinstance(other, SmtDict):
                (other_arr, other_len) = other.var
                return SmtBool(z3.And(self_len == other_len, self_arr == other_arr))
        # Manually check equality. Drive the loop from the (likely) concrete value 'other':
        if not isinstance(other, collections.abc.Mapping):
            return False
        if len(self) != len(other):
            return False
        for k, v in other.items():
            self_v = self.get(k, _MISSING)
            if self_v is _MISSING or self[k] != v:
                return False
        return True

    def __repr__(self):
        return str(dict(self.items()))

    def __getitem__(self, k):
        with self.statespace.framework():
            smt_key = None
            if self.ch_key_type:
                smt_key = self.ch_key_type._coerce_to_smt_sort(k)
            if smt_key is None:
                if getattr(k, "__hash__", None) is None:
                    raise TypeError("unhashable type")
                for self_k in iter(self):
                    if self_k == k:
                        return self[self_k]
                raise KeyError(k)
            possibly_missing = self._arr()[smt_key]
            is_missing = self.val_missing_checker(possibly_missing)
            if SmtBool(is_missing).__bool__():
                raise KeyError(k)
            if SmtBool(self._len() == 0).__bool__():
                raise IgnoreAttempt("SmtDict in inconsistent state")
            return smt_to_ch_value(
                self.statespace,
                self.snapshot,
                self.val_accessor(possibly_missing),
                self.val_pytype,
            )

    def __reversed__(self):
        return reversed(list(self))

    def __iter__(self):
        arr_var, len_var = self.var
        iter_cache = self._iter_cache
        space = self.statespace
        idx = 0
        arr_sort = self._arr().sort()
        is_missing = self.val_missing_checker
        while SmtBool(idx < len_var).__bool__():
            if not space.choose_possible(arr_var != self.empty, favor_true=True):
                raise IgnoreAttempt("SmtDict in inconsistent state")
            k = z3.Const("k" + str(idx) + space.uniq(), arr_sort.domain())
            v = z3.Const("v" + str(idx) + space.uniq(), self.val_constructor.domain(0))
            remaining = z3.Const("remaining" + str(idx) + space.uniq(), arr_sort)
            space.add(arr_var == z3.Store(remaining, k, self.val_constructor(v)))
            space.add(is_missing(z3.Select(remaining, k)))

            # TODO: is this true now? it's immutable these days?
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
            raise IgnoreAttempt("SmtDict in inconsistent state")

    def copy(self):
        return SmtDict(self.var, self.python_type)

    # TODO: investigate this approach for type masquerading:
    # @property
    # def __class__(self):
    #    return dict


class SmtSet(SmtDictOrSet, collections.abc.Set):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        SmtDictOrSet.__init__(self, smtvar, typ)
        self._iter_cache: List[z3.Const] = []
        self.empty = z3.K(self._arr().sort().domain(), False)
        self.statespace.add((self._arr() == self.empty) == (self._len() == 0))

    def __eq__(self, other):
        (self_arr, self_len) = self.var
        if isinstance(other, SmtSet):
            (other_arr, other_len) = other.var
            if other_arr.sort() == self_arr.sort():
                return SmtBool(z3.And(self_len == other_len, self_arr == other_arr))
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
            z3.Const(
                varname + "_map" + self.statespace.uniq(),
                z3.ArraySort(self.smt_key_sort, z3.BoolSort()),
            ),
            z3.Const(varname + "_len" + self.statespace.uniq(), z3.IntSort()),
        )

    def __contains__(self, key):
        if getattr(key, "__hash__", None) is None:
            raise TypeError("unhashable type")
        if self.ch_key_type:
            k = self.ch_key_type._coerce_to_smt_sort(key)
        else:
            k = None
        if k is not None:
            present = self._arr()[k]
            return SmtBool(present)
        # Fall back to standard equality and iteration
        for self_item in self:
            if self_item == key:
                return True
        return False

    def __iter__(self):
        arr_var, len_var = self.var
        iter_cache = self._iter_cache
        space = self.statespace
        idx = 0
        arr_sort = self._arr().sort()
        keys_on_heap = is_heapref_sort(arr_sort.domain())
        already_yielded = []
        while SmtBool(idx < len_var).__bool__():
            if not space.choose_possible(arr_var != self.empty, favor_true=True):
                raise IgnoreAttempt("SmtSet in inconsistent state")
            k = z3.Const("k" + str(idx) + space.uniq(), arr_sort.domain())
            remaining = z3.Const("remaining" + str(idx) + space.uniq(), arr_sort)
            space.add(arr_var == z3.Store(remaining, k, True))
            space.add(z3.Not(z3.Select(remaining, k)))

            if idx > len(iter_cache):
                raise CrosshairInternal()
            if idx == len(iter_cache):
                iter_cache.append(k)
            else:
                space.add(k == iter_cache[idx])

            idx += 1
            ch_value = smt_to_ch_value(space, self.snapshot, k, self.key_pytype)
            if keys_on_heap:
                # need to confirm that we do not yield two keys that are __eq__
                for previous_value in already_yielded:
                    if not prefer_true(ch_value != previous_value):
                        raise IgnoreAttempt("Duplicate items in set")
                already_yielded.append(ch_value)
            yield ch_value
            arr_var = remaining
        # In this conditional, we reconcile the parallel symbolic variables for length
        # and contents:
        if not self.statespace.choose_possible(arr_var == self.empty, favor_true=True):
            raise IgnoreAttempt("SmtSet in inconsistent state")

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
        return self._set_op("__ge__", other)

    def __gt__(self, other):
        return self._set_op("__gt__", other)

    def __le__(self, other):
        return self._set_op("__le__", other)

    def __lt__(self, other):
        return self._set_op("__lt__", other)

    def __and__(self, other):
        return self._set_op("__and__", other)

    __rand__ = __and__

    def __or__(self, other):
        return self._set_op("__or__", other)

    __ror__ = __or__

    def __xor__(self, other):
        return self._set_op("__xor__", other)

    __rxor__ = __xor__

    def __sub__(self, other):
        return self._set_op("__sub__", other)


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
    space: StateSpace, i: slice, smt_len: z3.ExprRef, fork_on_negative_index=False
) -> Union[z3.ExprRef, Tuple[z3.ExprRef, z3.ExprRef]]:
    def normalize_symbolic_index(idx) -> z3.ExprRef:
        if type(idx) is int:
            return z3.IntVal(idx) if idx >= 0 else (smt_len + z3.IntVal(idx))
        elif fork_on_negative_index:
            smt_idx = SmtInt._coerce_to_smt_sort(idx)
            if idx >= 0:
                return smt_idx
            else:
                return smt_len + smt_idx
        else:
            smt_idx = SmtInt._coerce_to_smt_sort(idx)
            return z3.If(smt_idx >= z3.IntVal(0), smt_idx, smt_len + smt_idx)

    if isinstance(i, (int, SmtInt)):
        smt_i = SmtInt._coerce_to_smt_sort(i)
        if space.smt_fork(z3.Or(smt_i >= smt_len, smt_i < -smt_len)):
            raise IndexError(f'index "{i}" is out of range')
        return normalize_symbolic_index(i)
    elif isinstance(i, slice):
        start, stop, step = (i.start, i.stop, i.step)
        for x in (start, stop, step):
            if (x is not None) and (not hasattr(x, "__index__")):
                raise TypeError(
                    "slice indices must be integers or None or have an __index__ method"
                )
        if step not in (None, 1):
            raise CrosshairUnsupported("slice steps not handled")  # TODO: handle this!
        start = normalize_symbolic_index(start) if i.start is not None else z3.IntVal(0)
        stop = normalize_symbolic_index(stop) if i.stop is not None else smt_len
        return (start, stop)
    else:
        raise TypeError("indices must be integers or slices, not " + str(type(i)))


class SmtSequence(SmtBackedValue, collections.abc.Sequence):
    def __ch_realize__(self):
        return origin_of(self.python_type)(self)

    def __iter__(self):
        idx = 0
        while len(self) > idx:
            yield self[idx]
            idx += 1

    def __len__(self):
        return SmtInt(z3.Length(self.var))

    def __bool__(self):
        return SmtBool(z3.Length(self.var) > 0).__bool__()

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
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        if type(smtvar) == str:
            pass
        else:
            assert type(smtvar) is tuple, f"incorrect type {type(smtvar)}"
            assert len(smtvar) == 2

        self.val_pytype = normalize_pytype(type_arg_of(typ, 0))
        ch_types = crosshair_types_for_python_type(self.val_pytype)
        if ch_types:
            self.ch_item_type: Optional[Type[AtomicSmtValue]] = ch_types[0]
            self.item_smt_sort = self.ch_item_type._ch_smt_sort()
        else:
            self.ch_item_type = None
            self.item_smt_sort = HeapRef

        SmtBackedValue.__init__(self, smtvar, typ)
        arr_var = self._arr()
        len_var = self._len()
        self.statespace.add(len_var >= 0)

    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        arr_smt_type = z3.ArraySort(z3.IntSort(), self.item_smt_sort)
        return (
            z3.Const(varname + "_map" + self.statespace.uniq(), arr_smt_type),
            z3.Const(varname + "_len" + self.statespace.uniq(), z3.IntSort()),
        )

    def _arr(self):
        return self.var[0]

    def _len(self):
        return self.var[1]

    def __len__(self):
        return SmtInt(self._len())

    def __bool__(self) -> bool:
        return SmtBool(self._len() != 0).__bool__()

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
        while SmtBool(idx < len_var).__bool__():
            yield smt_to_ch_value(
                self.statespace, self.snapshot, z3.Select(arr_var, idx), self.val_pytype
            )
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
            if not is_heapref_sort(self.item_smt_sort):
                smt_other = self.ch_item_type._coerce_to_smt_sort(other)
                if smt_other is not None:
                    # OK to perform a symbolic comparison
                    idx = z3.Const("possible_idx" + space.uniq(), z3.IntSort())
                    idx_in_range = z3.Exists(
                        idx,
                        z3.And(
                            0 <= idx,
                            idx < self._len(),
                            z3.Select(self._arr(), idx) == smt_other,
                        ),
                    )
                    return SmtBool(idx_in_range)
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
                start = SmtInt(start)
                stop = SmtInt(smt_min(mylen, smt_coerce(stop)))
                return SliceView(self, start, stop)
            else:
                smt_result = z3.Select(self._arr(), idx_or_pair)
                return smt_to_ch_value(
                    space, self.snapshot, smt_result, self.val_pytype
                )

    def insert(self, idx, obj):
        raise CrosshairUnsupported


class SmtList(ShellMutableSequence, collections.abc.MutableSequence, CrossHairValue):
    def __init__(self, *a):
        ShellMutableSequence.__init__(self, SmtArrayBasedUniformTuple(*a))

    def __ch_pytype__(self):
        return python_type(self.inner)

    def __ch_realize__(self):
        return list(map(realize, self))

    def _is_subclass_of_(cls, other):
        return other is list

    def __lt__(self, other):
        if not isinstance(other, (list, SmtList)):
            raise TypeError
        return super().__lt__(other)

    def __mod__(self, *a):
        raise TypeError

    def index(
        self, value: object, start: int = 0, stop: int = 9223372036854775807
    ) -> int:
        """
        Return first index of value.
        Raises ValueError if the value is not present.
        """
        try:
            start, stop = start.__index__(), stop.__index__()
        except AttributeError:
            # Re-create the error that list.index would give on bad start/stop values:
            raise TypeError(
                "slice indices must be integers or have an __index__ method"
            )
        self_len = self.__len__()
        if self_len < stop:
            stop = self_len
        i = start
        while i < self_len and i < stop:
            cur = self[i]
            if cur == value:
                return i
            i += 1
        raise ValueError(f"{value} is not in list")


class SmtType(AtomicSmtValue, SmtBackedValue):
    _realization: Optional[Type] = None

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        space = context_statespace()
        assert origin_of(typ) is type
        self.pytype_cap = (
            origin_of(typ.__args__[0]) if hasattr(typ, "__args__") else object
        )
        assert type(self.pytype_cap) is type
        smt_cap = space.type_repo.get_type(self.pytype_cap)
        SmtBackedValue.__init__(self, smtvar, typ)
        space.add(space.type_repo.smt_issubclass(self.var, smt_cap))

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return PYTYPE_SORT

    @classmethod
    def _pytype(cls) -> Type:
        return type

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, type):
            return context_statespace().type_repo.get_type(literal)
        return None

    def _is_superclass_of_(self, other):
        if self is SmtType:
            return False
        if type(other) is SmtType:
            # Prefer it this way because only _is_subcless_of_ does the type cap lowering.
            return other._is_subclass_of_(self)
        space = self.statespace
        with space.framework():
            coerced = SmtType._coerce_to_smt_sort(other)
            if coerced is None:
                return False
            return SmtBool(space.type_repo.smt_issubclass(coerced, self.var))

    def _is_subclass_of_(self, other):
        if self is SmtType:
            return False
        space = self.statespace
        with space.framework():
            coerced = SmtType._coerce_to_smt_sort(other)
            if coerced is None:
                return False
            ret = SmtBool(space.type_repo.smt_issubclass(self.var, coerced))
            if type(other) is SmtType:
                other_pytype = other.pytype_cap
            elif issubclass(other, SmtBackedValue):
                if issubclass(other, AtomicSmtValue):
                    other_pytype = other._pytype()
                else:
                    other_pytype = None
            else:
                other_pytype = other
            # consider lowering the type cap
            if (
                other_pytype not in (None, self.pytype_cap)
                and issubclass(other_pytype, self.pytype_cap)
                and ret
            ):
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
            raise CrosshairUnsupported("Will not exhaustively attempt `object` types")
        else:
            subtype = choose_type(space, cap)
            smt_type = space.type_repo.get_type(subtype)
            if space.smt_fork(self.var != smt_type):
                raise IgnoreAttempt
            return subtype

    def __call__(self, *a, **kw):
        return self._realized()(*a, **kw)

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
        inner = object.__getattribute__(self, "_inner")
        if inner is _MISSING:
            inner = self._realize()
            object.__setattr__(self, "_inner", inner)
        return inner

    def __ch_realize__(self):
        return realize(self._wrapped())

    def __deepcopy__(self, memo):
        if inside_realization():
            return self.__ch_realize__()
        inner = object.__getattribute__(self, "_inner")
        if inner is _MISSING:
            # CrossHair will deepcopy for mutation checking.
            # That's usually bad for LazyObjects, which want to defer their
            # realization, so we simply don't do mutation checking for these
            # kinds of values right now.
            return self
        else:
            return copy.deepcopy(inner)


class SmtObject(LazyObject, CrossHairValue):
    """
    An object with an unknown type.
    We lazily create a more specific smt-based value in hopes that an
    isinstance() check will be called before something is accessed on us.
    Note that this class is not an SmtBackedValue, but its _typ and _inner
    members can be.
    """

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        object.__setattr__(self, "_typ", SmtType(smtvar, type))
        object.__setattr__(self, "_space", context_statespace())
        object.__setattr__(self, "_varname", smtvar)

    def _realize(self):
        space = object.__getattribute__(self, "_space")
        varname = object.__getattribute__(self, "_varname")

        typ = object.__getattribute__(self, "_typ")
        pytype = realize(typ)
        debug("materializing the type of symbolic", varname, "to be", pytype)
        if pytype is object:
            return object()
        return proxy_for_type(pytype, varname, allow_subtypes=False)

    @property
    def python_type(self):
        return object.__getattribute__(self, "_typ")

    @property
    def __class__(self):
        return SmtObject

    @__class__.setter
    def __class__(self, value):
        raise CrosshairUnsupported


class SmtCallable(SmtBackedValue):
    __closure__ = None

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        SmtBackedValue.__init__(self, smtvar, typ)

    def __bool__(self):
        return True

    def __eq__(self, other):
        return (self.var is other.var) if isinstance(other, SmtCallable) else False

    def __hash__(self):
        return id(self.var)

    def __init_var__(self, typ: type, varname):
        type_args: Tuple[Any, ...] = type_args_of(self.python_type)
        if not type_args:
            type_args = (..., Any)
        (self.arg_pytypes, self.ret_pytype) = type_args
        if self.arg_pytypes == ...:
            raise CrosshairUnsupported
        arg_ch_types = []
        for arg_pytype in self.arg_pytypes:
            ch_types = crosshair_types_for_python_type(arg_pytype)
            if not ch_types:
                raise CrosshairUnsupported
            arg_ch_types.append(ch_types[0])
        self.arg_ch_types = arg_ch_types
        ret_ch_types = crosshair_types_for_python_type(self.ret_pytype)
        if not ret_ch_types:
            raise CrosshairUnsupported
        self.ret_ch_type = ret_ch_types[0]
        return z3.Function(
            varname + self.statespace.uniq(),
            *[ch_type._ch_smt_sort() for ch_type in arg_ch_types],
            self.ret_ch_type._ch_smt_sort(),
        )

    def __ch_realize__(self):
        return eval(self.__repr__())

    def __call__(self, *args):
        space = self.statespace
        if len(args) != len(self.arg_ch_types):
            raise TypeError("wrong number of arguments")
        smt_args = []
        for actual_arg, ch_type in zip(args, self.arg_ch_types):
            smt_arg = ch_type._coerce_to_smt_sort(actual_arg)
            if smt_arg is None:
                raise TypeError
            smt_args.append(smt_arg)
        smt_ret = self.var(*smt_args)
        # TODO: detect that `smt_ret` might be a HeapRef here
        return self.ret_ch_type(smt_ret, self.ret_pytype)

    def __repr__(self):
        finterp = self.statespace.find_model_value_for_function(self.var)
        if finterp is None:
            # (z3 model completion will not interpret a function for me currently)
            return "lambda *a: None"
        # 0-arg interpretations seem to be simply values:
        if type(finterp) is not z3.FuncInterp:
            return "lambda :" + repr(model_value_to_python(finterp))
        if finterp.arity() < 10:
            arg_names = [chr(ord("a") + i) for i in range(finterp.arity())]
        else:
            arg_names = ["a" + str(i + 1) for i in range(finterp.arity())]
        entries = finterp.as_list()
        body = repr(model_value_to_python(entries[-1]))
        for entry in reversed(entries[:-1]):
            conditions = [
                "{} == {}".format(arg, repr(model_value_to_python(val)))
                for (arg, val) in zip(arg_names, entry[:-1])
            ]
            body = "{} if ({}) else ({})".format(
                repr(model_value_to_python(entry[-1])), " and ".join(conditions), body
            )
        arg_str = ", ".join(arg_names)
        if len(arg_names) > 1:  # Enclose args in params if >1 arg
            arg_str = f"({arg_str})"
        return f"lambda {arg_str}: {body}"


class SmtUniformTuple(
    SmtArrayBasedUniformTuple, collections.abc.Sequence, collections.abc.Hashable
):
    def __repr__(self):
        return tuple(self).__repr__()

    def __hash__(self):
        return tuple(self).__hash__()


_SMTSTR_Z3_SORT = z3.SeqSort(z3.BitVecSort(8))


class SmtStr(AtomicSmtValue, SmtSequence, AbcString):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = str):
        assert typ == str
        SmtBackedValue.__init__(self, smtvar, typ)
        self.item_pytype = str

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return _SMTSTR_Z3_SORT

    @classmethod
    def _pytype(cls) -> Type:
        return str

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, str):
            if len(literal) <= 1:
                if len(literal) == 0:
                    return z3.Empty(_SMTSTR_Z3_SORT)
                return z3.Unit(z3.BitVecVal(ord(literal), 8))
            return z3.Concat([z3.Unit(z3.BitVecVal(ord(ch), 8)) for ch in literal])
        return None

    def __ch_realize__(self) -> object:
        return self.statespace.find_model_value(self.var)

    def __str__(self):
        return self.statespace.find_model_value(self.var)

    def __copy__(self):
        return SmtStr(self.var)

    def __repr__(self):
        return repr(self.__str__())

    def __hash__(self):
        return hash(self.__str__())

    def __add__(self, other):
        if isinstance(other, (SmtStr, str)):
            return SmtStr(self.var + smt_coerce(other))
        raise TypeError

    def __radd__(self, other):
        if isinstance(other, (SmtStr, str)):
            return SmtStr(smt_coerce(other) + self.var)
        raise TypeError

    def __mul__(self, other):
        space = self.statespace
        # If repetition count is a literal, use that first:
        if isinstance(other, Integral):
            if other <= 1:
                return self if other == 1 else ""
            # Note that in SmtInt, we attempt string multiplication via regex.
            # Z3 cannot do much with a symbolic regex, so we case-split on
            # the repetition count.
            return SmtStr(z3.Concat(*[self.var for _ in range(other)]))
        return NotImplemented

    __rmul__ = __mul__

    def __mod__(self, other):
        return self.__str__() % realize(other)

    def _cmp_op(self, other, op):
        forced = force_to_smt_sort(other, SmtStr)
        return SmtBool(op(self.var, forced))

    def __lt__(self, other):
        return self._cmp_op(other, ops.lt)

    def __le__(self, other):
        return self._cmp_op(other, ops.le)

    def __gt__(self, other):
        return self._cmp_op(other, ops.gt)

    def __ge__(self, other):
        return self._cmp_op(other, ops.ge)

    def __contains__(self, other):
        forced = force_to_smt_sort(other, SmtStr)
        return SmtBool(z3.Contains(self.var, forced))

    def __getitem__(self, i):
        idx_or_pair = process_slice_vs_symbolic_len(
            self.statespace,
            i,
            z3.Length(self.var),
            # At present, Z3's string solver performs poorly with ite()s in indices:
            fork_on_negative_index=True,
        )
        if isinstance(idx_or_pair, tuple):
            (start, stop) = idx_or_pair
            smt_result = z3.Extract(self.var, start, stop - start)
        else:
            smt_result = z3.Extract(self.var, idx_or_pair, 1)
        return SmtStr(smt_result)

    def find(self, substr, start=None, end=None):
        smt_mystr = self.var
        smt_substr = force_to_smt_sort(substr, SmtStr)
        if end is not None:
            end = force_to_smt_sort(end, SmtInt)
            smt_mystr = z3.SubString(smt_mystr, 0, end)
        start = 0 if start is None else force_to_smt_sort(start, SmtInt)
        return SmtInt(z3.IndexOf(smt_mystr, smt_substr, start))

    def rfind(self, substr, start=None, end=None):
        space = self.statespace
        sub = force_to_smt_sort(substr, SmtStr)

        if start is None:
            start = 0
        elif start < 0:
            start += len(self)
        if end is None or end > len(self):
            end = len(self)
        if start > len(self) or end < 0 or start > end:
            return -1

        smt_start = force_to_smt_sort(start, SmtInt)
        smt_end = force_to_smt_sort(end, SmtInt)
        value = z3.SubString(self.var, smt_start, smt_end - smt_start)

        if space.smt_fork(z3.Contains(value, sub)):
            match_index = z3.Int(f"match_index_{space.uniq()}")
            last_match = z3.SubString(value, match_index, z3.Length(sub))
            index_remaining = match_index + 1
            remaining = z3.SubString(
                value, index_remaining, z3.Length(value) - index_remaining
            )
            space.add(
                z3.And(
                    z3.Contains(last_match, sub), z3.Not(z3.Contains(remaining, sub))
                )
            )
            return SmtInt(match_index + start)
        else:
            return -1

    def index(self, substr, start=None, end=None):
        idx = self.find(substr, start, end)
        if idx == -1:
            raise ValueError
        return idx

    def startswith(self, substr):
        smt_substr = force_to_smt_sort(substr, SmtStr)
        return SmtBool(z3.PrefixOf(smt_substr, self.var))

    def endswith(self, substr):
        smt_substr = force_to_smt_sort(substr, SmtStr)
        return SmtBool(z3.SuffixOf(smt_substr, self.var))

    def split(self, sep: Optional[str] = None, maxsplit: int = -1):
        if sep is None:
            return self.__str__().split(sep=sep, maxsplit=maxsplit)
        smt_sep = force_to_smt_sort(sep, SmtStr)
        if not isinstance(maxsplit, Integral):
            raise TypeError
        if maxsplit == 0:
            return [self]
        first_occurance = SmtInt(z3.IndexOf(self.var, smt_sep, 0))
        if first_occurance == -1:
            return [self]
        ret = [self[: cast(int, first_occurance)]]
        new_maxsplit = -1 if maxsplit == -1 else maxsplit - 1
        ret.extend(self[first_occurance + 1 :].split(sep=sep, maxsplit=new_maxsplit))
        return ret

    def rsplit(self, sep: Optional[str] = None, maxsplit: int = -1):
        if sep is None:
            return self.__str__().rsplit(sep=sep, maxsplit=maxsplit)
        smt_sep = force_to_smt_sort(sep, SmtStr)
        if not isinstance(maxsplit, Integral):
            raise TypeError
        if maxsplit == 0:
            return [self]
        last_occurence = self.rfind(sep)
        if last_occurence == -1:
            return [self]
        new_maxsplit = -1 if maxsplit < 0 else maxsplit - 1
        remaining = self[: cast(int, last_occurence)]
        ret = self[:last_occurence].rsplit(sep, new_maxsplit)
        index_after = len(sep) + last_occurence
        ret.append(self[index_after:])
        return ret


_CACHED_TYPE_ENUMS: Dict[FrozenSet[type], z3.SortRef] = {}


_PYTYPE_TO_WRAPPER_TYPE = {
    bool: (SmtBool,),
    int: (SmtInt,),
    float: (SmtFloat,),
    str: (SmtStr,),
    type: (SmtType,),
}

# Type ignore pending https://github.com/python/mypy/issues/6864
_PYTYPE_TO_WRAPPER_TYPE[collections.abc.Callable] = (SmtCallable,)  # type:ignore

_WRAPPER_TYPE_TO_PYTYPE = dict(
    (v, k) for (k, vs) in _PYTYPE_TO_WRAPPER_TYPE.items() for v in vs
)


#
# Proxy making helpers
#


def make_union_choice(creator, *pytypes):
    for typ in pytypes[:-1]:
        if creator.space.smt_fork(desc="choose_" + name_of_type(typ)):
            return creator(typ)
    return creator(pytypes[-1])


def make_optional_smt(smt_type):
    def make(creator, *type_args):
        space = context_statespace()
        varname, pytype = creator.varname, creator.pytype
        ret = smt_type(creator.varname, pytype)

        premature_stats, symbolic_stats = space.stats_lookahead()
        bad_iters = (
            # SMT unknowns, unsupported, etc:
            symbolic_stats[VerificationStatus.UNKNOWN]
            +
            # Or, we ended up realizing this var anyway:
            symbolic_stats[f"realize_{varname}"]
        )
        bad_pct = bad_iters / (symbolic_stats.iterations + 10)
        symbolic_probability = 0.98 - (bad_pct * 0.8)
        if space.fork_parallel(
            false_probability=symbolic_probability, desc=f"premature realize {varname}"
        ):
            debug("Prematurely realizing", pytype, "value")
            ret = realize(ret)
        return ret

    return make


def make_dictionary(creator, key_type=Any, value_type=Any):
    space, varname = creator.space, creator.varname
    if pytype_uses_heap(key_type):
        kv = proxy_for_type(
            List[Tuple[key_type, value_type]],  # type: ignore
            varname + "items",
            allow_subtypes=False,
        )
        orig_kv = kv[:]

        def ensure_keys_are_unique() -> bool:
            return len(set(k for k, _ in orig_kv)) == len(orig_kv)

        space.defer_assumption("dict keys are unique", ensure_keys_are_unique)
        return SimpleDict(kv)
    return ShellMutableMap(SmtDict(varname, creator.pytype))


def make_tuple(creator, *type_args):
    if not type_args:
        type_args = (object, ...)  # type: ignore
    if len(type_args) == 2 and type_args[1] == ...:
        return SmtUniformTuple(creator.varname, creator.pytype)
    else:
        return tuple(
            proxy_for_type(t, creator.varname + "_at_" + str(idx), allow_subtypes=True)
            for (idx, t) in enumerate(type_args)
        )


def make_set(creator, *type_args):
    if type_args:
        return ShellMutableSet(creator(FrozenSet.__getitem__(*type_args)))
    else:
        return ShellMutableSet(creator(FrozenSet))


def make_raiser(exc, *a) -> Callable:
    def do_raise(*ra, **rkw) -> NoReturn:
        raise exc(*a)

    return do_raise


#
# Monkey Patches
#

_T = TypeVar("_T")
_VT = TypeVar("_VT")


class _BuiltinsCopy:
    pass


_TRUE_BUILTINS: Any = _BuiltinsCopy()
_TRUE_BUILTINS.__dict__.update(orig_builtins.__dict__)


def fork_on_useful_attr_names(obj: object, name: SmtStr) -> None:
    # This function appears to do nothing at all!
    # It exists to force a symbolic string into useful candidate states.
    with context_statespace().framework():
        obj = realize(obj)
        for key in reversed(dir(obj)):
            # We use reverse() above to handle __dunder__ methods last.
            if name == key:
                return


_ascii = with_realized_args(orig_builtins.ascii)

_bin = with_realized_args(orig_builtins.bin)

_callable = with_realized_args(orig_builtins.callable)

_orig_eval = orig_builtins.eval


def _eval(expr: str, _globals=None, _locals=None) -> object:
    calling_frame = sys._getframe(2)
    _globals = calling_frame.f_globals if _globals is None else realize(_globals)
    _locals = calling_frame.f_locals if _locals is None else realize(_locals)
    return _orig_eval(realize(expr), _globals, _locals)


_orig_format = orig_builtins.format


def _format(obj: object, format_spec: str = "") -> str:
    if isinstance(obj, SmtBackedValue):
        obj = realize(obj)
    if type(format_spec) is SmtStr:
        format_spec = realize(format_spec)
    return _orig_format(obj, format_spec)


_orig_getattr = orig_builtins.getattr


def _getattr(obj: object, name: str, default=_MISSING) -> object:
    if type(name) is SmtStr:
        fork_on_useful_attr_names(obj, name)  # type:ignore
        name = realize(name)
    if default is _MISSING:
        return _orig_getattr(obj, name)
    else:
        return _orig_getattr(obj, name, default)


_orig_hasattr = orig_builtins.hasattr


def _hasattr(obj: object, name: str) -> bool:
    if type(name) is SmtStr:
        fork_on_useful_attr_names(obj, name)  # type:ignore
        name = realize(name)
    return _orig_hasattr(obj, name)


def _hash(obj: Hashable) -> int:
    """
    post[]: smt_and(-2**63 <= _, _ < 2**63)
    """
    # Skip the built-in hash if possible, because it requires the output
    # to be a native int:
    if is_hashable(obj):
        # You might think we'd say "return obj.__hash__()" here, but we need some
        # special gymnastics to avoid "metaclass confusion".
        # See: https://docs.python.org/3/reference/datamodel.html#special-method-lookup
        return type(obj).__hash__(obj)
    else:
        return _TRUE_BUILTINS.hash(obj)


# Trick the system into believing that symbolic values are
# native types.
def _issubclass(subclass, superclass):
    if not isinstance(subclass, type):
        raise TypeError("issubclass() arg 1 must be a class")
    subclass_is_special = hasattr(subclass, "_is_subclass_of_")
    if not subclass_is_special:
        # We could also check superclass(es) for a special method, but
        # the native function won't return True in those cases anyway.
        try:
            ret = _TRUE_BUILTINS.issubclass(subclass, superclass)
            if ret:
                return True
        except TypeError:
            pass
    if type(superclass) is tuple:
        for cur_super in superclass:
            if _issubclass(subclass, cur_super):
                return True
        return False
    if not isinstance(superclass, type):
        raise TypeError("issubclass() arg 2 must be a class or tuple of classes")
    if hasattr(superclass, "_is_superclass_of_"):
        method = superclass._is_superclass_of_
        if (
            method(subclass)
            if hasattr(method, "__self__")
            else method(subclass, superclass)
        ):
            return True
    if subclass_is_special:
        method = subclass._is_subclass_of_
        if (
            method(superclass)
            if hasattr(method, "__self__")
            else method(subclass, superclass)
        ):
            return True
    return False


def _isinstance(obj, types):
    try:
        ret = _TRUE_BUILTINS.isinstance(obj, types)
        if ret:
            return True
    except TypeError:
        pass
    if hasattr(obj, "python_type"):
        obj_type = obj.python_type
        if hasattr(obj_type, "__origin__"):
            obj_type = obj_type.__origin__
    else:
        obj_type = type(obj)
    return issubclass(obj_type, types)


# CPython's len() forces the return value to be a native integer.
# Avoid that requirement by making it only call __len__().
def _len(l):
    return l.__len__() if hasattr(l, "__len__") else [x for x in l].__len__()


def _max(*values, key=lambda x: x, default=_MISSING):
    if len(values) <= 1:
        if not values:
            raise TypeError("expected 1 argument, got 0")
        if not is_iterable(values[0]):
            raise TypeError("object is not iterable")
        values = values[0]
    return _max_iter(values, key=key, default=default)


def _max_iter(
    values: Iterable[_T],
    *,
    key: Callable = lambda x: x,
    default: Union[_Missing, _VT] = _MISSING,
) -> _T:
    """
    pre: bool(values) or default is not _MISSING
    post[]::
      (_ in values) if default is _MISSING else True
      ((_ in values) or (_ is default)) if default is not _MISSING else True
    """
    kw = {} if default is _MISSING else {"default": default}
    return _TRUE_BUILTINS.max(values, key=key, **kw)


def _min(*values, key=lambda x: x, default=_MISSING):
    if len(values) <= 1:
        if not values:
            raise TypeError("expected 1 argument, got 0")
        if not is_iterable(values[0]):
            raise TypeError("object is not iterable")
        values = values[0]
    return _min_iter(values, key=key, default=default)


def _min_iter(
    values: Iterable[_T],
    *,
    key: Callable = lambda x: x,
    default: Union[_Missing, _VT] = _MISSING,
) -> _T:
    """
    pre: bool(values) or default is not _MISSING
    post[]::
      (_ in values) if default is _MISSING else True
      ((_ in values) or (_ is default)) if default is not _MISSING else True
    """
    kw = {} if default is _MISSING else {"default": default}
    return _TRUE_BUILTINS.min(values, key=key, **kw)


_orig_ord = orig_builtins.ord


def _ord(x: str) -> int:
    return _orig_ord(realize(x))


_orig_pow = orig_builtins.pow


def _pow(base, exp, mod=None):
    return _orig_pow(realize(base), realize(exp), realize(mod))


# TODO consider what to do
# def print(*a: object, **kw: Any) -> None:
#    '''
#    post: True
#    '''
#    _TRUE_BUILTINS.print(*a, **kw)


def _repr(arg: object) -> str:
    """
    post[]: True
    """
    # Skip the built-in repr if possible, because it requires the output
    # to be a native string:
    if hasattr(arg, "__repr__"):
        # You might think we'd say "return obj.__repr__()" here, but we need some
        # special gymnastics to avoid "metaclass confusion".
        # See: https://docs.python.org/3/reference/datamodel.html#special-method-lookup
        return type(arg).__repr__(arg)
    else:
        return _TRUE_BUILTINS.repr(arg)


_orig_setattr = orig_builtins.setattr


def _setattr(obj: object, name: str, value: object) -> None:
    # TODO: we could do symbolic stuff like getattr does here!
    if isinstance(obj, SmtBackedValue):
        obj = realize(obj)
    if type(name) is SmtStr:
        name = realize(name)
    return _orig_setattr(obj, name, value)


# TODO: is this important? Feels like the builtin might do the same?
def _sorted(l, key=None, reverse=False):
    if not is_iterable(l):
        raise TypeError("object is not iterable")
    ret = list(l.__iter__())
    ret.sort(key=key, reverse=realize(reverse))
    return ret


# TODO: consider patching super() so that masquerade'd classes do the right
# thing.

# TODO: consider what to do here
# def sum(i: Iterable[_T]) -> Union[_T, int]:
#    '''
#    post[]: _ == 0 or len(i) > 0
#    '''
#    return _TRUE_BUILTINS.sum(i)


#
# Patches on builtin classes
#


_orig_list_index = orig_builtins.list.index


def _list_index(self, value, start=0, stop=9223372036854775807):
    return _orig_list_index(self, value, realize(start), realize(stop))


def _list_repr(self):
    # A pure python implementation so that we get the monkey-patched
    # version of repr when appropriate:
    return "[" + ", ".join(repr(x) for x in self) + "]"


def _str_join(self, itr) -> str:
    # An obviously slow implementation, but describable in terms of
    # string concatenation, which we can do symbolically.
    # Realizes the length of the list asrgument but not the contents.
    result = ""
    for idx, item in enumerate(itr):
        if idx > 0:
            result = result + self
        result = result + item
    return result


#
# Registrations
#


def make_registrations():

    register_type(Union, make_union_choice)

    if sys.version_info >= (3, 8):
        register_type(Final, lambda p, t: p(t))

    # Types modeled in the SMT solver:

    register_type(NoneType, lambda *a: None)
    register_type(bool, make_optional_smt(SmtBool))
    register_type(int, make_optional_smt(SmtInt))
    register_type(float, make_optional_smt(SmtFloat))
    register_type(str, make_optional_smt(SmtStr))
    register_type(list, make_optional_smt(SmtList))
    register_type(dict, make_dictionary)
    register_type(tuple, make_tuple)
    register_type(set, make_set)
    register_type(frozenset, make_optional_smt(SmtFrozenSet))
    register_type(type, make_optional_smt(SmtType))
    register_type(collections.abc.Callable, make_optional_smt(SmtCallable))

    # Most types are not directly modeled in the solver, rather they are built
    # on top of the modeled types. Such types are enumerated here:

    register_type(object, lambda p: SmtObject(p.varname, p.pytype))
    register_type(complex, lambda p: complex(p(float, "_real"), p(float, "_imag")))
    register_type(
        slice,
        lambda p: slice(
            p(Optional[int], "_start"),
            p(Optional[int], "_stop"),
            p(Optional[int], "_step"),
        ),
    )
    register_type(NoReturn, make_raiser(IgnoreAttempt, "Attempted to short circuit a NoReturn function"))  # type: ignore

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
    register_type(
        typing.IO, lambda p, t=Any: p(BinaryIO) if t == "bytes" else p(TextIO)
    )
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

    register_patch(orig_builtins, _ascii, "ascii")
    register_patch(orig_builtins, _bin, "bin")
    register_patch(orig_builtins, _callable, "callable")
    register_patch(orig_builtins, _eval, "eval")
    register_patch(orig_builtins, _format, "format")
    register_patch(orig_builtins, _getattr, "getattr")
    register_patch(orig_builtins, _hasattr, "hasattr")
    register_patch(orig_builtins, _hash, "hash")
    register_patch(orig_builtins, _isinstance, "isinstance")
    register_patch(orig_builtins, _issubclass, "issubclass")
    register_patch(orig_builtins, _len, "len")
    register_patch(orig_builtins, _max, "max")
    register_patch(orig_builtins, _min, "min")
    register_patch(orig_builtins, _ord, "ord")
    register_patch(orig_builtins, _pow, "pow")
    register_patch(orig_builtins, _repr, "repr")
    register_patch(orig_builtins, _setattr, "setattr")
    register_patch(orig_builtins, _sorted, "sorted")

    # Patches on str
    for name in [
        "center",
        "count",
        "encode",
        "endswith",
        "expandtabs",
        "find",
        "format",  # TODO: shallow realization likely isn't sufficient
        "format_map",
        "index",
        "ljust",
        "lstrip",
        "partition",
        "replace",
        "rfind",
        "rindex",
        "rjust",
        "rpartition",
        "rsplit",
        "rstrip",
        "split",
        "splitlines",
        "startswith",
        "strip",
        "translate",
        "zfill",
    ]:
        orig_impl = getattr(orig_builtins.str, name)
        register_patch(orig_builtins.str, with_realized_args(orig_impl), name)

    orig_join = orig_builtins.str.join
    register_patch(orig_builtins.str, _str_join, "join")

    # TODO: override str.__new__ to make symbolic strings

    # Patches on list
    register_patch(orig_builtins.list, _list_index, "index")
    # TODO: forbiddenfruit can't patch __repr__ yet:
    # register_patch(orig_builtins.list, _list_repr, '__repr__')

    # Patches on int
    register_patch(
        orig_builtins.int,
        with_realized_args(orig_builtins.int.from_bytes),
        "from_bytes",
    )

    # Patches on float
    register_patch(
        orig_builtins.float, with_realized_args(orig_builtins.float.fromhex), "fromhex"
    )

    setup_binops()
