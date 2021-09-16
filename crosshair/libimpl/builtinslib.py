import builtins as orig_builtins
import collections
import copy
from dataclasses import dataclass
import enum
from functools import total_ordering
import io
import math
from numbers import Number
from numbers import Real
from numbers import Integral
import operator as ops
import re
import typing
from typing import *
import sys

from crosshair.abcstring import AbcString
from crosshair.core import deep_realize
from crosshair.core import inside_realization
from crosshair.core import register_patch
from crosshair.core import register_type
from crosshair.core import realize
from crosshair.core import proxy_for_type
from crosshair.core import python_type
from crosshair.core import normalize_pytype
from crosshair.core import choose_type
from crosshair.core import type_arg_of
from crosshair.core import type_args_of
from crosshair.core import with_realized_args
from crosshair.core import CrossHairValue
from crosshair.core import SymbolicFactory
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
from crosshair.tracers import is_tracing
from crosshair.tracers import NoTracing
from crosshair.tracers import ResumedTracing
from crosshair.type_repo import PYTYPE_SORT
from crosshair.util import debug
from crosshair.util import is_iterable
from crosshair.util import is_hashable
from crosshair.util import name_of_type
from crosshair.util import memo
from crosshair.util import smtlib_typename
from crosshair.util import CrosshairInternal
from crosshair.util import CrosshairUnsupported
from crosshair.util import IgnoreAttempt

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
    if isinstance(a, SymbolicBool) and isinstance(b, SymbolicBool):
        ret = SymbolicBool(z3.And(a.var, b.var))
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
    Foces values of unknown type (SymbolicObject) into a typed (but possibly still symbolic) value.
    """
    while type(val) is SymbolicObject:
        val = cast(SymbolicObject, val)._wrapped()
    return val


_SMT_INT_SORT = z3.IntSort()
_SMT_BOOL_SORT = z3.BoolSort()
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


SymbolicGenerator = Callable[[Union[str, z3.ExprRef], type], object]


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
    if isinstance(val, SymbolicValue):
        return val.var
    return val


class SymbolicValue(CrossHairValue):
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
            result = copy.deepcopy(self.__ch_realize__())
        else:
            result = copy.copy(self)
            result.snapshot = self.statespace.current_snapshot()
        memo[id(self)] = result
        return result

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


class AtomicSymbolicValue(SymbolicValue):
    def __init_var__(self, typ, varname):
        if is_tracing():
            raise CrosshairInternal("Tracing while creating symbolic")
        z3type = self.__class__._ch_smt_sort()
        return z3.Const(varname, z3type)

    def __ch_is_deeply_immutable__(self) -> bool:
        return True

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
        if is_tracing():
            raise CrosshairInternal("_coerce_to_smt_sort called while tracing")
        input_value = typeable_value(input_value)
        target_pytype = cls._pytype()

        # check the likely cases first
        if isinstance(input_value, cls):
            return input_value.var
        elif isinstance(input_value, target_pytype):
            return cls._smt_promote_literal(input_value)

        # see whether we can safely cast and retry
        if isinstance(input_value, Number) and issubclass(cls, Number):
            if isinstance(input_value, SymbolicValue):
                casting_fn_name = "__" + target_pytype.__name__ + "__"
                converted = getattr(input_value, casting_fn_name)()
                return cls._coerce_to_smt_sort(converted)
            else:  # non-symbolic
                casted = target_pytype(input_value)
                if casted == input_value:
                    return cls._coerce_to_smt_sort(casted)

        return None

    def __eq__(self, other):
        with NoTracing():
            coerced = type(self)._coerce_to_smt_sort(other)
            if coerced is None:
                return False
            return SymbolicBool(self.var == coerced)


_PYTYPE_TO_WRAPPER_TYPE: Dict[
    type, Tuple[Type[AtomicSymbolicValue], ...]
] = {}  # to be populated later
_WRAPPER_TYPE_TO_PYTYPE: Dict[SymbolicGenerator, type] = {}


def crosshair_types_for_python_type(typ: Type) -> Tuple[Type[AtomicSymbolicValue], ...]:
    typ = normalize_pytype(typ)
    origin = origin_of(typ)
    return _PYTYPE_TO_WRAPPER_TYPE.get(origin, ())


def smt_to_ch_value(
    space: StateSpace, snapshot: SnapshotRef, smt_val: z3.ExprRef, pytype: type
) -> object:
    def proxy_generator(typ: Type) -> object:
        return proxy_for_type(typ, smtlib_typename(typ) + "_inheap" + space.uniq())

    if smt_val.sort() == HeapRef:
        return space.find_key_in_heap(smt_val, pytype, proxy_generator, snapshot)
    ch_type = crosshair_types_for_python_type(pytype)
    assert ch_type
    return ch_type[0](smt_val, pytype)


def force_to_smt_sort(
    input_value: Any, desired_ch_type: Type[AtomicSymbolicValue]
) -> z3.ExprRef:
    with NoTracing():
        ret = desired_ch_type._coerce_to_smt_sort(input_value)
        if ret is None:
            raise TypeError(
                f"Could not derive smt type '{desired_ch_type}' from {input_value}:{type(input_value)}"
            )
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


@dataclass
class KindedFloat:
    val: float


class FiniteFloat(KindedFloat):
    pass


class NonFiniteFloat(KindedFloat):
    pass


def numeric_binop(op: BinFn, a: Number, b: Number):
    if not is_tracing():
        raise CrosshairInternal("Numeric operation on symbolic while not tracing")
    with NoTracing():
        return numeric_binop_internal(op, a, b)


def numeric_binop_internal(op: BinFn, a: Number, b: Number):
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
    with ResumedTracing():  # TODO: <-- can we instead selectively resume?
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

                def promotion_forward(o, x, y):
                    x2, y2 = fn(x, y)
                    return numeric_binop(o, x2, y2)

                def promotion_backward(o, x, y):
                    y2, x2 = fn(y, x)
                    return numeric_binop(o, x2, y2)

                _BIN_OPS_SEARCH_ORDER.append((op, a_type, b_type, promotion_forward))
                _BIN_OPS_SEARCH_ORDER.append((op, b_type, a_type, promotion_backward))


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
    return op(x, y)


_ARITHMETIC_AND_COMPARISON_OPS = _ARITHMETIC_OPS.union(_COMPARISON_OPS)
_ALL_OPS = _ARITHMETIC_AND_COMPARISON_OPS.union(_BITWISE_OPS)


def setup_binops():
    # Lower entries take precendence when searching.

    # We check NaN and infitity immediately; not all
    # symbolic floats support these cases.
    def _(a: Real, b: float):
        if math.isfinite(b):
            return (a, FiniteFloat(b))  # type: ignore
        return (a, NonFiniteFloat(b))

    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # Almost all operators involving booleans should upconvert to integers.
    def _(a: SymbolicBool, b: Number):
        return (SymbolicInt(z3.If(a.var, 1, 0)), b)

    setup_promotion(_, _ALL_OPS)

    # Implicitly upconvert symbolic ints to floats.
    def _(a: SymbolicInt, b: Union[float, FiniteFloat, SymbolicFloat, complex]):
        return (SymbolicFloat(z3.ToReal(a.var)), b)

    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # Implicitly upconvert native ints to floats.
    def _(a: int, b: Union[float, FiniteFloat, SymbolicFloat, complex]):
        return (float(a), b)

    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # Implicitly upconvert native bools to ints.
    def _(a: bool, b: Union[SymbolicInt, SymbolicFloat]):
        return (int(a), b)

    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # complex
    def _(op: BinFn, a: SymbolicNumberAble, b: complex):
        return op(complex(a), b)  # type: ignore

    setup_binop(_, _ALL_OPS)

    # float
    def _(op: BinFn, a: SymbolicFloat, b: SymbolicFloat):
        return SymbolicFloat(apply_smt(op, a.var, b.var))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: SymbolicFloat, b: SymbolicFloat):
        return SymbolicBool(apply_smt(op, a.var, b.var))

    setup_binop(_, _COMPARISON_OPS)

    def _(op: BinFn, a: SymbolicFloat, b: FiniteFloat):
        return SymbolicFloat(apply_smt(op, a.var, z3.RealVal(b.val)))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: FiniteFloat, b: SymbolicFloat):
        return SymbolicFloat(apply_smt(op, z3.RealVal(a.val), b.var))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: Union[FiniteFloat, SymbolicFloat], b: NonFiniteFloat):
        if isinstance(a, FiniteFloat):
            comparable_a: Union[float, SymbolicFloat] = a.val
        else:
            comparable_a = a
        # These three cases help cover operations like `a * -inf` which is either
        # positive of negative infinity depending on the sign of `a`.
        if comparable_a > 0:  # type: ignore
            return op(1, b.val)  # type: ignore
        elif comparable_a < 0:
            return op(-1, b.val)  # type: ignore
        else:
            return op(0, b.val)  # type: ignore

    setup_binop(_, _ARITHMETIC_AND_COMPARISON_OPS)

    def _(op: BinFn, a: NonFiniteFloat, b: NonFiniteFloat):
        return op(a.val, b.val)  # type: ignore

    setup_binop(_, _ARITHMETIC_AND_COMPARISON_OPS)

    def _(op: BinFn, a: SymbolicFloat, b: FiniteFloat):
        return SymbolicBool(apply_smt(op, a.var, z3.RealVal(b.val)))

    setup_binop(_, _COMPARISON_OPS)

    # int
    def _(op: BinFn, a: SymbolicInt, b: SymbolicInt):
        return SymbolicInt(apply_smt(op, a.var, b.var))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: SymbolicInt, b: SymbolicInt):
        return SymbolicBool(apply_smt(op, a.var, b.var))

    setup_binop(_, _COMPARISON_OPS)

    def _(op: BinFn, a: SymbolicInt, b: int):
        return SymbolicInt(apply_smt(op, a.var, z3.IntVal(b)))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: int, b: SymbolicInt):
        return SymbolicInt(apply_smt(op, z3.IntVal(a), b.var))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: SymbolicInt, b: int):
        return SymbolicBool(apply_smt(op, a.var, z3.IntVal(b)))

    setup_binop(_, _COMPARISON_OPS)

    def _(op: BinFn, a: Integral, b: Integral):
        # Some bitwise operators require realization presently.
        # TODO: when one side is already realized, we could do something smarter.
        return op(a.__index__(), b.__index__())  # type: ignore

    setup_binop(_, {ops.or_, ops.xor})

    def _(op: BinFn, a: Integral, b: Integral):
        if b < 0:
            raise ValueError("negative shift count")
        b = realize(b)  # Symbolic exponents defeat the solver
        if op == ops.lshift:
            return a * (2 ** b)
        else:
            return a // (2 ** b)

    setup_binop(_, {ops.lshift, ops.rshift})

    _AND_MASKS_TO_MOD = {
        # It's common to use & to mask low bits. We can avoid realization by converting
        # these situations into mod operations.
        0x01: 2,
        0x03: 4,
        0x07: 8,
        0x0F: 16,
        0x1F: 32,
        0x3F: 64,
        0x7F: 128,
        0xFF: 256,
    }

    def _(op: BinFn, a: Integral, b: Integral):
        with NoTracing():
            if isinstance(b, SymbolicInt):
                # Have `a` be symbolic, if possible
                (a, b) = (b, a)

            # Check whether we can interpret the mask as a mod operation:
            b = realize(b)
            if isinstance(a, SymbolicInt) and context_statespace().smt_fork(
                a.var >= 0, favor_true=True
            ):
                if b == 0:
                    return 0
                mask_mod = _AND_MASKS_TO_MOD.get(b)
                if mask_mod:
                    return SymbolicInt(a.var % mask_mod)

            # Fall back to full realization
            return op(realize(a), b)  # type: ignore

    setup_binop(_, {ops.and_})

    # TODO: is this necessary still?
    def _(
        op: BinFn, a: Integral, b: Integral
    ):  # Floor division over ints requires realization, at present
        return op(a.__index__(), b.__index__())  # type: ignore

    setup_binop(_, {ops.truediv})

    def _(a: SymbolicInt, b: Number):  # Division over ints must produce float
        return (a.__float__(), b)

    setup_promotion(_, {ops.truediv})

    # bool
    def _(op: BinFn, a: SymbolicBool, b: SymbolicBool):
        return SymbolicBool(apply_smt(op, a.var, b.var))

    setup_binop(_, {ops.eq, ops.ne})


#
#  END new numbers
#


class SymbolicNumberAble(SymbolicValue, Real):
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


class SymbolicIntable(SymbolicNumberAble, Integral):
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
            if self <= 0:
                return ""
            with NoTracing():
                # Create a symbolic string that regex-matches as a repetition.
                other = realize(other)  # z3 does not handle symbolic regexes
                space = self.statespace
                result = SymbolicStr(f"{self.var}_str{space.uniq()}")
                space.add(z3.InRe(result.var, z3.Star(z3.Re(other))))
                space.add(z3.Length(result.var) == len(other) * self.var)
                return result
        return numeric_binop(ops.mul, self, other)

    __rmul__ = __mul__


class SymbolicBool(SymbolicIntable, AtomicSymbolicValue):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = bool):
        assert typ == bool
        SymbolicValue.__init__(self, smtvar, typ)

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return _SMT_BOOL_SORT

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
        return SymbolicInt(z3.If(self.var, -1, 0))

    def __repr__(self):
        return self.__bool__().__repr__()

    def __hash__(self):
        return self.__bool__().__hash__()

    def __index__(self):
        return SymbolicInt(z3.If(self.var, 1, 0))

    def __bool__(self):
        with NoTracing():
            return self.statespace.choose_possible(self.var)

    def __int__(self):
        return SymbolicInt(z3.If(self.var, 1, 0))

    def __float__(self):
        return SymbolicFloat(smt_bool_to_float(self.var))

    def __complex__(self):
        return complex(self.__float__())

    def __round__(self, ndigits=None):
        # This could be smarter, but nobody rounds a bool right?:
        return round(realize(self), realize(ndigits))


class SymbolicInt(SymbolicIntable, AtomicSymbolicValue):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = int):
        assert typ == int
        SymbolicIntable.__init__(self, smtvar, typ)

    # Now that type() on symbolic ints returns `int`, do we need these classmethods?:

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return _SMT_INT_SORT

    @classmethod
    def _pytype(cls) -> Type:
        return int

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, int):
            # Additional __int__() cast in case literal is a bool:
            literal = literal.__int__()
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
        # return SymbolicStr(z3.IntToStr(self.var))

    def __hash__(self):
        return self.__index__().__hash__()

    def __float__(self):
        return SymbolicFloat(smt_int_to_float(self.var))

    def __complex__(self):
        return complex(self.__float__())

    def __index__(self):
        with NoTracing():
            space = context_statespace()
            if space.smt_fork(self.var == 0):
                return 0
            ret = space.find_model_value(self.var)
            assert (
                type(ret) is int
            ), f"SymbolicInt with wrong SMT var type ({type(ret)})"
            return ret

    def __bool__(self):
        return SymbolicBool(self.var != 0).__bool__()

    def __int__(self):
        return self.__index__()

    def __round__(self, ndigits=None):
        if ndigits is None or ndigits >= 0:
            return self
        if self < 0:
            return -((-self).__round__(ndigits))
        with NoTracing():
            space = context_statespace()
            var = self.var
            factor = 10 ** (-realize(ndigits))
            half = factor // 2
            on_border = (var + half) % factor == 0
            if space.smt_fork(on_border):
                floor = var - (var % factor)
                if space.smt_fork((var / factor) % 2 == 0):
                    smt_ret = floor
                else:
                    smt_ret = floor + factor
            else:
                var = var + half
                smt_ret = var - (var % factor)
            return SymbolicInt(smt_ret)

    def bit_length(self) -> "SymbolicInt":
        abs_self = -self if self < 0 else self
        if abs_self >= 256:
            return (abs_self // 256).bit_length() + 8
        with NoTracing():
            val = abs_self.var
            # fmt: off
            return SymbolicInt(
                z3.If(val == 0, 0,
                z3.If(val < 2, 1,
                z3.If(val < 4, 2,
                z3.If(val < 8, 3,
                z3.If(val < 16, 4,
                z3.If(val < 32, 5,
                z3.If(val < 64, 6,
                z3.If(val < 128, 7, 8)))))))))
            # fmt: on

    def to_bytes(self, length, byteorder, *, signed=False):
        if not isinstance(length, int):
            raise TypeError
        if not isinstance(byteorder, str):
            raise TypeError
        if not isinstance(signed, bool):
            raise TypeError
        length = realize(length)
        if signed:
            half = (256 ** length) >> 1
            if self < -half or self >= half:
                raise OverflowError
            if self < 0:
                self = 256 ** length + self
        else:
            if self < 0 or self >= 256 ** length:
                raise OverflowError
        intarray = [
            SymbolicInt((self.var / (2 ** (i * 8))) % 256) for i in range(length)
        ]
        if byteorder == "big":
            intarray.reverse()
        return SymbolicBytes(intarray)

    def as_integer_ratio(self) -> Tuple["SymbolicInt", int]:
        return (self, 1)


_Z3_ONE_HALF = z3.RealVal("1/2")


class SymbolicFloat(SymbolicNumberAble, AtomicSymbolicValue):
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
        assert typ is float, f"SymbolicFloat with unexpected python type ({type(typ)})"
        SymbolicValue.__init__(self, smtvar, typ)

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return z3.RealSort()

    @classmethod
    def _pytype(cls) -> Type:
        return float

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, float):
            return z3.RealVal(literal)
        return None

    def __ch_realize__(self) -> object:
        return self.statespace.find_model_value(self.var).__float__()  # type: ignore

    def __repr__(self):
        return self.statespace.find_model_value(self.var).__repr__()

    def __hash__(self):
        return self.statespace.find_model_value(self.var).__hash__()

    def __bool__(self):
        return SymbolicBool(self.var != 0).__bool__()

    def __int__(self):
        var = self.var
        return SymbolicInt(z3.If(var >= 0, z3.ToInt(var), -z3.ToInt(-var)))

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
        with NoTracing():
            var, floor, nearest = (
                self.var,
                z3.ToInt(self.var),
                z3.ToInt(self.var + _Z3_ONE_HALF),
            )
            ret = SymbolicInt(
                z3.If(
                    var != floor + _Z3_ONE_HALF,
                    nearest,
                    z3.If(floor % 2 == 0, floor, floor + 1),
                )
            )
            context_statespace().defer_assumption(
                # Float representation can thwart the rounding behavior;
                # perform an extra check after-the-fact.
                "float rounds as expected",
                lambda: realize(ret) == round(realize(self), realize(ndigits)),
            )
            return ret

    def __floor__(self):
        return SymbolicInt(z3.ToInt(self.var))

    def __ceil__(self):
        var, floor = self.var, z3.ToInt(self.var)
        return SymbolicInt(z3.If(var == floor, floor, floor + 1))

    def __mod__(self, other):
        return realize(self) % realize(
            other
        )  # TODO: z3 does not support modulo on reals

    def __trunc__(self):
        var, floor = self.var, z3.ToInt(self.var)
        return SymbolicInt(z3.If(var >= 0, floor, floor + 1))

    def as_integer_ratio(self) -> Tuple[Integral, Integral]:
        with NoTracing():
            space = context_statespace()
            numerator = SymbolicInt("numerator" + space.uniq())
            denominator = SymbolicInt("denominator" + space.uniq())
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

    def is_integer(self) -> SymbolicBool:
        return SymbolicBool(z3.IsInt(self.var))

    def hex(self) -> str:
        return realize(self).hex()


class SymbolicDictOrSet(SymbolicValue):
    """
    TODO: Ordering is a challenging issue here.
    Modern pythons have in-order iteration for dictionaries but not sets.
    """

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        self.key_pytype = normalize_pytype(type_arg_of(typ, 0))
        ch_types = crosshair_types_for_python_type(self.key_pytype)
        if ch_types:
            self.ch_key_type: Optional[Type[AtomicSymbolicValue]] = ch_types[0]
            self.smt_key_sort = self.ch_key_type._ch_smt_sort()
        else:
            self.ch_key_type = None
            self.smt_key_sort = HeapRef
        SymbolicValue.__init__(self, smtvar, typ)
        self.statespace.add(self._len() >= 0)

    def __ch_realize__(self):
        return origin_of(self.python_type)(self)

    def _arr(self):
        return self.var[0]

    def _len(self):
        return self.var[1]

    def __len__(self):
        return SymbolicInt(self._len())

    def __bool__(self):
        return SymbolicBool(self._len() != 0).__bool__()


# TODO: rename to SymbolicImmutableMap (ShellMutableMap is the real symbolic `dict` class)
class SymbolicDict(SymbolicDictOrSet, collections.abc.Mapping):
    """ An immutable symbolic dictionary. """

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        space = context_statespace()
        self.val_pytype = normalize_pytype(type_arg_of(typ, 1))
        val_ch_types = crosshair_types_for_python_type(self.val_pytype)
        if val_ch_types:
            self.ch_val_type: Optional[Type[AtomicSymbolicValue]] = val_ch_types[0]
            self.smt_val_sort = self.ch_val_type._ch_smt_sort()
        else:
            self.ch_val_type = None
            self.smt_val_sort = HeapRef
        SymbolicDictOrSet.__init__(self, smtvar, typ)
        arr_var = self._arr()
        len_var = self._len()
        self.val_missing_checker = arr_var.sort().range().recognizer(0)
        self.val_missing_constructor = arr_var.sort().range().constructor(0)
        self.val_constructor = arr_var.sort().range().constructor(1)
        self.val_accessor = arr_var.sort().range().accessor(1, 0)
        self.empty = z3.K(arr_var.sort().domain(), self.val_missing_constructor())
        self._iter_cache: List[z3.Const] = []
        space.add((arr_var == self.empty) == (len_var == 0))

        def dict_can_be_iterated():
            list(self.__iter__())
            return True

        space.defer_assumption(
            "dict iteration is consistent with items", dict_can_be_iterated
        )

    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        arr_smt_sort = z3.ArraySort(
            self.smt_key_sort, possibly_missing_sort(self.smt_val_sort)
        )
        return (
            z3.Const(varname + "_map" + self.statespace.uniq(), arr_smt_sort),
            z3.Const(varname + "_len" + self.statespace.uniq(), _SMT_INT_SORT),
        )

    def __eq__(self, other):
        (self_arr, self_len) = self.var
        has_heapref = is_heapref_sort(self.var[0].sort().domain()) or is_heapref_sort(
            self.var[0].sort().range()
        )
        if not has_heapref:
            if isinstance(other, SymbolicDict):
                (other_arr, other_len) = other.var
                return SymbolicBool(
                    z3.And(self_len == other_len, self_arr == other_arr)
                )
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

    # TODO: __contains__ could be implemented without any path forks

    def __getitem__(self, k):
        with NoTracing():
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
            if SymbolicBool(is_missing).__bool__():
                raise KeyError(k)
            if SymbolicBool(self._len() == 0).__bool__():
                raise IgnoreAttempt("SymbolicDict in inconsistent state")
            return smt_to_ch_value(
                self.statespace,
                self.snapshot,
                self.val_accessor(possibly_missing),
                self.val_pytype,
            )

    def __reversed__(self):
        return reversed(list(self))

    def __iter__(self):
        with NoTracing():
            arr_var, len_var = self.var
            iter_cache = self._iter_cache
            space = self.statespace
            idx = 0
            arr_sort = self._arr().sort()
            is_missing = self.val_missing_checker
            while SymbolicBool(idx < len_var).__bool__():
                if not space.choose_possible(arr_var != self.empty, favor_true=True):
                    raise IgnoreAttempt("SymbolicDict in inconsistent state")
                k = z3.Const("k" + str(idx) + space.uniq(), arr_sort.domain())
                v = z3.Const(
                    "v" + str(idx) + space.uniq(), self.val_constructor.domain(0)
                )
                remaining = z3.Const("remaining" + str(idx) + space.uniq(), arr_sort)
                space.add(arr_var == z3.Store(remaining, k, self.val_constructor(v)))
                space.add(is_missing(z3.Select(remaining, k)))

                # TODO: is this true now? it's immutable these days?
                # our iter_cache might contain old keys that were removed;
                # check to make sure the current key is still present:
                while idx < len(iter_cache):
                    still_present = z3.Not(
                        is_missing(z3.Select(arr_var, iter_cache[idx]))
                    )
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
                with ResumedTracing():
                    yield smt_to_ch_value(space, self.snapshot, k, self.key_pytype)
                arr_var = remaining
            # In this conditional, we reconcile the parallel symbolic variables for length
            # and contents:
            if not space.choose_possible(arr_var == self.empty, favor_true=True):
                raise IgnoreAttempt("SymbolicDict in inconsistent state")

    def copy(self):
        return SymbolicDict(self.var, self.python_type)

    # TODO: investigate this approach for type masquerading:
    # @property
    # def __class__(self):
    #    return dict


class SymbolicSet(SymbolicDictOrSet, collections.abc.Set):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        SymbolicDictOrSet.__init__(self, smtvar, typ)
        self._iter_cache: List[z3.Const] = []
        self.empty = z3.K(self._arr().sort().domain(), False)
        self.statespace.add((self._arr() == self.empty) == (self._len() == 0))

    def __eq__(self, other):
        (self_arr, self_len) = self.var
        if isinstance(other, SymbolicSet):
            (other_arr, other_len) = other.var
            if other_arr.sort() == self_arr.sort():
                # TODO: this is wrong for HeapRef sets (which could customize __eq__)
                return SymbolicBool(
                    z3.And(self_len == other_len, self_arr == other_arr)
                )
        if not isinstance(other, (set, frozenset, SymbolicSet, collections.abc.Set)):
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
                z3.ArraySort(self.smt_key_sort, _SMT_BOOL_SORT),
            ),
            z3.Const(varname + "_len" + self.statespace.uniq(), _SMT_INT_SORT),
        )

    def __contains__(self, key):
        with NoTracing():
            if getattr(key, "__hash__", None) is None:
                raise TypeError("unhashable type")
            if self.ch_key_type:
                k = self.ch_key_type._coerce_to_smt_sort(key)
            else:
                k = None
            if k is not None:
                present = self._arr()[k]
                return SymbolicBool(present)
        # Fall back to standard equality and iteration
        for self_item in self:
            if self_item == key:
                return True
        return False

    def __iter__(self):
        with NoTracing():
            arr_var, len_var = self.var
            iter_cache = self._iter_cache
            space = self.statespace
            idx = 0
            arr_sort = self._arr().sort()
            keys_on_heap = is_heapref_sort(arr_sort.domain())
            already_yielded = []
            while SymbolicBool(idx < len_var).__bool__():
                if not space.choose_possible(arr_var != self.empty, favor_true=True):
                    raise IgnoreAttempt("SymbolicSet in inconsistent state")
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
                with ResumedTracing():
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
            if not self.statespace.choose_possible(
                arr_var == self.empty, favor_true=True
            ):
                raise IgnoreAttempt("SymbolicSet in inconsistent state")

    def _set_op(self, attr, other):
        # We need to check the type of other here, because builtin sets
        # do not accept iterable args (but the abc Set does)
        if isinstance(other, collections.abc.Set):
            return getattr(collections.abc.Set, attr)(self, other)
        else:
            raise TypeError

    # Hardwire some operations into abc methods
    # (SymbolicValue defaults these operations into
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


class SymbolicFrozenSet(SymbolicSet):
    def __repr__(self):
        return frozenset(self).__repr__()

    def __hash__(self):
        return frozenset(self).__hash__()

    @classmethod
    def _from_iterable(cls, it):
        # overrides collections.abc.Set's version
        return frozenset(it)


def flip_slice_vs_symbolic_len(
    space: StateSpace,
    i: Union[int, slice],
    smt_len: z3.ExprRef,
) -> Union[z3.ExprRef, Tuple[z3.ExprRef, z3.ExprRef]]:
    if is_tracing():
        raise CrosshairInternal("index math while tracing")

    def normalize_symbolic_index(idx) -> z3.ExprRef:
        if type(idx) is int:
            return z3.IntVal(idx) if idx >= 0 else (smt_len + z3.IntVal(idx))
        else:
            smt_idx = SymbolicInt._coerce_to_smt_sort(idx)
            if space.smt_fork(smt_idx >= 0):  # type: ignore
                return smt_idx
            else:
                return smt_len + smt_idx

    if isinstance(i, (int, SymbolicInt)):
        smt_i = SymbolicInt._coerce_to_smt_sort(i)
        if space.smt_fork(z3.Or(smt_i >= smt_len, smt_i < -smt_len)):
            raise IndexError
        return normalize_symbolic_index(i)
    elif isinstance(i, slice):
        start, stop, step = (i.start, i.stop, i.step)
        for x in (start, stop, step):
            if (x is not None) and (not hasattr(x, "__index__")):
                raise TypeError(
                    "slice indices must be integers or None or have an __index__ method"
                )
        if step is not None:
            with ResumedTracing():  # Resume tracing for symbolic equality comparison:
                if step != 1:
                    # TODO: do more with slices and steps
                    raise CrosshairUnsupported("slice steps not handled")
        if i.start is None:
            start = z3.IntVal(0)
        else:
            start = normalize_symbolic_index(start)
        if i.stop is None:
            stop = smt_len
        else:
            stop = normalize_symbolic_index(stop)
        return (start, stop)
    else:
        raise TypeError("indices must be integers or slices, not " + str(type(i)))


def clip_range_to_symbolic_len(
    space: StateSpace,
    start: z3.ExprRef,
    stop: z3.ExprRef,
    smt_len: z3.ExprRef,
) -> Tuple[z3.ExprRef, z3.ExprRef]:
    if space.smt_fork(start < 0):
        start = z3.IntVal(0)
    elif space.smt_fork(smt_len < start):
        start = smt_len
    if space.smt_fork(stop < 0):
        stop = z3.IntVal(0)
    elif space.smt_fork(smt_len < stop):
        stop = smt_len
    return (start, stop)


def process_slice_vs_symbolic_len(
    space: StateSpace,
    i: Union[int, slice],
    smt_len: z3.ExprRef,
) -> Union[z3.ExprRef, Tuple[z3.ExprRef, z3.ExprRef]]:
    ret = flip_slice_vs_symbolic_len(space, i, smt_len)
    if isinstance(ret, tuple):
        (start, stop) = ret
        return clip_range_to_symbolic_len(space, start, stop, smt_len)
    return ret


class SymbolicSequence(SymbolicValue, collections.abc.Sequence):
    def __ch_realize__(self):
        return origin_of(self.python_type)(self)

    def __iter__(self):
        idx = 0
        while len(self) > idx:
            yield self[idx]
            idx += 1

    def __len__(self):
        return SymbolicInt(z3.Length(self.var))

    def __bool__(self):
        return SymbolicBool(z3.Length(self.var) > 0).__bool__()

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


class SymbolicArrayBasedUniformTuple(SymbolicSequence):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Any):
        if type(smtvar) == str:
            pass
        else:
            assert type(smtvar) is tuple, f"incorrect type {type(smtvar)}"
            assert len(smtvar) == 2

        self.val_pytype = normalize_pytype(type_arg_of(typ, 0))
        ch_types = crosshair_types_for_python_type(self.val_pytype)
        if ch_types:
            self.ch_item_type: Optional[Type[AtomicSymbolicValue]] = ch_types[0]
            self.item_smt_sort = self.ch_item_type._ch_smt_sort()
        else:
            self.ch_item_type = None
            self.item_smt_sort = HeapRef

        SymbolicValue.__init__(self, smtvar, typ)
        arr_var = self._arr()
        len_var = self._len()
        self.statespace.add(len_var >= 0)

    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        arr_smt_type = z3.ArraySort(_SMT_INT_SORT, self.item_smt_sort)
        return (
            z3.Const(varname + "_map" + self.statespace.uniq(), arr_smt_type),
            z3.Const(varname + "_len" + self.statespace.uniq(), _SMT_INT_SORT),
        )

    def _arr(self):
        return self.var[0]

    def _len(self):
        return self.var[1]

    def __len__(self):
        return SymbolicInt(self._len())

    def __bool__(self) -> bool:
        return SymbolicBool(self._len() != 0).__bool__()

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

    def __iter__(self):
        arr_var, len_var = self.var
        idx = 0
        while SymbolicBool(idx < len_var).__bool__():
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
        with NoTracing():
            if not is_heapref_sort(self.item_smt_sort):
                smt_other = self.ch_item_type._coerce_to_smt_sort(other)
                if smt_other is not None:
                    # OK to perform a symbolic comparison
                    idx = z3.Const("possible_idx" + space.uniq(), _SMT_INT_SORT)
                    idx_in_range = z3.Exists(
                        idx,
                        z3.And(
                            0 <= idx,
                            idx < self._len(),
                            z3.Select(self._arr(), idx) == smt_other,
                        ),
                    )
                    return SymbolicBool(idx_in_range)
        # Fall back to standard equality and iteration
        for self_item in self:
            if self_item == other:  # TODO test customized equality better
                return True
        return False

    def __getitem__(self, i):
        space = self.statespace
        with NoTracing():
            if (
                isinstance(i, slice)
                and i.start is None
                and i.stop is None
                and i.step is None
            ):
                return self
            idx_or_pair = process_slice_vs_symbolic_len(space, i, self._len())
            if isinstance(idx_or_pair, tuple):
                (start, stop) = idx_or_pair
                (myarr, mylen) = self.var
                start = SymbolicInt(start)
                stop = SymbolicInt(smt_min(mylen, smt_coerce(stop)))
                with ResumedTracing():
                    return SliceView(self, start, stop)
            else:
                smt_result = z3.Select(self._arr(), idx_or_pair)
                return smt_to_ch_value(
                    space, self.snapshot, smt_result, self.val_pytype
                )

    def index(
        self, value: object, start: int = 0, stop: int = 9223372036854775807
    ) -> int:
        try:
            start, stop = start.__index__(), stop.__index__()
        except AttributeError:
            # Re-create the error that list.index would give on bad start/stop values:
            raise TypeError(
                "slice indices must be integers or have an __index__ method"
            )
        mylen = len(self)
        if start < 0:
            start += mylen
        if stop < 0:
            stop += mylen
        for idx in range(max(start, 0), min(stop, mylen)):
            if self[idx] == value:
                return idx
        raise ValueError


class SymbolicList(
    ShellMutableSequence, collections.abc.MutableSequence, CrossHairValue
):
    def __init__(self, *a):
        ShellMutableSequence.__init__(self, SymbolicArrayBasedUniformTuple(*a))

    def __ch_pytype__(self):
        return python_type(self.inner)

    def __ch_realize__(self):
        items = tuple(i for i in self)
        with NoTracing():
            return list(items)

    def __lt__(self, other):
        if not isinstance(other, (list, SymbolicList)):
            raise TypeError
        return super().__lt__(other)

    def __mod__(self, *a):
        raise TypeError


class SymbolicType(AtomicSymbolicValue, SymbolicValue):
    _realization: Optional[Type] = None

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        space = context_statespace()
        assert origin_of(typ) is type
        self.pytype_cap = (
            origin_of(typ.__args__[0]) if hasattr(typ, "__args__") else object
        )
        assert type(self.pytype_cap) is type
        smt_cap = space.type_repo.get_type(self.pytype_cap)
        SymbolicValue.__init__(self, smtvar, typ)
        space.add(space.type_repo.smt_issubclass(self.var, smt_cap))
        # Our __getattr__() forces realization.
        # Explicitly set some attributes to avoid this:
        self.__origin__ = self

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
        if self is SymbolicType:
            return False
        if type(other) is SymbolicType:
            # Prefer it this way because only _is_subcless_of_ does the type cap lowering.
            return other._is_subclass_of_(self)
        space = self.statespace
        with NoTracing():
            coerced = SymbolicType._coerce_to_smt_sort(other)
            if coerced is None:
                return False
            return SymbolicBool(space.type_repo.smt_issubclass(coerced, self.var))

    def _is_subclass_of_(self, other):
        if self is SymbolicType:
            return False
        space = self.statespace
        with NoTracing():
            coerced = SymbolicType._coerce_to_smt_sort(other)
            if coerced is None:
                return False
            ret = SymbolicBool(space.type_repo.smt_issubclass(self.var, coerced))
            if type(other) is SymbolicType:
                other_pytype = other.pytype_cap
            elif issubclass(other, SymbolicValue):
                if issubclass(other, AtomicSymbolicValue):
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

    def __getattr__(self, attrname: str) -> object:
        return getattr(self._realized(), attrname)

    def _realized(self):
        if self._realization is None:
            self._realization = self._realize()
        return self._realization

    def _realize(self) -> Type:
        with NoTracing():
            cap = self.pytype_cap
            space = self.statespace
            if cap is object:
                # We don't attempt every possible Python type! Just some basic ones.
                type_repo = space.type_repo
                for pytype in (int, str):
                    smt_type = type_repo.get_type(pytype)
                    if space.smt_fork(self.var == smt_type, favor_true=True):
                        return pytype
                raise CrosshairUnsupported(
                    "Will not exhaustively attempt `object` types"
                )
            else:
                subtype = choose_type(space, cap)
                smt_type = space.type_repo.get_type(subtype)
                if space.smt_fork(self.var == smt_type, favor_true=True):
                    return subtype
                else:
                    raise IgnoreAttempt

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
            # TODO: add deepcopy here. (this breaks a few tests)
            result = self.__ch_realize__()
        else:
            inner = object.__getattribute__(self, "_inner")
            if inner is _MISSING:
                # CrossHair will deepcopy for mutation checking.
                # That's usually bad for LazyObjects, which want to defer their
                # realization, so we simply don't do mutation checking for these
                # kinds of values right now.
                result = self
            else:
                result = copy.deepcopy(inner)
        memo[id(self)] = result
        return result


class SymbolicObject(LazyObject, CrossHairValue):
    """
    An object with an unknown type.
    We lazily create a more specific smt-based value in hopes that an
    isinstance() check will be called before something is accessed on us.
    Note that this class is not an SymbolicValue, but its _typ and _inner
    members can be.
    """

    # TODO: prefix comparison checks with type checks to encourage us to become the
    # right type.

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        object.__setattr__(self, "_typ", SymbolicType(smtvar, type))
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

    def __ch_pytype__(self):
        return object.__getattribute__(self, "_typ")

    @property
    def __class__(self):
        return SymbolicObject

    @__class__.setter
    def __class__(self, value):
        raise CrosshairUnsupported


class SymbolicCallable(SymbolicValue):
    __closure__ = None

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        SymbolicValue.__init__(self, smtvar, typ)

    def __bool__(self):
        return True

    def __eq__(self, other):
        return (self.var is other.var) if isinstance(other, SymbolicCallable) else False

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
        with NoTracing():
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


class SymbolicUniformTuple(
    SymbolicArrayBasedUniformTuple, collections.abc.Sequence, collections.abc.Hashable
):
    def __repr__(self):
        return tuple(self).__repr__()

    def __hash__(self):
        return tuple(self).__hash__()


_SMTSTR_Z3_SORT = z3.SeqSort(z3.BitVecSort(8))


class SymbolicStr(AtomicSymbolicValue, SymbolicSequence, AbcString):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = str):
        assert typ == str
        SymbolicValue.__init__(self, smtvar, typ)
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
        return SymbolicStr(self.var)

    def __repr__(self):
        return repr(self.__str__())

    def __hash__(self):
        return hash(self.__str__())

    @staticmethod
    def _concat_strings(
        a: Union[str, "SymbolicStr"], b: Union[str, "SymbolicStr"]
    ) -> Union[str, "SymbolicStr"]:
        # Assumes at least one argument is symbolic and not tracing
        if isinstance(a, SymbolicStr) and isinstance(b, SymbolicStr):
            return SymbolicStr(a.var + b.var)
        elif (
            isinstance(a, str)
            and isinstance(b, SymbolicStr)
            and all(ord(c) < 256 for c in a)
        ):
            return SymbolicStr(SymbolicStr._coerce_to_smt_sort(a) + b.var)
        elif (
            isinstance(a, SymbolicStr)
            and isinstance(b, str)
            and all(ord(c) < 256 for c in b)
        ):
            return SymbolicStr(a.var + SymbolicStr._coerce_to_smt_sort(b))
        else:
            return realize(a) + realize(b)

    def __add__(self, other):
        with NoTracing():
            if not isinstance(other, (SymbolicStr, str)):
                raise TypeError
            return SymbolicStr._concat_strings(self, other)

    def __radd__(self, other):
        with NoTracing():
            if not isinstance(other, (SymbolicStr, str)):
                raise TypeError
            return SymbolicStr._concat_strings(other, self)

    def __mul__(self, other):
        space = self.statespace
        if isinstance(other, Integral):
            if other <= 1:
                return self if other == 1 else ""
            # Note that in SymbolicInt, we attempt string multiplication via regex.
            # Z3 cannot do much with a symbolic regex, so we case-split on
            # the repetition count.
            return SymbolicStr(z3.Concat(*[self.var for _ in range(other)]))
        return NotImplemented

    __rmul__ = __mul__

    def __mod__(self, other):
        return self.__str__() % realize(other)

    def _cmp_op(self, other, op):
        forced = force_to_smt_sort(other, SymbolicStr)
        return SymbolicBool(op(self.var, forced))

    def __lt__(self, other):
        return self._cmp_op(other, ops.lt)

    def __le__(self, other):
        return self._cmp_op(other, ops.le)

    def __gt__(self, other):
        return self._cmp_op(other, ops.gt)

    def __ge__(self, other):
        return self._cmp_op(other, ops.ge)

    def __contains__(self, other):
        forced = force_to_smt_sort(other, SymbolicStr)
        return SymbolicBool(z3.Contains(self.var, forced))

    def __getitem__(self, i: Union[int, slice]):
        with NoTracing():
            idx_or_pair = process_slice_vs_symbolic_len(
                self.statespace, i, z3.Length(self.var)
            )
            if isinstance(idx_or_pair, tuple):
                (start, stop) = idx_or_pair
                smt_result = z3.Extract(self.var, start, stop - start)
            else:
                smt_result = z3.Extract(self.var, idx_or_pair, 1)
            return SymbolicStr(smt_result)

    def count(self, substr, start=None, end=None):
        sliced = self[start:end]
        if substr == "":
            return len(sliced) + 1
        return len(sliced.split(substr)) - 1

    def endswith(self, substr):
        smt_substr = force_to_smt_sort(substr, SymbolicStr)
        return SymbolicBool(z3.SuffixOf(smt_substr, self.var))

    def find(self, substr, start=None, end=None):
        if not isinstance(substr, str):
            raise TypeError
        with NoTracing():
            space = self.statespace
            smt_my_len = z3.Length(self.var)
            if start is None and end is None:
                smt_start = z3.IntVal(0)
                smt_end = smt_my_len
                smt_str = self.var
                if len(substr) == 0:
                    return 0
            else:
                (smt_start, smt_end) = flip_slice_vs_symbolic_len(
                    space, slice(start, end, None), smt_my_len
                )
                if len(substr) == 0:
                    # Add oddity of CPython. We can find the empty string when over-slicing
                    # off the left side of the string, but not off the right:
                    # ''.find('', 3, 4) == -1
                    # ''.find('', -4, -3) == 0
                    if space.smt_fork(smt_start > smt_my_len):
                        return -1
                    elif space.smt_fork(smt_start > 0):
                        return SymbolicInt(smt_start)
                    else:
                        return 0
                (smt_start, smt_end) = clip_range_to_symbolic_len(
                    space, smt_start, smt_end, smt_my_len
                )
                smt_str = z3.SubString(self.var, smt_start, smt_end - smt_start)

            smt_sub = force_to_smt_sort(substr, SymbolicStr)
            if space.smt_fork(z3.Contains(smt_str, smt_sub)):
                return SymbolicInt(z3.IndexOf(smt_str, smt_sub, 0) + smt_start)
            else:
                return -1

    def index(self, substr, start=None, end=None):
        idx = self.find(substr, start, end)
        if idx == -1:
            raise ValueError
        return idx

    def join(self, itr):
        return _join(self, itr, self_type=str, item_type=str)

    def ljust(self, width, fillchar=" "):
        if not isinstance(fillchar, str):
            raise TypeError
        if not isinstance(width, int):
            raise TypeError
        if len(fillchar) != 1:
            raise TypeError
        return self + fillchar * max(0, width - len(self))

    def partition(self, sep: str):
        result = self.split(sep, maxsplit=1)
        if len(result) == 1:
            return (self, "", "")
        elif len(result) == 2:
            return (result[0], sep, result[1])

    def replace(self, old, new, count=-1):
        if not isinstance(old, str) or not isinstance(new, str):
            raise TypeError
        if count == 0:
            return self
        if self == "":
            return new if old == "" else self
        elif old == "":
            return new + self[:1] + self[1:].replace(old, new, count - 1)

        index = self.find(old)
        if index == -1:
            return self
        return (
            self[:index] + new + self[index + len(old) :].replace(old, new, count - 1)
        )

    def rfind(self, substr, start=None, end=None) -> Union[int, SymbolicInt]:
        if not isinstance(substr, str):
            raise TypeError
        with NoTracing():
            space = self.statespace
            smt_my_len = z3.Length(self.var)
            if start is None and end is None:
                smt_start = z3.IntVal(0)
                smt_end = smt_my_len
                smt_str = self.var
                if len(substr) == 0:
                    return SymbolicInt(smt_my_len)
            else:
                (smt_start, smt_end) = flip_slice_vs_symbolic_len(
                    space, slice(start, end, None), smt_my_len
                )
                if len(substr) == 0:
                    # Add oddity of CPython. We can find the empty string when over-slicing
                    # off the left side of the string, but not off the right:
                    # ''.find('', 3, 4) == -1
                    # ''.find('', -4, -3) == 0
                    if space.smt_fork(smt_start > smt_my_len):
                        return -1
                    elif space.smt_fork(smt_end < 0):
                        return 0
                    elif space.smt_fork(smt_end < smt_my_len):
                        return SymbolicInt(smt_end)
                    else:
                        return SymbolicInt(smt_my_len)
                (smt_start, smt_end) = clip_range_to_symbolic_len(
                    space, smt_start, smt_end, smt_my_len
                )
                smt_str = z3.SubString(self.var, smt_start, smt_end - smt_start)
            smt_sub = force_to_smt_sort(substr, SymbolicStr)
            if space.smt_fork(z3.Contains(smt_str, smt_sub)):
                uniq = space.uniq()
                # Divide my contents into 4 concatenated parts:
                prefix = SymbolicStr(f"prefix_{uniq}")
                match1 = SymbolicStr(f"match1_{uniq}")
                match_tail = SymbolicStr(f"match_tail_{uniq}")
                suffix = SymbolicStr(f"suffix_{uniq}")
                space.add(z3.Length(match1.var) == 1)
                space.add(smt_sub == z3.Concat(match1.var, match_tail.var))
                space.add(smt_str == z3.Concat(prefix.var, smt_sub, suffix.var))
                space.add(
                    z3.Not(z3.Contains(z3.Concat(match_tail.var, suffix.var), smt_sub))
                )
                return SymbolicInt(smt_start + z3.Length(prefix.var))
            else:
                return -1

    def rindex(self, substr, start=None, end=None):
        result = self.rfind(substr, start, end)
        if result == -1:
            raise ValueError
        else:
            return result

    def rjust(self, width, fillchar=" "):
        if not isinstance(fillchar, str):
            raise TypeError
        if not isinstance(width, int):
            raise TypeError
        if len(fillchar) != 1:
            raise TypeError
        return fillchar * max(0, width - len(self)) + self

    def rpartition(self, sep: str):
        result = self.rsplit(sep, maxsplit=1)
        if len(result) == 1:
            return ("", "", self)
        elif len(result) == 2:
            return (result[0], sep, result[1])

    def rsplit(self, sep: Optional[str] = None, maxsplit: int = -1):
        if sep is None:
            return self.__str__().rsplit(sep=sep, maxsplit=maxsplit)
        if not isinstance(sep, str):
            raise TypeError
        if not isinstance(maxsplit, Integral):
            raise TypeError
        if len(sep) == 0:
            raise ValueError("empty separator")
        smt_sep = force_to_smt_sort(sep, SymbolicStr)
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

    def split(self, sep: Optional[str] = None, maxsplit: int = -1):
        if sep is None:
            return self.__str__().split(sep=sep, maxsplit=maxsplit)
        if not isinstance(sep, str):
            raise TypeError
        if not isinstance(maxsplit, Integral):
            raise TypeError
        if len(sep) == 0:
            raise ValueError("empty separator")
        smt_sep = force_to_smt_sort(sep, SymbolicStr)
        if maxsplit == 0:
            return [self]
        first_occurance = SymbolicInt(z3.IndexOf(self.var, smt_sep, 0))
        if first_occurance == -1:
            return [self]
        ret = [self[: cast(int, first_occurance)]]
        new_maxsplit = -1 if maxsplit < 0 else maxsplit - 1
        ret.extend(
            self[first_occurance + len(sep) :].split(sep=sep, maxsplit=new_maxsplit)
        )
        return ret

    def startswith(self, substr, start=None, end=None):
        if isinstance(substr, tuple):
            return any(self.startswith(s, start, end) for s in substr)
        smt_substr = force_to_smt_sort(substr, SymbolicStr)
        if start is not None or end is not None:
            # TODO: "".startswith("", 1) should be False, not True
            return self[start:end].startswith(substr)
        return SymbolicBool(z3.PrefixOf(smt_substr, self.var))

    def translate(self, table):
        retparts: List[str] = []
        for ch in self:
            try:
                target = table[ord(ch)]
            except (KeyError, IndexError):
                retparts.append(ch)
                continue
            if isinstance(target, int):
                retparts.append(chr(target))
            elif isinstance(target, str):
                retparts.append(target)
            elif target is not None:
                raise TypeError
        return "".join(retparts)

    def zfill(self, width):
        if not isinstance(width, int):
            raise TypeError
        fill_length = max(0, width - len(self))
        if self.startswith("+") or self.startswith("-"):
            return self[0] + "0" * fill_length + self[1:]
        else:
            return "0" * fill_length + self


def _bytes_data_prop(s):
    with NoTracing():
        return bytes(s.inner)


@total_ordering
class SymbolicBytes(collections.abc.ByteString, AbcString, CrossHairValue):
    def __init__(self, inner):
        self.inner = inner

    data = property(_bytes_data_prop)

    def __ch_realize__(self):
        return bytes(self.inner)

    def __ch_pytype__(self):
        return bytes

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i: Union[int, slice]):
        if isinstance(i, slice):
            return SymbolicBytes(self.inner.__getitem__(i))
        else:
            return self.inner.__getitem__(i)

    def __repr__(self):
        return repr(realize(self))

    def __iter__(self):
        return self.inner.__iter__()

    def __eq__(self, other) -> bool:
        if isinstance(other, collections.abc.ByteString):
            return self.inner == list(other)
        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, collections.abc.ByteString):
            return self.inner < list(other)
        else:
            raise TypeError

    def __copy__(self):
        return SymbolicBytes(self.inner)

    def decode(self, encoding="utf-8", errors="strict"):
        realize(self).decode(encoding=encoding, errors=errors)


def make_byte_string(creator: SymbolicFactory):
    nums = creator(Tuple[int, ...])

    # A quantifier-based approach:
    z3_array = nums._arr()
    space = context_statespace()
    qvar = z3.Int("bytevar" + space.uniq())
    space.add(z3.ForAll([qvar], 0 <= z3.Select(z3_array, qvar)))
    space.add(z3.ForAll([qvar], z3.Select(z3_array, qvar) < 256))

    # An alternative, deferred-assuption approach:
    # creator.space.defer_assumption(
    #     "bytes are valid bytes", lambda: all(0 <= v < 256 for v in values)
    # )

    # Yet another alternative would be to used bounded-size bytstrings to avoid
    # quantification.

    return SymbolicBytes(nums)


class SymbolicByteArray(
    ShellMutableSequence,
    AbcString,
    CrossHairValue,
):
    def __init__(self, byte_string: Sequence):
        super().__init__(byte_string)

    __hash__ = None  # type: ignore
    data = property(_bytes_data_prop)

    def __ch_realize__(self):
        return bytearray(self.inner)

    def __ch_pytype__(self):
        return bytearray

    def _spawn(self, items: Sequence) -> ShellMutableSequence:
        return SymbolicByteArray(items)


_CACHED_TYPE_ENUMS: Dict[FrozenSet[type], z3.SortRef] = {}


_PYTYPE_TO_WRAPPER_TYPE = {
    bool: (SymbolicBool,),
    int: (SymbolicInt,),
    float: (SymbolicFloat,),
    str: (SymbolicStr,),
    type: (SymbolicType,),
}

# Type ignore pending https://github.com/python/mypy/issues/6864
_PYTYPE_TO_WRAPPER_TYPE[collections.abc.Callable] = (SymbolicCallable,)  # type:ignore

_WRAPPER_TYPE_TO_PYTYPE = dict(
    (v, k) for (k, vs) in _PYTYPE_TO_WRAPPER_TYPE.items() for v in vs
)


#
# Symbolic-making helpers
#


def make_union_choice(creator: SymbolicFactory, *pytypes):
    for typ in pytypes[:-1]:
        if creator.space.smt_fork(desc="choose_" + smtlib_typename(typ)):
            return creator(typ)
    return creator(pytypes[-1])


def make_optional_smt(smt_type):
    def make(creator: SymbolicFactory, *type_args):
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


def make_dictionary(creator: SymbolicFactory, key_type=Any, value_type=Any):
    space, varname = creator.space, creator.varname
    if pytype_uses_heap(key_type):
        kv = proxy_for_type(
            List[Tuple[key_type, value_type]],  # type: ignore
            varname + "items",
            allow_subtypes=False,
        )
        orig_kv = kv[:]

        def ensure_keys_are_unique() -> bool:
            return len(set(deep_realize(k) for k, _ in orig_kv)) == len(orig_kv)

        space.defer_assumption("dict keys are unique", ensure_keys_are_unique)
        return SimpleDict(kv)
    return ShellMutableMap(SymbolicDict(varname, creator.pytype))


def make_tuple(creator: SymbolicFactory, *type_args):
    if not type_args:
        type_args = (object, ...)  # type: ignore
    if len(type_args) == 2 and type_args[1] == ...:
        return SymbolicUniformTuple(creator.varname, creator.pytype)
    else:
        return tuple(
            proxy_for_type(t, creator.varname + "_at_" + str(idx), allow_subtypes=True)
            for (idx, t) in enumerate(type_args)
        )


def make_set(creator: SymbolicFactory, *type_args) -> ShellMutableSet:
    if type_args:
        return ShellMutableSet(creator(FrozenSet.__getitem__(*type_args)))  # type: ignore
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


def fork_on_useful_attr_names(obj: object, name: SymbolicStr) -> None:
    # This function appears to do nothing at all!
    # It exists to force a symbolic string into useful candidate states.
    with NoTracing():
        obj = realize(obj)
        for key in reversed(dir(obj)):
            # We use reverse() above to handle __dunder__ methods last.
            if name == key:
                return


_ascii = with_realized_args(orig_builtins.ascii)

_bin = with_realized_args(orig_builtins.bin)


def _bytearray(*a):
    if len(a) == 1:
        with NoTracing():
            (source,) = a
            if isinstance(source, SymbolicByteArray):
                return SymbolicByteArray(source.inner)
            elif isinstance(source, SymbolicBytes):
                return SymbolicByteArray(source.inner)
    # We make ALL bytearrays symbolic.
    # (concrete bytearrays are impossible to mutate symbolically)
    # return bytearray(*a)
    return SymbolicByteArray(bytes(*a))  # type: ignore


def _bytes(*a):
    with NoTracing():
        if len(a) != 1:
            return bytes(*a)  # type: ignore
        (source,) = a
        if isinstance(source, SymbolicByteArray):
            return SymbolicBytes(source.inner)
        elif isinstance(source, SymbolicBytes):
            return SymbolicBytes(source.inner)
        if is_iterable(source):
            source = list(source)
            if any(isinstance(i, SymbolicIntable) for i in source):
                return SymbolicBytes(source)
        return bytes(source)


_callable = with_realized_args(orig_builtins.callable)


def _chr(i: int) -> Union[str, SymbolicStr]:
    if i < 0 or i > 0x10FFFF:
        raise ValueError
    with NoTracing():
        if isinstance(i, SymbolicInt):
            space = context_statespace()
            if space.smt_fork(i.var < 256, favor_true=True):
                ret = SymbolicStr("chr" + space.uniq())
                space.add(ret.var == z3.Unit(z3.Int2BV(i.var, 8)))
                return ret
    return chr(realize(i))


def _eval(expr: str, _globals=None, _locals=None) -> object:
    # This is fragile: consider detecting _crosshair_wrapper(s):
    calling_frame = sys._getframe(1)
    _globals = calling_frame.f_globals if _globals is None else realize(_globals)
    _locals = calling_frame.f_locals if _locals is None else realize(_locals)
    return eval(realize(expr), _globals, _locals)


def _format(obj: object, format_spec: str = "") -> str:
    with NoTracing():
        if isinstance(obj, SymbolicValue):
            obj = realize(obj)
        if type(format_spec) is SymbolicStr:
            format_spec = realize(format_spec)
    return format(obj, format_spec)


def _getattr(obj: object, name: str, default=_MISSING) -> object:
    with NoTracing():
        if type(name) is SymbolicStr:
            fork_on_useful_attr_names(obj, name)  # type:ignore
            name = realize(name)
        if default is _MISSING:
            return getattr(obj, name)
        else:
            return getattr(obj, name, default)


def _hasattr(obj: object, name: str) -> bool:
    with NoTracing():
        if type(name) is SymbolicStr:
            fork_on_useful_attr_names(obj, name)  # type:ignore
            name = realize(name)
        return hasattr(obj, name)


def _hash(obj: Hashable) -> int:
    """
    post[]: smt_and(-2**63 <= _, _ < 2**63)
    """
    # Skip the built-in hash if possible, because it requires the output
    # to be a native int.
    with NoTracing():
        if not is_hashable(obj):
            return hash(obj)  # error in the native way
        objtype = type(obj)
    # You might think we'd say "return obj.__hash__()" here, but we need some
    # special gymnastics to avoid "metaclass confusion".
    # See: https://docs.python.org/3/reference/datamodel.html#special-method-lookup
    return objtype.__hash__(obj)


def _int(val: object = 0, *a):
    with NoTracing():
        if isinstance(val, SymbolicInt):
            return val
        if isinstance(val, SymbolicStr) and a == ():
            # TODO symbolic string parsing with a base; e.g. int("a7", 16)
            nonnumeric = z3.Not(z3.InRe(val.var, z3.Plus(z3.Range("0", "9"))))
            if not context_statespace().smt_fork(nonnumeric):
                return SymbolicInt(z3.StrToInt(val.var))
    return int(realize(val), *realize(a))  # type: ignore


_FLOAT_REGEX = re.compile(
    r"""
      (?P<posneg> (\+|\-|))
      (?P<intpart>(\d+))
      (\.(?P<fraction>\d*))?
""",
    re.VERBOSE,
)
# TODO handle exponents: ((e|E)(\+|\-|) (\d+)(\.\d+)?)?
# TODO allow underscores (only in between digits)
# TODO handle special floats (nan, inf, -inf)
# TODO once regex is perfect, return ValueError directly, instead of realizing input
#      (this is important because realization impacts search heuristics)


def _float(val=0.0):
    with NoTracing():
        if isinstance(val, SymbolicFloat):
            return val
        is_symbolic_str = isinstance(val, SymbolicStr)
        is_symbolic_int = isinstance(val, SymbolicInt)
    if is_symbolic_str:
        match = _FLOAT_REGEX.fullmatch(val)
        if match:
            ret = _float(int(match.group("intpart")))
            decimal_digits = match.group("fraction")
            if decimal_digits:
                denominator = realize(len(decimal_digits))
                ret += _float(int(decimal_digits)) / (10 ** denominator)
            if match.group("posneg") == "-":
                ret = -ret
            return ret
    elif is_symbolic_int:
        with NoTracing():
            return SymbolicFloat(z3.ToReal(val.var))
    return float(realize(val))


# Trick the system into believing that symbolic values are
# native types.
def _issubclass(subclass, superclass):
    with NoTracing():
        if not isinstance(subclass, (type, SymbolicType)):
            return issubclass(subclass, superclass)
        if not isinstance(superclass, (type, SymbolicType)):
            return issubclass(subclass, superclass)
        subclass_is_special = hasattr(subclass, "_is_subclass_of_")
        if not subclass_is_special:
            # We could also check superclass(es) for a special method, but
            # the native function won't return True in those cases anyway.
            try:
                ret = issubclass(subclass, superclass)
                if ret:
                    return True
            except TypeError:
                pass
        if type(superclass) is tuple:
            for cur_super in superclass:
                if _issubclass(subclass, cur_super):
                    return True
            return False
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
            # TODO: some confusion about whether this is a classmethod?
            # Test that "issubclass(SmtList, list) == True", but
            # that "issubclass(SmtList(), list) == False"
            if (
                method(superclass)
                if hasattr(method, "__self__")
                else method(subclass, superclass)
            ):
                return True
        return False


def _isinstance(obj, types):
    return issubclass(type(obj), types)


# CPython's len() forces the return value to be a native integer.
# Avoid that requirement by making it only call __len__().
def _len(l):
    return l.__len__() if hasattr(l, "__len__") else [x for x in l].__len__()


def _ord(c: str) -> int:
    if len(c) != 1:
        raise TypeError
    with NoTracing():
        if isinstance(c, SymbolicStr):
            space = context_statespace()
            ret = SymbolicInt("ord" + space.uniq())
            space.add(c.var == z3.Unit(z3.Int2BV(ret.var, 8)))
            # Int2BV takes the low bits; also force result in the expected range:
            space.add(0 <= ret.var)
            space.add(ret.var < 256)
            return ret
    return ord(realize(c))


def _pow(base, exp, mod=None):
    return pow(realize(base), realize(exp), realize(mod))


# TODO consider what to do
# def print(*a: object, **kw: Any) -> None:
#    '''
#    post: True
#    '''
#    print(*a, **kw)


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
        with NoTracing():
            real_type = type(arg)
        return real_type.__repr__(arg)
    else:
        return repr(arg)


def _setattr(obj: object, name: str, value: object) -> None:
    # TODO: we could do symbolic stuff like getattr does here!
    with NoTracing():
        if isinstance(obj, SymbolicValue):
            obj = realize(obj)
        if type(name) is SymbolicStr:
            name = realize(name)
        return setattr(obj, name, value)


# TODO: is this important? Feels like the builtin might do the same?
def _sorted(l, key=None, reverse=False):
    if not is_iterable(l):
        raise TypeError("object is not iterable")
    ret = list(l.__iter__())
    ret.sort(key=key, reverse=realize(reverse))
    return ret


# TODO: consider what to do here
# def sum(i: Iterable[_T]) -> Union[_T, int]:
#    '''
#    post[]: _ == 0 or len(i) > 0
#    '''
#    return sum(i)


# TODO: I think this breaks dynamically constructed type usages:
# e.g. type(<name>, <bases>, <body dict>)
def _type(obj: object) -> type:
    with NoTracing():
        return python_type(obj)


#
# Patches on builtin classes
#


def _int_from_bytes(b: bytes, byteorder: str, *, signed=False) -> int:
    if not isinstance(byteorder, str):
        raise TypeError
    if byteorder == "big":
        little = False
    elif byteorder == "little":
        little = True
    else:
        raise ValueError
    if not isinstance(b, (bytes, bytearray)):
        raise TypeError
    byteitr: Iterable[int] = reversed(b) if little else b
    val = 0
    invert = None
    numbytes = realize(len(b))
    for byt in byteitr:
        if invert is None and signed and byt >= 128:
            invert = True
        val = (val * 256) + byt
    if invert:
        val -= 256 ** realize(len(b))
    return val


def _list_index(self, value, start=0, stop=9223372036854775807):
    return list.index(self, value, realize(start), realize(stop))


def _join(self: _T, itr: Sequence, self_type: Type[_T], item_type: Type) -> _T:
    # An slow implementation of join for str/bytes, but describable in terms of
    # concatenation, which we can do symbolically.
    # Realizes the length of the argument but not the contents.
    if not isinstance(self, self_type):
        raise TypeError
    result = self_type()
    for idx, item in enumerate(itr):
        if not isinstance(item, item_type):
            raise TypeError
        if idx > 0:
            result = result + self  # type: ignore
        result = result + item
    return result


def _str_join(self, itr) -> str:
    return _join(self, itr, self_type=str, item_type=str)


def _bytes_join(self, itr) -> str:
    return _join(self, itr, self_type=bytes, item_type=collections.abc.ByteString)


def _bytearray_join(self, itr) -> str:
    return _join(self, itr, self_type=bytearray, item_type=collections.abc.ByteString)


def _str_startswith(self, substr, start=None, end=None) -> bool:
    if not isinstance(self, str):
        raise TypeError
    with NoTracing():
        # Handle native values with native implementation:
        if type(substr) is str:
            return self.startswith(substr, start, end)
        if type(substr) is tuple:
            if all(type(i) is str for i in substr):
                return self.startswith(substr, start, end)
        symbolic_self = SymbolicStr(SymbolicStr._smt_promote_literal(self))
    return symbolic_self.startswith(substr, start, end)


def _str_contains(self: str, other: Union[str, SymbolicStr]) -> bool:
    with NoTracing():
        if not isinstance(self, str):
            raise TypeError
        if not isinstance(other, SymbolicStr):
            return self.__contains__(other)
        symbolic_self = SymbolicStr(SymbolicStr._smt_promote_literal(self))
    return symbolic_self.__contains__(other)


#
# Registrations
#


def make_registrations():

    register_type(Union, make_union_choice)

    if sys.version_info >= (3, 8):
        register_type(Final, lambda p, t: p(t))

    # Types modeled in the SMT solver:

    register_type(NoneType, lambda *a: None)
    register_type(bool, make_optional_smt(SymbolicBool))
    register_type(int, make_optional_smt(SymbolicInt))
    register_type(float, make_optional_smt(SymbolicFloat))
    register_type(str, make_optional_smt(SymbolicStr))
    register_type(list, make_optional_smt(SymbolicList))
    register_type(dict, make_dictionary)
    register_type(tuple, make_tuple)
    register_type(set, make_set)
    register_type(frozenset, make_optional_smt(SymbolicFrozenSet))
    register_type(type, make_optional_smt(SymbolicType))
    register_type(collections.abc.Callable, make_optional_smt(SymbolicCallable))

    # Most types are not directly modeled in the solver, rather they are built
    # on top of the modeled types. Such types are enumerated here:

    register_type(object, lambda p: SymbolicObject(p.varname, p.pytype))
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
    register_type(bytes, make_byte_string)
    register_type(bytearray, lambda p: SymbolicByteArray(p(bytes)))
    register_type(memoryview, lambda p: p(bytes))
    # AnyStr,  (it's a type var)

    register_type(typing.BinaryIO, lambda p: io.BytesIO(p(bytes)))
    # TODO: handle Any/AnyStr with a custom class that accepts str/bytes interchangably?:
    register_type(
        typing.IO, lambda p, t=Any: p(BinaryIO) if t == "bytes" else p(TextIO)
    )
    # TODO: StringIO (and BytesIO) won't accept SymbolicStr writes.
    # Consider clean symbolic implementations of these.
    register_type(typing.TextIO, lambda p: io.StringIO(str(p(str))))

    register_type(SupportsAbs, lambda p: p(int))
    register_type(SupportsFloat, lambda p: p(float))
    register_type(SupportsInt, lambda p: p(int))
    register_type(SupportsRound, lambda p: p(float))
    register_type(SupportsBytes, lambda p: p(ByteString))
    register_type(SupportsComplex, lambda p: p(complex))

    # Patches

    register_patch(orig_builtins.ascii, _ascii)
    register_patch(orig_builtins.bin, _bin)
    register_patch(orig_builtins.callable, _callable)
    register_patch(orig_builtins.chr, _chr)
    register_patch(orig_builtins.eval, _eval)
    register_patch(orig_builtins.format, _format)
    register_patch(orig_builtins.getattr, _getattr)
    register_patch(orig_builtins.hasattr, _hasattr)
    register_patch(orig_builtins.hash, _hash)
    register_patch(orig_builtins.isinstance, _isinstance)
    register_patch(orig_builtins.issubclass, _issubclass)
    register_patch(orig_builtins.len, _len)
    register_patch(orig_builtins.ord, _ord)
    register_patch(orig_builtins.pow, _pow)
    register_patch(orig_builtins.repr, _repr)
    register_patch(orig_builtins.setattr, _setattr)
    register_patch(orig_builtins.sorted, _sorted)
    register_patch(orig_builtins.type, _type)

    # Patches on constructors
    register_patch(orig_builtins.bytearray, _bytearray)
    register_patch(orig_builtins.bytes, _bytes)
    register_patch(orig_builtins.float, _float)
    register_patch(orig_builtins.int, _int)

    # Patches on str
    names_to_str_patch = [
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
        "strip",
        "translate",
        "zfill",
    ]
    if sys.version_info >= (3, 9):
        names_to_str_patch.append("removeprefix")
        names_to_str_patch.append("removesuffix")
    for name in names_to_str_patch:
        orig_impl = getattr(orig_builtins.str, name)
        register_patch(orig_impl, with_realized_args(orig_impl))
        bytes_orig_impl = getattr(orig_builtins.bytes, name, None)
        if bytes_orig_impl is not None:
            register_patch(bytes_orig_impl, with_realized_args(bytes_orig_impl))

    register_patch(orig_builtins.str.startswith, _str_startswith)
    register_patch(orig_builtins.str.__contains__, _str_contains)
    register_patch(orig_builtins.str.join, _str_join)
    register_patch(orig_builtins.bytes.join, _bytes_join)
    register_patch(orig_builtins.bytearray.join, _bytearray_join)

    # TODO: override str.__new__ to make symbolic strings

    # Patches on list
    register_patch(orig_builtins.list.index, _list_index)

    # Patches on int
    register_patch(orig_builtins.int.from_bytes, _int_from_bytes)

    # Patches on float
    register_patch(
        orig_builtins.float.fromhex, with_realized_args(orig_builtins.float.fromhex)
    )

    setup_binops()
