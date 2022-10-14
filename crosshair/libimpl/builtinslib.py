import codecs
import collections
import copy
import enum
import io
import math
import operator as ops
import re
import string
import sys
import typing
from abc import ABCMeta
from array import array
from dataclasses import dataclass
from functools import wraps
from itertools import zip_longest
from numbers import Integral, Number, Real
from sys import maxunicode
from typing import (
    Any,
    BinaryIO,
    ByteString,
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    Set,
    SupportsAbs,
    SupportsBytes,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    SupportsRound,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

import typing_inspect  # type: ignore
import z3  # type: ignore

from crosshair.abcstring import AbcString
from crosshair.core import (
    CrossHairValue,
    SymbolicFactory,
    deep_realize,
    iter_types,
    normalize_pytype,
    proxy_for_type,
    python_type,
    realize,
    register_patch,
    register_type,
    type_arg_of,
    type_args_of,
    with_realized_args,
    with_symbolic_self,
    with_uniform_probabilities,
)
from crosshair.objectproxy import ObjectProxy
from crosshair.simplestructs import (
    SequenceConcatenation,
    ShellMutableMap,
    ShellMutableSequence,
    ShellMutableSet,
    SimpleDict,
    SliceView,
    compose_slices,
    concatenate_sequences,
)
from crosshair.statespace import (
    HeapRef,
    SnapshotRef,
    StateSpace,
    VerificationStatus,
    context_statespace,
    model_value_to_python,
    prefer_true,
)
from crosshair.tracers import NoTracing, ResumedTracing, is_tracing
from crosshair.type_repo import PYTYPE_SORT, SymbolicTypeRepository
from crosshair.unicode_categories import UnicodeMaskCache
from crosshair.util import (
    ATOMIC_IMMUTABLE_TYPES,
    CrosshairInternal,
    CrosshairUnsupported,
    IgnoreAttempt,
    debug,
    is_hashable,
    is_iterable,
    memo,
    smtlib_typename,
)
from crosshair.z3util import z3IntVal

_T = TypeVar("_T")
_VT = TypeVar("_VT")


class _Missing(enum.Enum):
    value = 0


_MISSING = _Missing.value
NoneType = type(None)


def smt_min(x, y):
    if x is y:
        return x
    return z3.If(x <= y, x, y)


def smt_and(a: bool, b: bool) -> bool:
    with NoTracing():
        if isinstance(a, SymbolicBool) and isinstance(b, SymbolicBool):
            return SymbolicBool(z3.And(a.var, b.var))
    return a and b


def smt_or(a: bool, b: bool) -> bool:
    with NoTracing():
        if isinstance(a, SymbolicBool) and isinstance(b, SymbolicBool):
            return SymbolicBool(z3.Or(a.var, b.var))
    return a or b


def smt_not(x: object) -> Union[bool, "SymbolicBool"]:
    with NoTracing():
        if isinstance(x, SymbolicBool):
            return SymbolicBool(z3.Not(x.var))
    return not x


_NONHEAP_PYTYPES = set([int, float, bool, NoneType, complex])

# TODO: isn't this pretty close to isinstance(typ, AtomicSymbolicValue)?
def pytype_uses_heap(typ: Type) -> bool:
    return not (typ in _NONHEAP_PYTYPES)


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
    typ = _WRAPPER_TYPE_TO_PYTYPE.get(typ, typ)  # TODO is the still required?
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


def invoke_dunder(obj: object, method_name: str, *args, **kwargs):
    """
    Invoke a special method in the same way Python does.


    Emulates how Python calls special methods, avoiding:
    (1) methods directly set on the instance, and
    (2) normal attribute resolution logic (descriptors, etc)
    See https://docs.python.org/3/reference/datamodel.html#special-method-lookup
    """
    method = _MISSING
    with NoTracing():
        mro = type.__dict__["__mro__"].__get__(type(obj))  # type: ignore
        for klass in mro:
            method = klass.__dict__.get(method_name, _MISSING)
            if method is not _MISSING:
                break
    if method is _MISSING:
        return _MISSING
    return method(obj, *args, **kwargs)


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
            casting_fn_name = "__" + target_pytype.__name__ + "__"
            caster = getattr(input_value, casting_fn_name, None)
            if not caster:
                return None
            try:
                converted = caster()
            except TypeError:
                return None
            return cls._coerce_to_smt_sort(converted)
        return None

    def __eq__(self, other):
        with NoTracing():
            coerced = type(self)._coerce_to_smt_sort(other)
            if coerced is None:
                return False
            return SymbolicBool(self.var == coerced)

    def __ne__(self, other):
        with NoTracing():
            coerced = type(self)._coerce_to_smt_sort(other)
            if coerced is None:
                return True
            return SymbolicBool(self.var != coerced)


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
            raise TypeError
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
BinFn = Callable[[Any, Any], Any]
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


_FLIPPED_OPS: Dict[BinFn, BinFn] = {
    ops.ge: ops.le,
    ops.gt: ops.lt,
    ops.le: ops.ge,
    ops.lt: ops.gt,
}


def setup_binop(fn: Callable[[BinFn, Number, Number], Number], reg_ops: Set[BinFn]):
    a, b = _binop_type_hints(fn)
    for a_type in a:
        for b_type in b:
            for op in reg_ops:
                _BIN_OPS_SEARCH_ORDER.append((op, a_type, b_type, fn))

                # Also, handle flipped comparisons transparently:
                ## (a >= b)   <==>   (b <= a)
                if op in (ops.ge, ops.gt, ops.le, ops.lt):

                    def flipped(o: BinFn, x: Number, y: Number) -> Number:
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
        return SymbolicInt(apply_smt(op, a.var, z3IntVal(b)))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: int, b: SymbolicInt):
        return SymbolicInt(apply_smt(op, z3IntVal(a), b.var))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: SymbolicInt, b: int):
        return SymbolicBool(apply_smt(op, a.var, z3IntVal(b)))

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
            return a * (2**b)
        else:
            return a // (2**b)

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
            if b == 0:
                return 0
            mask_mod = _AND_MASKS_TO_MOD.get(b)
            if mask_mod and isinstance(a, SymbolicInt):
                if context_statespace().smt_fork(a.var >= 0, probability_true=0.75):
                    return SymbolicInt(a.var % mask_mod)
                else:
                    return SymbolicInt(b - ((-a.var - 1) % mask_mod))

            # Fall back to full realization
            return op(realize(a), b)

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

    def __format__(self, fmt: str):
        return realize(self).__format__(realize(fmt))


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
            return other * realize(self)
        return numeric_binop(ops.mul, self, other)

    __rmul__ = __mul__

    def bit_count(self):
        if self < 0:
            return (-self).bit_count()
        count = 0
        threshold = 2
        halfway = 1
        while self >= halfway:
            if self % threshold >= halfway:
                count += 1
            threshold *= 2
            halfway *= 2
        return count


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

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return _SMT_INT_SORT

    @classmethod
    def _pytype(cls) -> Type:
        return int

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, int):
            return z3IntVal(literal)
        return None

    @classmethod
    def from_bytes(cls, b: bytes, byteorder: str, signed=False) -> int:
        return int.from_bytes(b, byteorder, signed=signed)  # type: ignore

    def __ch_realize__(self) -> object:
        return self.statespace.find_model_value(self.var)

    def __repr__(self):
        if self < 0:
            return "-" + abs(self).__repr__()
        codepoints = []
        while self >= 10:
            codepoints.append(48 + (self % 10))
            self = self // 10
        codepoints.append(48 + self)
        with NoTracing():
            codepoints.reverse()
            return LazyIntSymbolicStr(codepoints)

    def _symbolic_repr(self):
        # Create a symbolic string representation. Only used in targeted situations.
        # (much of CPython isn't tolerant to symbolic strings)
        is_negative = realize(self < 0)
        if is_negative:
            self = -self
        threshold = 10
        digits = [48 + self % 10]
        while self >= threshold:
            digits.append(48 + (self // threshold) % 10)
            threshold *= 10
        if is_negative:
            digits.append(ord("-"))
        digits.reverse()
        return LazyIntSymbolicStr(digits)

    def __hash__(self):
        return self.__index__().__hash__()

    def __float__(self):
        return SymbolicFloat(smt_int_to_float(self.var))

    def __complex__(self):
        return complex(self.__float__())

    def __index__(self):
        with NoTracing():
            space = context_statespace()
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
            half = (256**length) >> 1
            if self < -half or self >= half:
                raise OverflowError
            if self < 0:
                self = 256**length + self
        else:
            if self < 0 or self >= 256**length:
                raise OverflowError
        intarray = [
            SymbolicInt((self.var / (2 ** (i * 8))) % 256) for i in range(length)
        ]
        if byteorder == "big":
            intarray.reverse()
        return SymbolicBytes(intarray)

    def as_integer_ratio(self) -> Tuple["SymbolicInt", int]:
        return (self, 1)


def make_bounded_int(
    varname: str, minimum: Optional[int] = None, maximum: Optional[int] = None
) -> SymbolicInt:
    space = context_statespace()
    symbolic = SymbolicInt(varname)
    if minimum is not None:
        space.add(symbolic.var >= minimum)
    if maximum is not None:
        space.add(symbolic.var <= maximum)
    return symbolic


_Z3_ONE_HALF = z3.RealVal("1/2")


class SymbolicFloat(SymbolicNumberAble, AtomicSymbolicValue):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = float):
        assert typ is float, f"SymbolicFloat with unexpected python type ({type(typ)})"
        context_statespace().cap_result_at_unknown()
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
        with NoTracing():
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
        var = self.var
        return SymbolicInt(z3.If(var >= 0, z3.ToInt(var), -z3.ToInt(-var)))

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
    """An immutable symbolic dictionary."""

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
                # TODO: this class isn't used with heap-able keys any more I think.
                # So, remove?
                if getattr(k, "__hash__", None) is None:
                    raise TypeError("unhashable type")
                for self_k in iter(self):
                    if self_k == k:
                        return self[self_k]
                raise KeyError
            possibly_missing = self._arr()[smt_key]
            is_missing = self.val_missing_checker(possibly_missing)
            if SymbolicBool(is_missing).__bool__():
                raise KeyError
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
                if space.choose_possible(arr_var == self.empty, probability_true=0.0):
                    raise IgnoreAttempt("SymbolicDict in inconsistent state")
                k = z3.Const("k" + str(idx) + space.uniq(), arr_sort.domain())
                v = z3.Const(
                    "v" + str(idx) + space.uniq(), self.val_constructor.domain(0)
                )
                remaining = z3.Const("remaining" + str(idx) + space.uniq(), arr_sort)
                space.add(arr_var == z3.Store(remaining, k, self.val_constructor(v)))
                space.add(is_missing(z3.Select(remaining, k)))

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
            # In this conditional, we reconcile the parallel symbolic variables for
            # length and contents:
            if space.choose_possible(arr_var != self.empty, probability_true=0.0):
                raise IgnoreAttempt("SymbolicDict in inconsistent state")

    def copy(self):
        return SymbolicDict(self.var, self.python_type)


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
                if space.choose_possible(arr_var == self.empty, probability_true=0.0):
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
                    # NOTE: yield from within tracing/notracing context managers should
                    # be avoided. In the case of a NoTracing + ResumedTracing (only!),
                    # this work out, because it leaves the config stack as it left it,
                    # and, whenever __del__ runs, it also won't harm the config stack.
                    yield ch_value
                arr_var = remaining
            # In this conditional, we reconcile the parallel symbolic variables for length
            # and contents:
            if self.statespace.choose_possible(
                arr_var != self.empty, probability_true=0.0
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
        return deep_realize(self).__repr__()

    def __hash__(self):
        return deep_realize(self).__hash__()

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
            return z3IntVal(idx) if idx >= 0 else (smt_len + z3IntVal(idx))
        else:
            smt_idx = SymbolicInt._coerce_to_smt_sort(idx)
            if space.smt_fork(smt_idx >= 0):  # type: ignore
                return smt_idx
            else:
                return smt_len + smt_idx

    if isinstance(i, (int, SymbolicInt)):  # TODO: what about bools as indexes?
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
            start = z3IntVal(0)
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
        start = z3IntVal(0)
    elif space.smt_fork(smt_len < start):
        start = smt_len
    if space.smt_fork(stop < 0):
        stop = z3IntVal(0)
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
        with NoTracing():
            if self is other:
                return True
            (self_arr, self_len) = self.var
            if isinstance(other, SymbolicArrayBasedUniformTuple):
                return SymbolicBool(
                    z3.And(self_len == other._len(), self_arr == other._arr())
                )
            if not is_iterable(other):
                return False
        if len(self) != len(other):
            return False
        for idx, otherval in enumerate(other):
            myval = self[idx]
            if myval is otherval:
                continue
            if myval != otherval:
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
            return concatenate_sequences(self, other)
        return NotImplemented

    def __radd__(self, other: object):
        if isinstance(other, collections.abc.Sequence):
            return concatenate_sequences(other, self)
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
            if self_item == other:
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
                    return SliceView.slice(self, start, stop)
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
    def __init__(self, arg: Union[Sequence, str], typ=list):
        if isinstance(arg, str):
            ShellMutableSequence.__init__(
                self, SymbolicArrayBasedUniformTuple(arg, typ)
            )
        else:
            ShellMutableSequence.__init__(self, arg)

    def __ch_pytype__(self):
        return list

    def __ch_realize__(self):
        return list(i for i in self)

    def _spawn(self, items: Sequence) -> "ShellMutableSequence":
        return SymbolicList(items)

    def __lt__(self, other):
        if not isinstance(other, (list, SymbolicList)):
            raise TypeError
        return super().__lt__(other)

    def __mod__(self, *a):
        raise TypeError


_EAGER_OBJECT_SUBTYPES = with_uniform_probabilities([int, str])


class SymbolicType(AtomicSymbolicValue, SymbolicValue):
    _realization: Optional[Type] = None

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ):
        assert not is_tracing()
        space = context_statespace()
        assert origin_of(typ) is type
        self.pytype_cap = object
        if hasattr(typ, "__args__"):
            captype = typ.__args__[0]
            # no paramaterized types allowed, e.g. SymbolicType("t", Type[List[int]])
            assert not hasattr(captype, "__args__")
            self.pytype_cap = origin_of(captype)
        assert isinstance(self.pytype_cap, (type, ABCMeta))
        type_repo = space.extra(SymbolicTypeRepository)
        smt_cap = type_repo.get_type(self.pytype_cap)
        SymbolicValue.__init__(self, smtvar, typ)
        space.add(type_repo.smt_issubclass(self.var, smt_cap))
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
            return context_statespace().extra(SymbolicTypeRepository).get_type(literal)
        return None

    def _is_superclass_of_(self, other):
        assert not is_tracing()
        if self is SymbolicType:
            return False
        if type(other) is SymbolicType:
            # Prefer it this way because only _is_subcless_of_ does the type cap lowering.
            return other._is_subclass_of_(self)
        space = self.statespace
        coerced = SymbolicType._coerce_to_smt_sort(other)
        if coerced is None:
            return False
        type_repo = space.extra(SymbolicTypeRepository)
        if space.smt_fork(
            type_repo.smt_can_subclass(coerced, self.var), probability_true=1.0
        ):
            return SymbolicBool(type_repo.smt_issubclass(coerced, self.var))
        else:
            return issubclass(realize(other), realize(self))

    def _is_subclass_of_(self, other):
        assert not is_tracing()
        if self is SymbolicType:
            return False
        space = self.statespace
        coerced = SymbolicType._coerce_to_smt_sort(other)
        if coerced is None:
            return False
        type_repo = space.extra(SymbolicTypeRepository)
        if not space.smt_fork(
            type_repo.smt_can_subclass(self.var, coerced), probability_true=1.0
        ):
            return issubclass(realize(self), realize(other))
        ret = SymbolicBool(type_repo.smt_issubclass(self.var, coerced))
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
            and space.smt_fork(ret.var, probability_true=0.75)
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
            type_repo = space.extra(SymbolicTypeRepository)
            if cap is object:
                # We don't attempt every possible Python type! Just some basic ones.
                for pytype, probability_true in _EAGER_OBJECT_SUBTYPES:
                    smt_type = type_repo.get_type(pytype)
                    if space.smt_fork(
                        self.var == smt_type, probability_true=probability_true
                    ):
                        return pytype
                raise CrosshairUnsupported(
                    "Will not exhaustively attempt `object` types"
                )
            else:
                for pytype, probability_true in iter_types(cap):
                    smt_type = type_repo.get_type(pytype)
                    if space.smt_fork(
                        self.var == smt_type, probability_true=probability_true
                    ):
                        return pytype
                # Do not assume that we have a complete set of possible subclasses:
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


def symbolic_obj_binop(symbolic_obj: "SymbolicObject", other, op):
    other_type = type(other)
    with NoTracing():
        mytype = symbolic_obj._typ

        # This just encourages a useful type realization; we discard the result:
        other_smt_type = SymbolicType._coerce_to_smt_sort(other_type)
        if other_smt_type is not None:
            space = context_statespace()
            space.smt_fork(mytype.var == other_smt_type, probability_true=0.9)

        # The following call then lowers the type cap.
        # TODO: This does more work than is really needed. But it might be good for
        # subclass realizations. We want the equality check above mostly because
        # `object`` realizes to int|str and we don't want to spend lots of time
        # considering (usually enum-based) int and str subclasses.
        mytype._is_subclass_of_(other_type)

    return op(symbolic_obj._wrapped(), other)


class SymbolicObject(ObjectProxy, CrossHairValue):
    """
    An object with an unknown type.
    We lazily create a more specific smt-based value in hopes that an
    isinstance() check will be called before something is accessed on us.
    Note that this class is not an SymbolicValue, but its _typ and _inner
    members can be.
    """

    def __init__(self, smtvar: str, typ: Type):
        object.__setattr__(self, "_typ", SymbolicType(smtvar + "_type", Type[typ]))
        object.__setattr__(self, "_space", context_statespace())
        object.__setattr__(self, "_varname", smtvar)

    def _realize(self):
        object.__getattribute__(self, "_space")
        varname = object.__getattribute__(self, "_varname")

        typ = object.__getattribute__(self, "_typ")
        pytype = realize(typ)
        debug("materializing the type of symbolic", varname, "to be", pytype)
        if pytype is object:
            return object()
        return proxy_for_type(pytype, varname, allow_subtypes=False)

    def _wrapped(self):
        try:
            inner = object.__getattribute__(self, "_inner")
        except AttributeError:
            inner = self._realize()
            object.__setattr__(self, "_inner", inner)
        return inner

    def __ch_realize__(self):
        return realize(self._wrapped())

    def __deepcopy__(self, memo):
        try:
            inner = object.__getattribute__(self, "_inner")
        except AttributeError:
            # CrossHair will deepcopy for mutation checking.
            # That's usually bad for LazyObjects, which want to defer their
            # realization, so we simply don't do mutation checking for these
            # kinds of values right now.
            result = self
        else:
            result = copy.deepcopy(inner)
        memo[id(self)] = result
        return result

    def __ch_pytype__(self):
        return object.__getattribute__(self, "_typ")

    @property
    def __class__(self):
        return SymbolicObject

    @__class__.setter
    def __class__(self, value):
        raise CrosshairUnsupported

    def __lt__(self, other):
        return symbolic_obj_binop(self, other, ops.lt)

    def __le__(self, other):
        return symbolic_obj_binop(self, other, ops.le)

    def __hash__(self):
        return realize(self).__hash__()

    def __eq__(self, other):
        return symbolic_obj_binop(self, other, ops.eq)

    def __ne__(self, other):
        return symbolic_obj_binop(self, other, ops.ne)

    def __gt__(self, other):
        return symbolic_obj_binop(self, other, ops.gt)

    def __ge__(self, other):
        return symbolic_obj_binop(self, other, ops.ge)


class SymbolicCallable:
    __closure__ = None
    __annotations__: dict = {}

    def __init__(self, values: list):
        assert not is_tracing()
        with ResumedTracing():
            if not values:
                raise IgnoreAttempt
        self.values = values
        self.idx = 0

    def __copy__(self):
        return SymbolicCallable(self.values[self.idx :])

    def __eq__(self, other):
        return (
            isinstance(other, SymbolicCallable)
            and self.values[self.idx :] == other.values[self.idx :]
        )

    def __hash__(self):
        # This is needed because contract caching by function uses a hash.
        # And because we want __eq__ for checking equivalence of args post-execution.
        # It's safe to hash to a constant, and doing so defers realization.
        return 42

    def __call__(self, *a, **kw):
        values, idx = self.values, self.idx
        if idx >= len(values):
            return values[-1]
        else:
            return values[idx]

    def __bool__(self):
        return True

    def __repr__(self):
        values = self.values
        if len(values) == 1 and isinstance(values[0], ATOMIC_IMMUTABLE_TYPES):
            return f"lambda *a: {values[0]}"
        value_repr = repr(list(values))
        return f"(x:={value_repr}, lambda *a: x.pop(0) if len(x) > 1 else x[0])[1]"


class SymbolicUniformTuple(
    SymbolicArrayBasedUniformTuple, collections.abc.Sequence, collections.abc.Hashable
):
    def __repr__(self):
        return tuple(self).__repr__()

    def __hash__(self):
        return tuple(self).__hash__()


_SMTSTR_Z3_SORT = z3.SeqSort(z3.IntSort())


def tracing_iter(itr: Iterable[_T]) -> Iterable[_T]:
    """Selectively re-enable tracing only during iteration."""
    assert not is_tracing()
    itr = iter(itr)
    while True:
        try:
            with ResumedTracing():
                value = next(itr)
        except StopIteration:
            return
        yield value


class SymbolicBoundedIntTuple(collections.abc.Sequence):
    def __init__(self, minval: int, maxval: int, varname: str):
        assert not is_tracing()
        self._minval, self._maxval = minval, maxval
        space = context_statespace()
        smtlen = z3.Int(varname + "len" + space.uniq())
        space.add(smtlen >= 0)
        self._varname = varname
        self._len = SymbolicInt(smtlen)
        self._created_vars: List[SymbolicInt] = []

    def _create_up_to(self, size: int) -> None:
        assert not is_tracing()
        assert isinstance(size, int)
        space = context_statespace()
        # TODO: this check is moderately expensive.
        # Investigate whether we can let _created_vars exceed our length.
        if space.smt_fork(self._len.var < z3IntVal(size), probability_true=0.5):
            size = realize(self._len)
        created_vars = self._created_vars
        minval, maxval = self._minval, self._maxval
        for idx in range(len(created_vars), size):
            assert idx == len(created_vars)
            smtval = z3.Int(self._varname + "@" + str(idx))
            space.add(smtval >= minval)
            space.add(smtval <= maxval)
            created_vars.append(SymbolicInt(smtval))

    def __len__(self):
        return self._len

    def __bool__(self) -> bool:
        return SymbolicBool(self._len.var == 0).__bool__()

    def __eq__(self, other):
        if self is other:
            return True
        if not is_iterable(other):
            return False
        if len(self) != len(other):
            return False
        otherlen = realize(len(other))
        with NoTracing():
            self._create_up_to(otherlen)
            constraints = []
            for (int1, int2) in zip(self._created_vars, tracing_iter(other)):
                smtint2 = force_to_smt_sort(int2, SymbolicInt)
                constraints.append(int1.var == smtint2)
            return SymbolicBool(z3.And(*constraints))

    def __repr__(self):
        return str(tuple(self))

    def __iter__(self):
        with NoTracing():
            my_smt_len = self._len.var
            created_vars = self._created_vars
            space = context_statespace()
            idx = -1
        while True:
            with NoTracing():
                idx += 1
                if not space.smt_fork(idx < my_smt_len):
                    return
                self._create_up_to(idx + 1)
            yield created_vars[idx]

    def __add__(self, other: object):
        if isinstance(other, collections.abc.Sequence):
            return concatenate_sequences(self, other)
        return NotImplemented

    def __radd__(self, other: object):
        if isinstance(other, collections.abc.Sequence):
            return concatenate_sequences(other, self)
        return NotImplemented

    def __getitem__(self, argument):
        with NoTracing():
            space = context_statespace()
            if isinstance(argument, slice):
                start, stop, step = argument.start, argument.stop, argument.step
                if start is None and stop is None and step is None:
                    return self
                start, stop, step = realize(start), realize(stop), realize(step)
                mylen = self._len
                if stop and stop > 0 and space.smt_fork(mylen.var >= stop):
                    self._create_up_to(stop)
                elif (
                    stop is None
                    and step is None
                    and (
                        start is None
                        or (0 <= start and space.smt_fork(start <= mylen.var))
                    )
                ):
                    return SliceView(self, start, mylen)
                else:
                    self._create_up_to(realize(mylen))
                return self._created_vars[start:stop:step]
            else:
                argument = realize(argument)
                if argument >= 0 and space.smt_fork(self._len.var > argument):
                    self._create_up_to(realize(argument) + 1)
                else:
                    self._create_up_to(realize(self._len))
                return self._created_vars[argument]

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
        mylen = self._len
        if start < 0:
            start += mylen
        if stop < 0:
            stop += mylen
        for idx in range(max(start, 0), min(stop, mylen)):  # type: ignore
            if self[idx] == value:
                return idx
        raise ValueError


_ASCII_IDENTIFIER_RE = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")


class AnySymbolicStr(AbcString):
    def __ch_is_deeply_immutable__(self) -> bool:
        return True

    def __ch_pytype__(self):
        return str

    def __ch_realize__(self):
        raise NotImplementedError

    def __str__(self):
        with NoTracing():
            return self.__ch_realize__()

    def __repr__(self):
        return repr(self.__str__())

    def _cmp_op(self, other, op):
        assert op in (ops.lt, ops.le, ops.gt, ops.ge)
        if not isinstance(other, str):
            raise TypeError
        if self == other:
            return True if op in (ops.le, ops.ge) else False
        for (mych, otherch) in zip_longest(iter(self), iter(other)):
            if mych == otherch:
                continue
            if mych is None:
                lessthan = True
            elif otherch is None:
                lessthan = False
            else:
                lessthan = ord(mych) < ord(otherch)
            return lessthan if op in (ops.lt, ops.le) else not lessthan
        assert False

    def __lt__(self, other):
        return self._cmp_op(other, ops.lt)

    def __le__(self, other):
        return self._cmp_op(other, ops.le)

    def __gt__(self, other):
        return self._cmp_op(other, ops.gt)

    def __ge__(self, other):
        return self._cmp_op(other, ops.ge)

    def __bool__(self):
        return realize(self.__len__() > 0)

    def capitalize(self):
        if self.__len__() == 0:
            return ""
        if sys.version_info >= (3, 8):
            firstchar = self[0].title()
        else:
            firstchar = self[0].upper()
        return firstchar + self[1:]

    def casefold(self):
        if len(self) != 1:
            return "".join([ch.casefold() for ch in self])
        char = self[0]
        codepoint = ord(char)
        with NoTracing():
            space = context_statespace()
            smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
            cache = space.extra(UnicodeMaskCache)
            if not space.smt_fork(cache.casefold_exists()(smt_codepoint)):
                return char
            smt_1st = cache.casefold_1st()(smt_codepoint)
            if not space.smt_fork(cache.casefold_2nd_exists()(smt_codepoint)):
                return LazyIntSymbolicStr([SymbolicInt(smt_1st)])
            smt_2nd = cache.casefold_2nd()(smt_codepoint)
            if not space.smt_fork(cache.casefold_3rd_exists()(smt_codepoint)):
                return LazyIntSymbolicStr([SymbolicInt(smt_1st), SymbolicInt(smt_2nd)])
            smt_3rd = cache.casefold_3rd()(smt_codepoint)
            return LazyIntSymbolicStr(
                [SymbolicInt(smt_1st), SymbolicInt(smt_2nd), SymbolicInt(smt_3rd)]
            )

    def center(self, width, fillchar=" "):
        if not isinstance(width, int):
            raise TypeError
        if (not isinstance(fillchar, str)) or len(fillchar) != 1:
            raise TypeError
        mylen = self.__len__()
        if mylen >= width:
            return self
        remainder = width - mylen
        smaller_half = remainder // 2
        larger_half = remainder - smaller_half
        if width % 2 == 1:
            return (fillchar * larger_half) + self + (fillchar * smaller_half)
        else:
            return (fillchar * smaller_half) + self + (fillchar * larger_half)

    def count(self, substr, start=None, end=None):
        sliced = self[start:end]
        if substr == "":
            return len(sliced) + 1
        return len(sliced.split(substr)) - 1

    def encode(self, encoding="utf-8", errors="strict"):
        return codecs.encode(self, encoding, errors)

    def expandtabs(self, tabsize=8):
        if not isinstance(tabsize, int):
            raise TypeError
        return self.replace("\t", " " * tabsize)

    def index(self, substr, start=None, end=None):
        idx = self.find(substr, start, end)
        if idx == -1:
            raise ValueError
        return idx

    def _chars_in_maskfn(self, maskfn: z3.ExprRef, ret_if_empty=False):
        # Holds common logic behind the str.is* methods
        space = context_statespace()
        with ResumedTracing():
            if self.__len__() == 0:
                return ret_if_empty
            for char in self:
                codepoint = ord(char)
                with NoTracing():
                    smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
                    if not space.smt_fork(maskfn(smt_codepoint)):
                        return False
        return True

    def isalnum(self):
        with NoTracing():
            maskfn = context_statespace().extra(UnicodeMaskCache).alnum()
            return self._chars_in_maskfn(maskfn)

    def isalpha(self):
        with NoTracing():
            maskfn = context_statespace().extra(UnicodeMaskCache).alpha()
            return self._chars_in_maskfn(maskfn)

    def isascii(self):
        with NoTracing():
            maskfn = context_statespace().extra(UnicodeMaskCache).ascii()
            return self._chars_in_maskfn(maskfn, ret_if_empty=True)

    def isdecimal(self):
        with NoTracing():
            maskfn = context_statespace().extra(UnicodeMaskCache).decimal()
            return self._chars_in_maskfn(maskfn)

    def isdigit(self):
        with NoTracing():
            maskfn = context_statespace().extra(UnicodeMaskCache).digit()
            return self._chars_in_maskfn(maskfn)

    def isidentifier(self):
        if _ASCII_IDENTIFIER_RE.fullmatch(self):
            return True
        elif self.isascii():
            return False
        # The full unicode rules are complex! Resort to realization.
        # (see https://docs.python.org/3.3/reference/lexical_analysis.html#identifiers)
        with NoTracing():
            return realize(self).isidentifier()

    def islower(self):
        with NoTracing():
            space = context_statespace()
            lowerfn = space.extra(UnicodeMaskCache).lower()
            upperfn = space.extra(UnicodeMaskCache).title()  # (covers title and upper)
        if self.__len__() == 0:
            return False
        found_one = False
        for char in self:
            codepoint = ord(char)
            with NoTracing():
                smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
                if space.smt_fork(upperfn(smt_codepoint)):
                    return False
                if space.smt_fork(lowerfn(smt_codepoint)):
                    found_one = True
        return found_one

    def isnumeric(self):
        with NoTracing():
            maskfn = context_statespace().extra(UnicodeMaskCache).numeric()
            return self._chars_in_maskfn(maskfn)

    def isprintable(self):
        with NoTracing():
            maskfn = context_statespace().extra(UnicodeMaskCache).printable()
            return self._chars_in_maskfn(maskfn, ret_if_empty=True)

    def isspace(self):
        with NoTracing():
            maskfn = context_statespace().extra(UnicodeMaskCache).space()
            return self._chars_in_maskfn(maskfn)

    def istitle(self):
        with NoTracing():
            space = context_statespace()
            lowerfn = space.extra(UnicodeMaskCache).lower()
            titlefn = space.extra(UnicodeMaskCache).title()
        expect_upper = True
        found_char = False
        for char in self:
            codepoint = ord(char)
            with NoTracing():
                smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
                if space.smt_fork(titlefn(smt_codepoint)):
                    if not expect_upper:
                        return False
                    expect_upper = False
                    found_char = True
                elif space.smt_fork(lowerfn(smt_codepoint)):
                    if expect_upper:
                        return False
                else:  # (uncased)
                    expect_upper = True
        return found_char

    def isupper(self):
        with NoTracing():
            space = context_statespace()
            lowerfn = space.extra(UnicodeMaskCache).lower()
            upperfn = space.extra(UnicodeMaskCache).upper()
        if self.__len__() == 0:
            return False
        found_one = False
        for char in self:
            codepoint = ord(char)
            with NoTracing():
                smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
                if space.smt_fork(lowerfn(smt_codepoint)):
                    return False
                if space.smt_fork(upperfn(smt_codepoint)):
                    found_one = True
        return found_one

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

    def lower(self):
        if len(self) != 1:
            return "".join([ch.lower() for ch in self])
        char = self[0]
        codepoint = ord(char)
        with NoTracing():
            space = context_statespace()
            smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
            cache = space.extra(UnicodeMaskCache)
            if not space.smt_fork(cache.tolower_exists()(smt_codepoint)):
                return char
            smt_1st = cache.tolower_1st()(smt_codepoint)
            if not space.smt_fork(cache.tolower_2nd_exists()(smt_codepoint)):
                return LazyIntSymbolicStr([SymbolicInt(smt_1st)])
            smt_2nd = cache.tolower_2nd()(smt_codepoint)
            return LazyIntSymbolicStr([SymbolicInt(smt_1st), SymbolicInt(smt_2nd)])

    def lstrip(self, chars=None):
        if chars is None:

            def filter(ch):
                return ch.isspace()

        elif isinstance(chars, str):

            def filter(ch):
                return ch in chars

        else:
            raise TypeError
        for (idx, ch) in enumerate(self):
            if not filter(ch):
                return self[idx:]
        return ""

    def splitlines(self, keepends=False):
        if not isinstance(keepends, int):
            raise TypeError
        mylen = self.__len__()
        if mylen == 0:
            return []
        for (idx, ch) in enumerate(self):
            codepoint = ord(ch)
            with NoTracing():
                space = context_statespace()
                smt_isnewline = space.extra(UnicodeMaskCache).newline()
                smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
                if not space.smt_fork(smt_isnewline(smt_codepoint)):
                    continue
            if codepoint == ord("\r"):
                if idx + 1 < mylen and self[idx + 1] == "\n":
                    token = self[: idx + 2] if keepends else self[:idx]
                    return [token] + self[idx + 2 :].splitlines(keepends)
            token = self[: idx + 1] if keepends else self[:idx]
            return [token] + self[idx + 1 :].splitlines(keepends)
        return [self]

    def removeprefix(self, prefix):
        if not isinstance(prefix, str):
            raise TypeError
        if self.startswith(prefix):
            return self[len(prefix) :]
        return self

    def removesuffix(self, suffix):
        if not isinstance(suffix, str):
            raise TypeError
        if len(suffix) > 0 and self.endswith(suffix):
            return self[: -len(suffix)]
        return self

    def replace(self, old, new, count=-1):
        if not isinstance(old, str) or not isinstance(new, str):
            raise TypeError
        if count == 0:
            return self
        if self == "":
            return new if old == "" else self
        elif old == "":
            return new + self[:1] + self[1:].replace(old, new, count - 1)

        (prefix, match, suffix) = self.partition(old)
        if not match:
            return self
        return prefix + new + suffix.replace(old, new, count - 1)

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

    def rsplit(self, sep: Optional[str] = None, maxsplit: int = -1):
        if sep is None:
            return self.__str__().rsplit(sep=sep, maxsplit=maxsplit)
        if not isinstance(sep, str):
            raise TypeError
        if not isinstance(maxsplit, Integral):
            raise TypeError
        if len(sep) == 0:
            raise ValueError("empty separator")
        if maxsplit == 0:
            return [self]
        last_occurence = self.rfind(sep)
        if last_occurence == -1:
            return [self]
        new_maxsplit = -1 if maxsplit < 0 else maxsplit - 1
        ret = self[:last_occurence].rsplit(sep, new_maxsplit)
        index_after = len(sep) + last_occurence
        ret.append(self[index_after:])
        return ret

    def rstrip(self, chars=None):
        if chars is None:

            def filter(ch):
                return ch.isspace()

        elif isinstance(chars, str):

            def filter(ch):
                return ch in chars

        else:
            raise TypeError
        if self.__len__() == 0:
            return ""
        if filter(self[-1]):
            return self[:-1].rstrip(chars)
        return self

    def split(self, sep: Optional[str] = None, maxsplit: int = -1):
        if sep is None:
            return self.__str__().split(sep=sep, maxsplit=maxsplit)
        if not isinstance(sep, str):
            raise TypeError
        if not isinstance(maxsplit, Integral):
            raise TypeError
        if len(sep) == 0:
            raise ValueError("empty separator")
        if maxsplit == 0:
            return [self]
        first_occurance = self.find(sep)
        if first_occurance == -1:
            return [self]
        ret = [self[: cast(int, first_occurance)]]
        new_maxsplit = -1 if maxsplit < 0 else maxsplit - 1
        ret.extend(
            self[first_occurance + len(sep) :].split(sep=sep, maxsplit=new_maxsplit)
        )
        return ret

    def strip(self, chars=None):
        return self.lstrip(chars).rstrip(chars)

    def swapcase(self):
        with NoTracing():
            space = context_statespace()
            islowerfn = space.extra(UnicodeMaskCache).lower()
            isupperfn = space.extra(UnicodeMaskCache).upper()
        ret = ""
        for char in self:
            codepoint = ord(char)
            with NoTracing():
                smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
                if space.smt_fork(islowerfn(smt_codepoint)):
                    generator = char.upper
                elif space.smt_fork(isupperfn(smt_codepoint)):
                    generator = char.lower
                else:

                    def generator():
                        return char

            ret += generator()
        return ret

    def _title_one_char(self, cache: UnicodeMaskCache, smt_codepoint: z3.ExprRef):
        space = context_statespace()
        smt_1st = cache.totitle_1st()(smt_codepoint)
        if not space.smt_fork(cache.totitle_2nd_exists()(smt_codepoint)):
            return LazyIntSymbolicStr([SymbolicInt(smt_1st)])
        smt_2nd = cache.totitle_2nd()(smt_codepoint)
        if not space.smt_fork(cache.totitle_3rd_exists()(smt_codepoint)):
            return LazyIntSymbolicStr([SymbolicInt(smt_1st), SymbolicInt(smt_2nd)])
        smt_3rd = cache.totitle_3rd()(smt_codepoint)
        return LazyIntSymbolicStr(
            [SymbolicInt(smt_1st), SymbolicInt(smt_2nd), SymbolicInt(smt_3rd)]
        )

    def title(self):
        with NoTracing():
            space = context_statespace()
            unicode_cache = space.extra(UnicodeMaskCache)
            title = unicode_cache.title()
            lower = unicode_cache.lower()
            totitle_exists = unicode_cache.totitle_exists()
            do_upper = True
            ret = ""
        for ch in self:
            codepoint = ord(ch)
            with NoTracing():
                smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
                smt_is_cased = z3.Or(title(smt_codepoint), lower(smt_codepoint))
                if not space.smt_fork(smt_is_cased):
                    ret += ch
                    do_upper = True
                    continue
                if not do_upper:
                    with ResumedTracing():
                        ret += ch.lower()
                    continue
                # Title case this one
                if space.smt_fork(totitle_exists(smt_codepoint)):
                    ret += self._title_one_char(unicode_cache, smt_codepoint)
                else:
                    # Already title cased
                    ret += ch
                do_upper = False
        return ret

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

    def upper(self):
        if len(self) != 1:
            return "".join([ch.upper() for ch in self])
        char = self[0]
        codepoint = ord(char)
        with NoTracing():
            space = context_statespace()
            smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
            cache = space.extra(UnicodeMaskCache)
            if not space.smt_fork(cache.toupper_exists()(smt_codepoint)):
                return char
            smt_1st = cache.toupper_1st()(smt_codepoint)
            if not space.smt_fork(cache.toupper_2nd_exists()(smt_codepoint)):
                return LazyIntSymbolicStr([SymbolicInt(smt_1st)])
            smt_2nd = cache.toupper_2nd()(smt_codepoint)
            if not space.smt_fork(cache.toupper_3rd_exists()(smt_codepoint)):
                return LazyIntSymbolicStr([SymbolicInt(smt_1st), SymbolicInt(smt_2nd)])
            smt_3rd = cache.toupper_3rd()(smt_codepoint)
            return LazyIntSymbolicStr(
                [SymbolicInt(smt_1st), SymbolicInt(smt_2nd), SymbolicInt(smt_3rd)]
            )

    def zfill(self, width):
        if not isinstance(width, int):
            raise TypeError
        fill_length = max(0, width - len(self))
        if self.startswith("+") or self.startswith("-"):
            return self[0] + "0" * fill_length + self[1:]
        else:
            return "0" * fill_length + self


class LazyIntSymbolicStr(AnySymbolicStr, CrossHairValue):
    """
    A symbolic string that lazily generates SymbolicInt-based characters as needed.

    It is backed by a concrete list of (SymbolicInt) codepoints.
    """

    def __init__(
        self, smtvar: Union[str, Sequence[Union[int, SymbolicInt]]], typ: Type = str
    ):
        assert typ == str
        if isinstance(smtvar, str):
            self._codepoints: Sequence[int] = SymbolicBoundedIntTuple(
                0, maxunicode, smtvar
            )
        elif isinstance(
            smtvar, (SymbolicBoundedIntTuple, SliceView, SequenceConcatenation, list)
        ):
            self._codepoints = smtvar
        else:
            raise CrosshairInternal(
                f"Unexpected LazyIntSymbolicStr initializer of type {type(smtvar)}"
            )

    def __ch_realize__(self) -> object:
        with ResumedTracing():
            codepoints = tuple(self._codepoints)
        return "".join(chr(realize(x)) for x in codepoints)

    # This is normally an AtomicSymbolicValue method, but sometimes it's used in a
    # duck-typing way.
    @classmethod
    def _smt_promote_literal(cls, val: object) -> Optional[z3.SortRef]:
        if isinstance(val, str):
            return LazyIntSymbolicStr(list(map(ord, val)))
        return None

    def __hash__(self):
        return hash(self.__str__())

    def __len__(self):
        return self._codepoints.__len__()

    def __contains__(self, other):
        if len(other) == 0:
            return True
        (_, match, _) = self.partition(other)
        return match != ""

    def __eq__(self, other):
        with NoTracing():
            mypoints = self._codepoints
            if isinstance(other, LazyIntSymbolicStr):
                with ResumedTracing():
                    return mypoints == other._codepoints
            elif isinstance(other, str):
                otherpoints = [ord(ch) for ch in other]
                with ResumedTracing():
                    return mypoints.__eq__(otherpoints)
            elif isinstance(other, SeqBasedSymbolicStr):
                with ResumedTracing():
                    otherpoints = [ord(ch) for ch in other]
                    return mypoints.__eq__(otherpoints)
            else:
                return NotImplemented

    def __getitem__(self, i):
        with NoTracing():
            i = deep_realize(i)
            with ResumedTracing():
                newcontents = self._codepoints[i]
            if not isinstance(i, slice):
                newcontents = [newcontents]
            return LazyIntSymbolicStr(newcontents)

    @classmethod
    def _force_into_codepoints(cls, other):
        assert not is_tracing()
        if isinstance(other, LazyIntSymbolicStr):
            return other._codepoints
        elif isinstance(other, (AnySymbolicStr, str)):
            with ResumedTracing():
                return list(map(ord, other))
        else:
            raise TypeError

    def __add__(self, other):
        with NoTracing():
            newpoints = LazyIntSymbolicStr._force_into_codepoints(other)
            return LazyIntSymbolicStr(self._codepoints + newpoints)

    def __radd__(self, other):
        with NoTracing():
            newpoints = LazyIntSymbolicStr._force_into_codepoints(other)
            return LazyIntSymbolicStr(newpoints + self._codepoints)

    def __mul__(self, other):
        if isinstance(other, Integral):
            ret = ""
            while other > 0:
                ret += self
                other -= 1
            return ret
        return NotImplemented

    __rmul__ = __mul__

    def partition(self, substr):
        if not isinstance(substr, str):
            raise TypeError
        if len(substr) == 0:
            raise ValueError
        mypoints = self._codepoints
        subpoints = [ord(ch) for ch in substr]
        if not subpoints:
            raise ValueError
        substrlen = len(subpoints)
        # TODO: We have no intercept to make `range` lazy!
        for start in range(1 + len(mypoints) - substrlen):
            if mypoints[start : start + substrlen] == subpoints:
                prefix_points = mypoints[:start]
                suffix_points = mypoints[start + substrlen :]
                with NoTracing():
                    return (
                        LazyIntSymbolicStr(prefix_points),
                        substr,
                        LazyIntSymbolicStr(suffix_points),
                    )
        return (self, "", "")

    def endswith(self, substr, start=None, end=None):
        if isinstance(substr, tuple):
            return any(self.endswith(s, start, end) for s in substr)
        if not isinstance(substr, str):
            raise TypeError
        if start is None and end is None:
            matchable = self
        else:
            matchable = self[start:end]
        return matchable[-len(substr) :] == substr

    def startswith(self, substr, start=None, end=None):
        if isinstance(substr, tuple):
            return any(self.startswith(s, start, end) for s in substr)
        if not isinstance(substr, str):
            raise TypeError
        if start is None and end is None:
            matchable = self
        else:
            matchable = self[start:end]
        return matchable[: len(substr)] == substr

    def rpartition(self, substr):
        if not isinstance(substr, str):
            raise TypeError
        if len(substr) == 0:
            raise ValueError
        mypoints = self._codepoints
        subpoints = [ord(ch) for ch in substr]
        if not subpoints:
            raise ValueError
        substrlen = len(subpoints)
        start = len(mypoints) - len(subpoints)
        for start in range(start, -1, -1):
            if mypoints[start : start + substrlen] == subpoints:
                prefix_points = mypoints[:start]
                suffix_points = mypoints[start + substrlen :]
                with NoTracing():
                    return (
                        LazyIntSymbolicStr(prefix_points),
                        substr,
                        LazyIntSymbolicStr(suffix_points),
                    )
        return ("", "", self)

    def _find(self, partitioner, substr, start=None, end=None):
        if not isinstance(substr, str):
            raise TypeError
        mylen = len(self)
        if start is None:
            start = 0
        if start < 0:
            start += mylen
        if end is None:
            end = mylen
        if end < 0:
            end += mylen
        matchstr = self[start:end]
        if len(substr) == 0:
            # Add oddity of CPython. We can find the empty string when over-slicing
            # off the left side of the string, but not off the right:
            # ''.find('', 3, 4) == -1
            # ''.find('', -4, -3) == 0
            if matchstr == "" and start > min(mylen, max(end, 0)):
                return -1
            else:
                return max(start, 0)
        else:
            (prefix, match, _) = partitioner(matchstr, substr)
            if match == "":
                return -1
            return start + len(prefix)

    def find(self, substr, start=None, end=None):
        return self._find(LazyIntSymbolicStr.partition, substr, start, end)

    def rfind(self, substr, start=None, end=None):
        return self._find(LazyIntSymbolicStr.rpartition, substr, start, end)


class SeqBasedSymbolicStr(AtomicSymbolicValue, SymbolicSequence, AnySymbolicStr):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = str):
        assert typ == str
        SymbolicValue.__init__(self, smtvar, typ)
        self.item_pytype = str
        if isinstance(smtvar, str):
            # Constrain fresh strings to valid codepoints
            space = context_statespace()
            idxvar = z3.Int("idxvar" + space.uniq())
            z3seq = self.var
            space.add(
                z3.ForAll(
                    [idxvar], z3.And(0 <= z3seq[idxvar], z3seq[idxvar] <= maxunicode)
                )
            )

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
                return z3.Unit(z3IntVal(ord(literal)))
            return z3.Concat([z3.Unit(z3IntVal(ord(ch))) for ch in literal])
        return None

    def __ch_realize__(self) -> object:
        codepoints = self.statespace.find_model_value(self.var)
        return "".join(chr(x) for x in codepoints)

    def __copy__(self):
        return SeqBasedSymbolicStr(self.var)

    def __hash__(self):
        return hash(self.__str__())

    @staticmethod
    def _concat_strings(
        a: Union[str, "SeqBasedSymbolicStr"], b: Union[str, "SeqBasedSymbolicStr"]
    ) -> Union[str, "SeqBasedSymbolicStr"]:
        assert not is_tracing()
        # Assumes at least one argument is symbolic and not tracing
        if isinstance(a, SeqBasedSymbolicStr) and isinstance(b, SeqBasedSymbolicStr):
            return SeqBasedSymbolicStr(a.var + b.var)
        elif isinstance(a, str) and isinstance(b, SeqBasedSymbolicStr):
            return SeqBasedSymbolicStr(
                SeqBasedSymbolicStr._coerce_to_smt_sort(a) + b.var
            )
        else:
            assert isinstance(a, SeqBasedSymbolicStr)
            assert isinstance(b, str)
            return SeqBasedSymbolicStr(
                a.var + SeqBasedSymbolicStr._coerce_to_smt_sort(b)
            )

    def __add__(self, other):
        with NoTracing():
            if isinstance(other, (SeqBasedSymbolicStr, str)):
                return SeqBasedSymbolicStr._concat_strings(self, other)
            if isinstance(other, AnySymbolicStr):
                return NotImplemented
            raise TypeError

    def __radd__(self, other):
        with NoTracing():
            if isinstance(other, (SeqBasedSymbolicStr, str)):
                return SeqBasedSymbolicStr._concat_strings(other, self)
            if isinstance(other, AnySymbolicStr):
                return NotImplemented
            raise TypeError

    def __mul__(self, other):
        self.statespace
        if isinstance(other, Integral):
            if other <= 1:
                return self if other == 1 else ""
            # Note that in SymbolicInt, we attempt string multiplication via regex.
            # Z3 cannot do much with a symbolic regex, so we case-split on
            # the repetition count.
            return SeqBasedSymbolicStr(z3.Concat(*[self.var for _ in range(other)]))
        return NotImplemented

    __rmul__ = __mul__

    def __mod__(self, other):
        return self.__str__() % realize(other)

    def __contains__(self, other):
        forced = force_to_smt_sort(other, SeqBasedSymbolicStr)
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
                smt_result = z3.Unit(self.var[idx_or_pair])
            return SeqBasedSymbolicStr(smt_result)

    def endswith(self, substr):
        smt_substr = force_to_smt_sort(substr, SeqBasedSymbolicStr)
        return SymbolicBool(z3.SuffixOf(smt_substr, self.var))

    def find(self, substr, start=None, end=None):
        if not isinstance(substr, str):
            raise TypeError
        with NoTracing():
            space = self.statespace
            smt_my_len = z3.Length(self.var)
            if start is None and end is None:
                smt_start = z3IntVal(0)
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

            smt_sub = force_to_smt_sort(substr, SeqBasedSymbolicStr)
            if space.smt_fork(z3.Contains(smt_str, smt_sub)):
                return SymbolicInt(z3.IndexOf(smt_str, smt_sub, 0) + smt_start)
            else:
                return -1

    def partition(self, sep: str):
        if not isinstance(sep, str):
            raise TypeError
        if len(sep) == 0:
            raise ValueError
        with NoTracing():
            space = context_statespace()
            smt_str = self.var
            smt_sep = force_to_smt_sort(sep, SeqBasedSymbolicStr)
            if space.smt_fork(z3.Contains(smt_str, smt_sep)):
                uniq = space.uniq()
                # Divide my contents into 4 concatenated parts:
                prefix = SeqBasedSymbolicStr(f"prefix{uniq}")
                match1 = SeqBasedSymbolicStr(
                    f"match1{uniq}"
                )  # the first character of the match
                match_tail = SeqBasedSymbolicStr(f"match_tail{uniq}")
                suffix = SeqBasedSymbolicStr(f"suffix{uniq}")
                space.add(z3.Length(match1.var) == 1)
                space.add(smt_sep == z3.Concat(match1.var, match_tail.var))
                space.add(smt_str == z3.Concat(prefix.var, smt_sep, suffix.var))
                space.add(
                    z3.Not(z3.Contains(z3.Concat(match_tail.var, suffix.var), smt_sep))
                )
                return (prefix, sep, suffix)
            else:
                return (self, "", "")

    def rfind(self, substr, start=None, end=None) -> Union[int, SymbolicInt]:
        if not isinstance(substr, str):
            raise TypeError
        with NoTracing():
            space = self.statespace
            smt_my_len = z3.Length(self.var)
            if start is None and end is None:
                smt_start = z3IntVal(0)
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
            smt_sub = force_to_smt_sort(substr, SeqBasedSymbolicStr)
            if space.smt_fork(z3.Contains(smt_str, smt_sub)):
                uniq = space.uniq()
                # Divide my contents into 4 concatenated parts:
                prefix = SeqBasedSymbolicStr(f"prefix{uniq}")
                match1 = SeqBasedSymbolicStr(f"match1{uniq}")
                match_tail = SeqBasedSymbolicStr(f"match_tail{uniq}")
                suffix = SeqBasedSymbolicStr(f"suffix{uniq}")
                space.add(z3.Length(match1.var) == 1)
                space.add(smt_sub == z3.Concat(match1.var, match_tail.var))
                space.add(smt_str == z3.Concat(prefix.var, smt_sub, suffix.var))
                space.add(
                    z3.Not(z3.Contains(z3.Concat(match_tail.var, suffix.var), smt_sub))
                )
                return SymbolicInt(smt_start + z3.Length(prefix.var))
            else:
                return -1

    def rpartition(self, sep: str):
        result = self.rsplit(sep, maxsplit=1)
        if len(result) == 1:
            return ("", "", self)
        elif len(result) == 2:
            return (result[0], sep, result[1])

    def startswith(self, substr, start=None, end=None):
        if isinstance(substr, tuple):
            return any(self.startswith(s, start, end) for s in substr)
        smt_substr = force_to_smt_sort(substr, SeqBasedSymbolicStr)
        if start is not None or end is not None:
            # TODO: "".startswith("", 1) should be False, not True
            return self[start:end].startswith(substr)
        return SymbolicBool(z3.PrefixOf(smt_substr, self.var))


def buffer_to_byte_seq(obj: object) -> Optional[Sequence[int]]:
    if isinstance(obj, (bytes, bytearray)):
        return list(obj)
    elif isinstance(obj, (array, memoryview)):
        if isinstance(obj, memoryview):
            if obj.ndim > 1 or obj.format != "B":
                return None
        else:
            if obj.typecode != "B":
                return None
        return list(obj)
    elif isinstance(obj, SymbolicBytes):
        return obj.inner
    elif isinstance(obj, SymbolicByteArray):
        return obj.inner
    elif isinstance(obj, SymbolicMemoryView):
        return obj._sliced
    elif isinstance(obj, Sequence):
        return obj
    return None


_ALL_BYTES_TYPES = (bytes, bytearray, memoryview, array)


class BytesLike(collections.abc.ByteString, AbcString, CrossHairValue):
    def __eq__(self, other) -> bool:
        if not isinstance(other, _ALL_BYTES_TYPES):
            return False
        if len(self) != len(other):
            return False
        return list(self) == list(other)

    def _cmp_op(self, other, op) -> bool:
        # Surprisingly, none of (bytes, memoryview, array) are ordered-comparable with
        # the other types.
        # Even more surprisingly, bytearray is comparable with all types.
        other_type = type(other)
        if other_type == type(self) or other_type == bytearray:
            return op(tuple(self), tuple(other))
        else:
            raise TypeError

    def __lt__(self, other):
        return self._cmp_op(other, ops.lt)

    def __le__(self, other):
        return self._cmp_op(other, ops.le)

    def __gt__(self, other):
        return self._cmp_op(other, ops.gt)

    def __ge__(self, other):
        return self._cmp_op(other, ops.ge)

    def __bytes__(self) -> bytes:
        with NoTracing():
            return bytes(tracing_iter(self))

    def __radd__(self, left):
        with NoTracing():
            if isinstance(left, bytes):
                left = SymbolicBytes(left)
            elif isinstance(left, bytearray):
                left = SymbolicByteArray(left)
            else:
                return NotImplemented
            with ResumedTracing():
                return left.__add__(self)

    def __repr__(self):
        return repr(realize(self))


def _bytes_data_prop(s):
    with NoTracing():
        return bytes(s.inner)


class SymbolicBytes(BytesLike):
    def __init__(self, inner):
        with NoTracing():
            inner = buffer_to_byte_seq(inner)
        self.inner = inner

    # TODO: find all uses of str() in AbcString and check SymbolicBytes behavior for
    # those cases.

    data = property(_bytes_data_prop)

    def __ch_realize__(self):
        return bytes(tracing_iter(self.inner))

    def __ch_pytype__(self):
        return bytes

    def __repr__(self):
        # TODO: implement this preserving symbolics. These are the cases:
        # [9]: "\t"
        # [10]: "\n"
        # [13]: "\r"
        # [32-91]: chr(i)
        # [92]: "\\"
        # [93-126]: chr(i)
        # [else]: ["\x00"-"\xff"]
        with NoTracing():
            return repr(self.__ch_realize__())

    def __len__(self):
        return self.inner.__len__()

    def __getitem__(self, i: Union[int, slice]):
        if isinstance(i, slice):
            return SymbolicBytes(self.inner.__getitem__(i))
        else:
            return self.inner.__getitem__(i)

    def __iter__(self):
        return self.inner.__iter__()

    def __copy__(self):
        return SymbolicBytes(self.inner)

    def __add__(self, other):
        with NoTracing():
            byte_seq = buffer_to_byte_seq(other)
            if byte_seq is other:
                # plain numeric sequences can't be added to byte-like objects
                raise TypeError
            if byte_seq is None:
                return self.__ch_realize__().__add__(realize(other))
        return SymbolicBytes(self.inner + byte_seq)

    def decode(self, encoding="utf-8", errors="strict"):
        return codecs.decode(self, encoding, errors=errors)


def make_byte_string(creator: SymbolicFactory):
    return SymbolicBytes(SymbolicBoundedIntTuple(0, 255, creator.varname))


class SymbolicByteArray(BytesLike, ShellMutableSequence):  # type: ignore
    def __init__(self, byte_seq):
        assert not is_tracing()
        byte_seq = buffer_to_byte_seq(byte_seq)
        if byte_seq is None:
            raise TypeError
        super().__init__(byte_seq)

    __hash__ = None  # type: ignore
    data = property(_bytes_data_prop)

    def __ch_realize__(self):
        return bytearray(tracing_iter(self.inner))

    def __ch_pytype__(self):
        return bytearray

    def __len__(self):
        return self.inner.__len__()

    def __getitem__(self, key):
        byte_seq_return = self.inner.__getitem__(key)
        if isinstance(key, slice):
            with NoTracing():
                return SymbolicByteArray(byte_seq_return)
        else:
            return byte_seq_return

    def _cmp_op(self, other, op) -> bool:
        if isinstance(other, _ALL_BYTES_TYPES):
            return op(tuple(self), tuple(other))
        else:
            raise TypeError

    def __add__(self, other):
        with NoTracing():
            byte_seq = buffer_to_byte_seq(other)
            if byte_seq is other:
                # plain numeric sequences can't be added to byte-like objects
                raise TypeError
            if byte_seq is None:
                raise TypeError
            with ResumedTracing():
                byte_seq = self.inner + byte_seq
            return SymbolicByteArray(byte_seq)

    def _spawn(self, items: Sequence) -> ShellMutableSequence:
        return SymbolicByteArray(items)

    def decode(self, encoding="utf-8", errors="strict"):
        return codecs.decode(self, encoding, errors=errors)


class SymbolicMemoryView(BytesLike):
    format = "B"
    itemsize = 1
    ndim = 1
    strides = (1,)
    suboffsets = ()
    c_contiguous = True
    f_contiguous = True
    contiguous = True

    def __init__(self, obj):
        assert not is_tracing()
        if not isinstance(obj, (_ALL_BYTES_TYPES, BytesLike)):
            raise TypeError
        objlen = obj.__len__()
        self.obj = obj
        self.nbytes = objlen
        self.shape = (objlen,)
        self.readonly = isinstance(obj, bytes)
        self._sliced = SliceView(obj, 0, objlen)

    def __ch_realize__(self):
        sliced = self._sliced
        obj, start, stop = self.obj, sliced.start, sliced.stop
        self.obj = obj
        return memoryview(realize(obj))[realize(start) : realize(stop)]

    def __ch_pytype__(self):
        return memoryview

    def _cmp_op(self, other, op) -> bool:
        # memoryview is the only bytes-like type that isn't ordered-comparable with
        # instances of its own type. But it is comparable with bytearrays!
        if isinstance(other, bytearray):
            return op(tuple(self), tuple(other))
        else:
            raise TypeError

    def __hash__(self):
        return hash(self.tobytes())

    def __setitem__(self, key, value):
        if self.readonly:
            raise TypeError
        obj, sliced = self.obj, self._sliced
        suffixlen = len(obj) - sliced.stop
        if isinstance(key, slice):
            key = compose_slices(sliced.start, suffixlen, key)
            if len(value) != len(obj[key]):
                raise ValueError
            obj[key] = value
        elif key < 0:
            obj[key - suffixlen] = value
        else:
            obj[key + sliced.start] = value

    def __getitem__(self, key):
        if isinstance(key, slice):
            newslice = self._sliced[key]
            if isinstance(newslice, SliceView):
                with NoTracing():
                    ret = SymbolicMemoryView(self.obj)
                    ret._sliced = newslice
                    return ret
            else:
                # Give up when there's a step in the slice:
                return realize(self).__getitem__(key)
        else:
            return self._sliced[key]

    def __add__(self, other):
        # Bytes and bytearrays can add a memoryview, but memoryview can't add anything.
        # Yeah, it's asymetric. Shrug!
        raise TypeError

    def __len__(self) -> int:
        return self._sliced.__len__()

    def __iter__(self):
        return self._sliced.__iter__()

    def tobytes(self):
        return SymbolicBytes(self._sliced)

    def hex(self, *a):
        # TODO: consider symbolic version (bytes.hex() too!)
        return realize(self).hex(*a)

    def tolist(self):
        return list(self._sliced)

    def toreadonly(self):
        with NoTracing():
            cpy = copy.copy(self)
            cpy.readonly = True
            return cpy

    def release(self):
        # This is going to be difficult to implement faithfully.
        # The mechanism by which objects track "exports" all happens at the C level.
        pass

    def cast(self, *a):
        return realize(self).cast(*map(realize, a))


_CACHED_TYPE_ENUMS: Dict[FrozenSet[type], z3.SortRef] = {}


_PYTYPE_TO_WRAPPER_TYPE = {
    # These are mappings for AtomicSymbolic values - values that we directly represent
    # as single z3 values.
    bool: (SymbolicBool,),
    int: (SymbolicInt,),
    float: (SymbolicFloat,),
    type: (SymbolicType,),
}

_WRAPPER_TYPE_TO_PYTYPE = dict(
    (v, k) for (k, vs) in _PYTYPE_TO_WRAPPER_TYPE.items() for v in vs
)


#
# Symbolic-making helpers
#


def make_union_choice(creator: SymbolicFactory, *pytypes):
    for typ, probability_true in with_uniform_probabilities(pytypes)[:-1]:
        if creator.space.smt_fork(
            probability_true=probability_true, desc="choose_" + smtlib_typename(typ)
        ):
            return creator(typ)
    return creator(pytypes[-1])


def make_optional_smt(smt_type):
    def make(creator: SymbolicFactory, *type_args):
        space = context_statespace()
        varname, pytype = creator.varname, creator.pytype
        ret = smt_type(creator.varname, pytype)

        premature_stats, symbolic_stats = space.stats_lookahead()
        bad_iters = (
            # SMT unknowns, unsupported, timeouts:
            symbolic_stats[VerificationStatus.UNKNOWN]
            +
            # Or, we ended up realizing this var anyway:
            symbolic_stats[f"realize_{varname}"]
        )
        bad_pct = bad_iters / (symbolic_stats.iterations + 10)
        symbolic_probability = 1.0 - (bad_pct * 0.8)
        if space.fork_parallel(
            false_probability=symbolic_probability, desc=f"premature realize {varname}"
        ):
            debug(
                f"Prematurely realizing",
                pytype,
                f"value (~{1.-symbolic_probability:.1%} chance)",
            )
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


def fork_on_useful_attr_names(obj: object, name: AnySymbolicStr) -> None:
    # This function appears to do nothing at all! However, it exists to
    # force a symbolic string into useful candidate states.
    obj = realize(obj)
    for key in reversed(dir(obj)):
        # We use reverse() above to handle __dunder__ methods last.
        with ResumedTracing():
            # Double negative to make the comparison more likely to succeed:
            if not smt_not(name == key):
                return


_ascii = with_realized_args(ascii)

_bin = with_realized_args(bin)


def _bytearray(*a):
    if len(a) == 1:
        with NoTracing():
            (source,) = a
            byte_seq = buffer_to_byte_seq(source)
            if byte_seq is not None:
                # We make all bytearrays symbolic when possible.
                # (concrete bytearrays are impossible to mutate symbolically)
                return SymbolicByteArray(byte_seq)
    return bytearray(*map(realize, a))


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
            source = list(tracing_iter(source))
            if any(isinstance(i, SymbolicIntable) for i in source):
                return SymbolicBytes(source)
        return bytes(source)


_callable = with_realized_args(callable)


def _chr(i: int) -> Union[str, LazyIntSymbolicStr]:
    if i < 0 or i > 0x10FFFF:
        raise ValueError
    with NoTracing():
        if isinstance(i, SymbolicInt):
            return LazyIntSymbolicStr([i])
    return chr(realize(i))


def _eval(expr: str, _globals=None, _locals=None) -> object:
    # This is fragile: consider detecting _crosshair_wrapper(s):
    calling_frame = sys._getframe(1)
    _globals = calling_frame.f_globals if _globals is None else realize(_globals)
    _locals = calling_frame.f_locals if _locals is None else realize(_locals)
    return eval(realize(expr), _globals, _locals)


def _format(obj: object, format_spec: str = "") -> Union[str, AnySymbolicStr]:
    with NoTracing():
        if isinstance(format_spec, AnySymbolicStr):
            format_spec = realize(format_spec)
        if format_spec in ("", "s") and isinstance(obj, AnySymbolicStr):
            return obj
        obj = deep_realize(obj)
        result = invoke_dunder(obj, "__format__", format_spec)
        if result is not _MISSING:
            return result
    return format(obj, format_spec)


def _getattr(obj: object, name: str, default=_MISSING) -> object:
    with NoTracing():
        if isinstance(name, AnySymbolicStr):
            fork_on_useful_attr_names(obj, name)  # type:ignore
            name = realize(name)
        if default is _MISSING:
            return getattr(obj, name)
        else:
            return getattr(obj, name, default)


def _hasattr(obj: object, name: str) -> bool:
    with NoTracing():
        if isinstance(name, AnySymbolicStr):
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
    return invoke_dunder(obj, "__hash__")


def _int(val: object = 0, *a):
    with NoTracing():
        if isinstance(val, SymbolicInt):
            return val
        if isinstance(val, AnySymbolicStr) and a == ():
            with ResumedTracing():
                if not val:
                    return int(realize(val))
                ord_zero = ord("0")
                ret = 0
                for ch in val:
                    ch_num = ord(ch) - ord_zero
                    if ch_num < 0 or ch_num > 9:
                        # TODO parse other digits with data from unicodedata.decimal()
                        return int(realize(val))
                    else:
                        ret = (ret * 10) + ch_num
                return ret
        # TODO: add symbolic handling when val is float (if possible)
        return int(realize(val), *realize(a))


_FLOAT_REGEX = re.compile(
    r"""
      (?P<posneg> (\+|\-|))
      (?P<intpart>(\d+))
      (\.(?P<fraction>\d*))?
""",
    re.VERBOSE | re.ASCII,
    # (ASCII because we only bother to handle the easy cases symbolically)
)
# TODO handle exponents: ((e|E)(\+|\-|) (\d+)(\.\d+)?)?
# TODO allow underscores (only in between digits)
# TODO handle special floats (nan, inf, -inf)
# TODO once regex is perfect, return ValueError directly, instead of realizing input
#      (this is important because realization impacts search heuristics)


def _filter(fn, *iters):
    # Wrap the `filter` callback in a pure Python lambda.
    # This de-optimization ensures that the callback can be intercepted.
    return filter(lambda x: fn(x), *iters)


def _float(val=0.0):
    with NoTracing():
        if isinstance(val, SymbolicFloat):
            return val
        is_symbolic_str = isinstance(val, AnySymbolicStr)
        is_symbolic_int = isinstance(val, SymbolicInt)
    if is_symbolic_str:
        match = _FLOAT_REGEX.fullmatch(val)
        if match:
            ret = _float(int(match.group("intpart")))
            decimal_digits = match.group("fraction")
            if decimal_digits:
                denominator = realize(len(decimal_digits))
                ret += _float(int(decimal_digits)) / (10**denominator)
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
            raise TypeError
        if type(superclass) is tuple:
            for cur_super in superclass:
                if _issubclass(subclass, cur_super):
                    return True
            return False
        if isinstance(subclass, SymbolicType) or isinstance(superclass, SymbolicType):
            if isinstance(superclass, SymbolicType):
                method = superclass._is_superclass_of_
                if (
                    method(subclass)
                    if hasattr(method, "__self__")
                    else method(subclass, superclass)
                ):
                    return True
            if isinstance(subclass, SymbolicType):
                method = subclass._is_subclass_of_
                return (
                    method(superclass)
                    if hasattr(method, "__self__")
                    else method(subclass, superclass)
                )
            return False
    return issubclass(subclass, superclass)


def _isinstance(obj, types):
    return issubclass(type(obj), types)


# CPython's len() forces the return value to be a native integer.
# Avoid that requirement by making it only call __len__().
def _len(ls):
    return ls.__len__() if hasattr(ls, "__len__") else [x for x in ls].__len__()


def _map(fn, *iters):
    # Wrap the `map` callback in a pure Python lambda.
    # This de-optimization ensures that the callback can be intercepted.
    return map(lambda x: fn(x), *iters)


def _memoryview(source):
    with NoTracing():
        if isinstance(source, CrossHairValue):
            return SymbolicMemoryView(source)
    return memoryview(source)


def _ord(c: str) -> int:
    if len(c) != 1:
        raise TypeError
    with NoTracing():
        if isinstance(c, LazyIntSymbolicStr):
            return c._codepoints[0]
        elif isinstance(c, SeqBasedSymbolicStr):
            space = context_statespace()
            ret = SymbolicInt("ord" + space.uniq())
            space.add(c.var == z3.Unit(ret.var))
            return ret
    return ord(realize(c))


def _pow(base, exp, mod=None):
    return pow(realize(base), realize(exp), realize(mod))


def _print(*a: object, **kw: Any) -> None:
    print(*deep_realize(a), **deep_realize(kw))


def _repr(obj: object) -> str:
    """
    post[]: True
    """
    # Skip the built-in repr if possible, because it requires the output
    # to be a native string:
    return invoke_dunder(obj, "__repr__")


def _setattr(obj: object, name: str, value: object) -> None:
    # TODO: we could do symbolic stuff like getattr does here!
    with NoTracing():
        if isinstance(obj, SymbolicValue):
            obj = realize(obj)
        if type(name) is AnySymbolicStr:
            name = realize(name)
        return setattr(obj, name, value)


# TODO: is this important? Feels like the builtin might do the same?
def _sorted(ls, key=None, reverse=False):
    if not is_iterable(ls):
        raise TypeError("object is not iterable")
    ret = list(ls.__iter__())
    ret.sort(key=key, reverse=realize(reverse))
    return ret


# TODO: consider what to do here
# def sum(i: Iterable[_T]) -> Union[_T, int]:
#    '''
#    post[]: _ == 0 or len(i) > 0
#    '''
#    return sum(i)


def _type(*a) -> type:
    with NoTracing():
        if len(a) == 1:
            return python_type(a[0])
    return type(*map(deep_realize, a))


#
# Patches on builtin classes
#


def _int_from_bytes(
    b: bytes, byteorder: Union[str, _Missing] = _MISSING, *, signed=False
) -> int:
    if byteorder is _MISSING:
        # byteorder defaults to "big" as of 3.11
        if sys.version_info >= (3, 11):
            byteorder = "big"
        else:
            raise TypeError
    if not isinstance(byteorder, str):
        raise TypeError
    if byteorder == "big":
        little = False
    elif byteorder == "little":
        little = True
    else:
        raise ValueError
    if not is_iterable(b):
        raise TypeError
    byteitr: Iterable[int] = reversed(b) if little else b
    val = 0
    invert = None
    realize(len(b))
    for byt in byteitr:
        if not hasattr(byt, "__index__"):
            raise TypeError
        if invert is None and signed and byt >= 128:
            invert = True
        val = (val * 256) + byt
    if invert:
        val -= 256 ** realize(len(b))
    return val


def _list_index(self, value, start=0, stop=9223372036854775807):
    return list.index(self, value, realize(start), realize(stop))


def _dict_get(self: dict, key, default=None):
    # Special handling for when concrete dict might be indexed by a symbolic key:
    with NoTracing():
        # We might check for CrossHairValue, but we also want to cover cases where the
        # key is, for instance, a tuple with symbolic contents. Err on the side of
        # assuming the key is symbolic.
        if not isinstance(key, (int, float, str)):
            if not isinstance(self, dict):
                raise TypeError
            symbolic_self = SimpleDict(list(self.items()))
            with ResumedTracing():
                return symbolic_self.get(key, default)
    return dict.get(self, key, default)


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


def _str(*a) -> Union[str, AnySymbolicStr]:
    with NoTracing():
        if len(a) == 1:
            (self,) = a
            if isinstance(self, AnySymbolicStr):
                return self
            with ResumedTracing():
                return invoke_dunder(self, "__str__")
    return str(*a)


def _str_join(self, itr) -> str:
    return _join(self, itr, self_type=str, item_type=str)


def _bytes_join(self, itr) -> str:
    return _join(self, itr, self_type=bytes, item_type=collections.abc.ByteString)


def _bytearray_join(self, itr) -> str:
    return _join(self, itr, self_type=bytearray, item_type=collections.abc.ByteString)


def _str_format(self, *a, **kw) -> Union[AnySymbolicStr, str]:
    template = realize(self)
    return string.Formatter().format(template, *a, **kw)


def _str_format_map(self, map) -> Union[AnySymbolicStr, str]:
    template = realize(self)
    return string.Formatter().vformat(template, (), map)


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
        symbolic_self = LazyIntSymbolicStr([ord(c) for c in self])
    return symbolic_self.startswith(substr, start, end)


def _str_contains(
    self: str, other: Union[str, AnySymbolicStr]
) -> Union[bool, SymbolicBool]:
    with NoTracing():
        if not isinstance(self, str):
            raise TypeError
        if not isinstance(other, AnySymbolicStr):
            return self.__contains__(other)
        len_to_find = realize(other.__len__())
        my_codepoints = [ord(c) for c in self]
        num_options = len(self) + 1 - len_to_find
        with ResumedTracing():
            other_codepoints: List = [ord(c) for c in other]
        other_codepoints = list(map(SymbolicInt._coerce_to_smt_sort, other_codepoints))
        codepoint_options = [
            my_codepoints[i : i + len_to_find] for i in range(num_options)
        ]
        conjunctions = (
            z3.And(*(cp1 == cp2 for (cp1, cp2) in zip(other_codepoints, cps)))
            for cps in codepoint_options
        )
        return SymbolicBool(z3.Or(*conjunctions))


#
# Registrations
#


def make_registrations():

    register_type(Union, make_union_choice)

    if sys.version_info >= (3, 8):
        from typing import Final

        register_type(Final, lambda p, t: p(t))

    # Types modeled in the SMT solver:

    register_type(NoneType, lambda *a: None)
    register_type(bool, make_optional_smt(SymbolicBool))
    register_type(int, make_optional_smt(SymbolicInt))
    register_type(float, make_optional_smt(SymbolicFloat))
    register_type(str, make_optional_smt(LazyIntSymbolicStr))
    register_type(list, make_optional_smt(SymbolicList))
    register_type(dict, make_dictionary)
    register_type(tuple, make_tuple)
    register_type(set, make_set)
    register_type(frozenset, make_optional_smt(SymbolicFrozenSet))
    register_type(type, make_optional_smt(SymbolicType))
    register_type(
        collections.abc.Callable,
        lambda p, *t: SymbolicCallable(p(List.__getitem__(t[1] if t else object))),
    )

    # Most types are not directly modeled in the solver, rather they are built
    # on top of the modeled types. Such types are enumerated here:

    register_type(object, lambda p: SymbolicObject(p.varname, object))
    # TODO: Need a symbolic version of complex (currently, this realizes immediately):
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
    register_type(memoryview, lambda p: SymbolicMemoryView(p(bytearray)))
    # AnyStr,  (it's a type var)

    register_type(typing.BinaryIO, lambda p: io.BytesIO(p(bytes)))
    # TODO: handle Any/AnyStr with a custom class that accepts str/bytes interchangably?:
    register_type(
        typing.IO, lambda p, t=Any: p(BinaryIO) if t == "bytes" else p(TextIO)
    )

    register_type(SupportsAbs, lambda p: p(int))
    register_type(SupportsFloat, lambda p: p(float))
    register_type(SupportsInt, lambda p: p(int))
    register_type(SupportsRound, lambda p: p(float))
    register_type(SupportsBytes, lambda p: p(ByteString))
    register_type(SupportsComplex, lambda p: p(complex))

    # Patches

    register_patch(ascii, _ascii)
    register_patch(bin, _bin)
    register_patch(callable, _callable)
    register_patch(chr, _chr)
    register_patch(eval, _eval)
    register_patch(filter, _filter)
    register_patch(format, _format)
    register_patch(getattr, _getattr)
    register_patch(hasattr, _hasattr)
    register_patch(hash, _hash)
    register_patch(hex, with_realized_args(hex))
    register_patch(isinstance, _isinstance)
    register_patch(issubclass, _issubclass)
    register_patch(len, _len)
    register_patch(ord, _ord)
    register_patch(map, _map)
    register_patch(pow, _pow)
    register_patch(print, _print)
    register_patch(repr, _repr)
    register_patch(setattr, _setattr)
    register_patch(sorted, _sorted)
    register_patch(type, _type)

    # Patches on constructors
    register_patch(bytearray, _bytearray)
    register_patch(bytes, _bytes)
    register_patch(float, _float)
    register_patch(int, _int)
    register_patch(memoryview, _memoryview)

    # Patches on str
    # Note that we even patch methods with no arguments like str.isspace() - this
    # handles (unlikely) situations like str.isspace(<symbolic string>).
    names_to_str_patch = [
        "capitalize",
        "casefold",
        "center",
        "count",
        "endswith",
        "expandtabs",
        "find",
        # TODO patch str.format str.format_map?
        "index",
        "isalnum",
        "isalpha",
        "isascii",
        "isdecimal",
        "isdigit",
        "isidentifier",
        "islower",
        "isnumeric",
        "isprintable",
        "isspace",
        "istitle",
        "isupper",
        # TODO patch str.join?
        "ljust",
        "lower",
        "lstrip",
        # TODO: patch makestrans?
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
        # TODO: patch startswith?
        "strip",
        "swapcase",
        "title",
        "translate",
        "upper",
        "zfill",
    ]
    if sys.version_info >= (3, 9):
        names_to_str_patch.append("removeprefix")
        names_to_str_patch.append("removesuffix")
    for name in names_to_str_patch:
        assert hasattr(str, name), f"'{name}' not on str"
        orig_impl = getattr(str, name)
        register_patch(orig_impl, with_symbolic_self(LazyIntSymbolicStr, orig_impl))
        if hasattr(bytes, name):
            bytes_orig_impl = getattr(bytes, name)
            register_patch(bytes_orig_impl, with_realized_args(bytes_orig_impl))

    register_patch(str, _str)
    register_patch(str.encode, with_realized_args(str.encode))
    register_patch(str.format, _str_format)
    register_patch(str.format_map, _str_format_map)
    register_patch(str.startswith, _str_startswith)
    register_patch(str.__contains__, _str_contains)
    register_patch(str.join, _str_join)

    # Patches on bytes
    register_patch(bytes.join, _bytes_join)

    # Patches on bytearrays
    register_patch(bytearray.join, _bytearray_join)

    # Patches on list
    register_patch(list.index, _list_index)

    # Patches on dict
    register_patch(dict.get, _dict_get)
    # TODO: dict.update (concrete w/ symbolic argument), __getitem__, & more?

    # Patches on int
    register_patch(int.from_bytes, _int_from_bytes)

    # Patches on float
    register_patch(float.fromhex, with_realized_args(float.fromhex))

    setup_binops()
