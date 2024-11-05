import codecs
import collections
import copy
import enum
import io
import operator as ops
import os
import re
import string
import sys
import typing
import warnings
from abc import ABCMeta
from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import zip_longest
from math import inf, isfinite, isinf, isnan, nan
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
    MutableMapping,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Sized,
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

try:
    # For reasons unknown, different distributions of z3 4.13.0 use differnt names.
    from z3 import fpEQ
except ImportError:
    from z3 import FfpEQ as fpEQ

from crosshair.abcstring import AbcString
from crosshair.core import (
    SymbolicFactory,
    deep_realize,
    iter_types,
    normalize_pytype,
    proxy_for_type,
    python_type,
    realize,
    register_patch,
    register_type,
    with_checked_self,
    with_realized_args,
    with_symbolic_self,
    with_uniform_probabilities,
)
from crosshair.objectproxy import ObjectProxy
from crosshair.simplestructs import (
    FrozenSetBase,
    LinearSet,
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
    optional_context_statespace,
    prefer_true,
)
from crosshair.tracers import (
    NoTracing,
    ResumedTracing,
    Untracable,
    is_tracing,
    tracing_iter,
)
from crosshair.type_repo import PYTYPE_SORT, SymbolicTypeRepository
from crosshair.unicode_categories import UnicodeMaskCache
from crosshair.util import (
    ATOMIC_IMMUTABLE_TYPES,
    CrossHairInternal,
    CrosshairUnsupported,
    CrossHairValue,
    IdKeyedDict,
    IgnoreAttempt,
    assert_tracing,
    ch_stack,
    debug,
    is_hashable,
    is_iterable,
    memo,
    name_of_type,
    smtlib_typename,
    type_arg_of,
)
from crosshair.z3util import z3And, z3Eq, z3Ge, z3Gt, z3IntVal, z3Or

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as BufferAbc
else:
    from collections.abc import ByteString as BufferAbc


_T = TypeVar("_T")
_VT = TypeVar("_VT")


class _Missing(enum.Enum):
    value = 0


_LIST_INDEX_START_DEFAULT = 0
_LIST_INDEX_STOP_DEFAULT = 9223372036854775807
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
    if hasattr(typ, "__origin__"):
        return typ.__origin__
    return typ


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
        if is_tracing():
            raise CrossHairInternal
        self.snapshot = SnapshotRef(-1)
        self.python_type = typ
        if type(smtvar) is str:
            self.var = self.__init_var__(typ, smtvar)
        else:
            self.var = smtvar
            # TODO test that smtvar's sort matches expected?

    def __init_var__(self, typ, varname):
        raise CrossHairInternal(f"__init_var__ not implemented in {type(self)}")

    def __deepcopy__(self, memo):
        result = copy.copy(self)
        result.snapshot = context_statespace().current_snapshot()
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
        with NoTracing():
            return self.__class__(op(self.var), self.python_type)


class AtomicSymbolicValue(SymbolicValue):
    def __init_var__(self, typ, varname):
        if is_tracing():
            raise CrossHairInternal("Tracing while creating symbolic")
        z3type = self.__class__._ch_smt_sort()
        return z3.Const(varname, z3type)

    def __ch_is_deeply_immutable__(self) -> bool:
        return True

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        raise CrossHairInternal(f"_ch_smt_sort not implemented in {cls}")

    @classmethod
    def _pytype(cls) -> Type:
        # TODO: unify this with __ch_pytype__()? (this is classmethod though)
        raise CrossHairInternal(f"_pytype not implemented in {cls}")

    @classmethod
    def _smt_promote_literal(cls, val: object) -> Optional[z3.SortRef]:
        raise CrossHairInternal(f"_smt_promote_literal not implemented in {cls}")

    @classmethod
    @assert_tracing(False)
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
    type, Tuple[Tuple[Type[AtomicSymbolicValue], float], ...]
] = {}  # to be populated later


def crosshair_types_for_python_type(
    typ: Type,
) -> Tuple[Tuple[Type[AtomicSymbolicValue], float], ...]:
    typ = normalize_pytype(typ)
    origin = origin_of(typ)
    return _PYTYPE_TO_WRAPPER_TYPE.get(origin, ())


class ModelingDirector:
    def __init__(self, *a) -> None:
        # Maps python type to the symbolic type we've chosen to represent it (on this iteration)
        self.global_representations: MutableMapping[
            type, Optional[Type[AtomicSymbolicValue]]
        ] = IdKeyedDict()

    def get(self, typ: Type) -> Optional[Type[AtomicSymbolicValue]]:
        representation_type = self.global_representations.get(typ, _MISSING)
        if representation_type is _MISSING:
            ch_types = crosshair_types_for_python_type(typ)
            if not ch_types:
                representation_type = None
            elif len(ch_types) == 1:
                representation_type = ch_types[0][0]
            else:
                space = context_statespace()
                for option, probability in ch_types[:-1]:
                    # NOTE: fork_parallel() is closer to what we want than smt_fork();
                    # however, exhausting an incomplete representation path
                    # (e.g. RealBasedSymbolicFloat) should not exhaust the branch.
                    if space.smt_fork(
                        desc=f"use_{option.__name__}", probability_true=probability
                    ):
                        representation_type = option
                        break
                else:
                    representation_type = ch_types[-1][0]
            self.global_representations[typ] = representation_type
        return representation_type

    def choose(self, typ: Type) -> Type[AtomicSymbolicValue]:
        chosen = self.get(typ)
        if chosen is None:
            raise CrossHairInternal(f"No symbolics for {typ}")
        return chosen


@assert_tracing(False)
def smt_to_ch_value(
    space: StateSpace, snapshot: SnapshotRef, smt_val: z3.ExprRef, pytype: type
) -> object:
    def proxy_generator(typ: Type) -> object:
        return proxy_for_type(
            typ, smtlib_typename(typ) + "_inheap" + space.uniq(), allow_subtypes=True
        )

    if isinstance(pytype, SymbolicType):
        pytype = realize(pytype)

    if smt_val.sort() == HeapRef:
        return space.find_key_in_heap(smt_val, pytype, proxy_generator, snapshot)
    ch_type = space.extra(ModelingDirector).choose(pytype)
    return ch_type(smt_val, pytype)


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
    def get_finite_comparable(self, other: Union[FiniteFloat, "SymbolicFloat"]):
        # These three cases help cover operations like `a * -inf` which is either
        # positive of negative infinity depending on the sign of `a`.
        if isinstance(other, FiniteFloat):
            comparable: Union[float, SymbolicFloat] = other.val
        else:
            comparable = other
        if comparable > 0:  # type: ignore
            return 1
        elif comparable < 0:
            return -1
        else:
            return 0


def numeric_binop(op: BinFn, a: Number, b: Number):
    if not is_tracing():
        raise CrossHairInternal("Numeric operation on symbolic while not tracing")
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
    with ResumedTracing():  # TODO: <-- can we instead selectively resume? Am I only doing this to satisfy the binop tracing check?
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
_VALID_OPS_ON_COMPLEX_TYPES: Set[BinFn] = {
    ops.eq,
    ops.ne,
    ops.add,
    ops.sub,
    ops.mul,
    ops.truediv,
    ops.pow,
}


def apply_smt(op: BinFn, x: z3.ExprRef, y: z3.ExprRef) -> z3.ExprRef:
    # Mostly, z3 overloads operators and things just work.
    # But some special cases need to be checked first.
    # TODO: we should investigate using the op override mechanism to
    # dispatch to the right SMT operations.
    space = context_statespace()
    if op in _ARITHMETIC_OPS:
        if op in (ops.truediv, ops.floordiv, ops.mod):
            iszero = (fpEQ(y, 0.0)) if isinstance(y, z3.FPRef) else (y == 0)
            if space.smt_fork(iszero):
                raise ZeroDivisionError
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
                if space.smt_fork(z3.Or(y >= 0, x % y == 0)):
                    return x % y
                else:
                    return (x % y) + y
        elif op == ops.pow:
            if space.smt_fork(z3.And(x == 0, y < 0)):
                raise ZeroDivisionError("zero cannot be raised to a negative power")
            if x.is_int() and y.is_int():
                return z3.ToInt(op(x, y))
    return op(x, y)


_ARITHMETIC_AND_COMPARISON_OPS = _ARITHMETIC_OPS.union(_COMPARISON_OPS)
_ALL_OPS = _ARITHMETIC_AND_COMPARISON_OPS.union(_BITWISE_OPS)


def setup_binops():
    # Lower entries take precendence when searching.

    # We check NaN and infitity immediately;
    # RealBasedSymbolicFloats don't support these cases.
    def _(a: Real, b: float):
        if isfinite(b):
            return (a, FiniteFloat(b))  # type: ignore
        return (a, NonFiniteFloat(b))

    setup_promotion(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # Almost all operators involving booleans should upconvert to integers.
    def _(a: SymbolicBool, b: Number):
        with NoTracing():
            return (SymbolicInt(z3.If(a.var, 1, 0)), b)

    setup_promotion(_, _ALL_OPS)

    # Implicitly upconvert symbolic ints to floats.
    def _(a: SymbolicInt, b: Union[float, FiniteFloat, SymbolicFloat, complex]):
        with NoTracing():
            return (a.__float__(), b)

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
        if op not in _VALID_OPS_ON_COMPLEX_TYPES:
            raise TypeError
        return op(complex(a), b)  # type: ignore

    setup_binop(_, _ALL_OPS)

    # float
    def _(op: BinFn, a: SymbolicFloat, b: KindedFloat):
        with NoTracing():
            symbolic_type = context_statespace().extra(ModelingDirector).choose(float)
            bval = symbolic_type._smt_promote_literal(b.val)
            return SymbolicBool(apply_smt(op, a.var, bval))

    setup_binop(_, _COMPARISON_OPS)

    def _(op: BinFn, a: SymbolicFloat, b: KindedFloat):
        with NoTracing():
            symbolic_type = context_statespace().extra(ModelingDirector).choose(float)
            bval = symbolic_type._smt_promote_literal(b.val)
            return symbolic_type(apply_smt(op, a.var, bval), float)

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: KindedFloat, b: SymbolicFloat):
        with NoTracing():
            symbolic_type = context_statespace().extra(ModelingDirector).choose(float)
            aval = symbolic_type._smt_promote_literal(a.val)
            return symbolic_type(apply_smt(op, aval, b.var), float)

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: SymbolicFloat, b: SymbolicFloat):
        with NoTracing():
            symbolic_type = context_statespace().extra(ModelingDirector).choose(float)
            return symbolic_type(apply_smt(op, a.var, b.var), float)

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: SymbolicFloat, b: SymbolicFloat):
        with NoTracing():
            return SymbolicBool(apply_smt(op, a.var, b.var))

    setup_binop(_, _COMPARISON_OPS)

    def _(op: BinFn, a: Union[FiniteFloat, RealBasedSymbolicFloat], b: NonFiniteFloat):
        return op(b.get_finite_comparable(a), b.val)

    setup_binop(_, _ARITHMETIC_AND_COMPARISON_OPS)

    def _(op: BinFn, a: NonFiniteFloat, b: Union[FiniteFloat, RealBasedSymbolicFloat]):
        return op(a.val, a.get_finite_comparable(b))

    setup_binop(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # def _(
    #     op: BinFn, a: NonFiniteFloat, b: NonFiniteFloat
    # ):  # TODO: isn't this impossible (one must be symbolic)?
    #     return op(a.val, b.val)  # type: ignore

    # setup_binop(_, _ARITHMETIC_AND_COMPARISON_OPS)

    # int
    def _(op: BinFn, a: SymbolicInt, b: SymbolicInt):
        with NoTracing():
            return SymbolicInt(apply_smt(op, a.var, b.var))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: SymbolicInt, b: SymbolicInt):
        with NoTracing():
            return SymbolicBool(apply_smt(op, a.var, b.var))

    setup_binop(_, _COMPARISON_OPS)

    def _(op: BinFn, a: SymbolicInt, b: int):
        with NoTracing():
            return SymbolicInt(apply_smt(op, a.var, z3IntVal(b)))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: int, b: SymbolicInt):
        with NoTracing():
            return SymbolicInt(apply_smt(op, z3IntVal(a), b.var))

    setup_binop(_, _ARITHMETIC_OPS)

    def _(op: BinFn, a: SymbolicInt, b: int):
        with NoTracing():
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
    # Floor division over ints requires realization, at present:
    def _(op: BinFn, a: Integral, b: Integral):
        return op(a.__index__(), b.__index__())  # type: ignore

    setup_binop(_, {ops.truediv})

    def _(a: SymbolicInt, b: Number):  # Division over ints must produce float
        return (a.__float__(), b)

    setup_promotion(_, {ops.truediv})

    # TODO : precise float divmod
    def _float_divmod(
        a: Union[NonFiniteFloat, FiniteFloat, RealBasedSymbolicFloat],
        b: Union[NonFiniteFloat, FiniteFloat, RealBasedSymbolicFloat],
    ):
        with NoTracing():
            space = context_statespace()
            bval = RealBasedSymbolicFloat._coerce_to_smt_sort(a)
            # division by zero is checked first
            if not isinstance(b, NonFiniteFloat):
                smt_b = (
                    RealBasedSymbolicFloat._smt_promote_literal(b.val)
                    if isinstance(b, FiniteFloat)
                    else b.var
                )
                if space.smt_fork(smt_b == 0):
                    raise ZeroDivisionError
            # then the non-finite cases:
            if isinstance(a, NonFiniteFloat):
                return (nan, nan)
            smt_a = (
                RealBasedSymbolicFloat._smt_promote_literal(a.val)
                if isinstance(a, FiniteFloat)
                else a.var
            )
            if isinstance(b, NonFiniteFloat):
                if isnan(b.val):
                    return (nan, nan)
                # Deduced the rules for infinitiy based on experimentation!:
                positive_a = space.smt_fork(smt_a >= 0)
                positive_b = b.val == inf
                if positive_a ^ positive_b:
                    if space.smt_fork(smt_a == 0):
                        return (0.0, 0.0) if positive_b else (-0.0, -0.0)
                    return (-1.0, b.val)
                else:
                    return (0.0, a.val if isinstance(a, FiniteFloat) else a)

            remainder = z3.Real(f"remainder{space.uniq()}")
            modproduct = z3.Int(f"modproduct{space.uniq()}")
            # From https://docs.python.org/3.3/reference/expressions.html#binary-arithmetic-operations:
            # The modulo operator always yields a result with the same sign as its second operand (or zero).
            # absolute value of the result is strictly smaller than the absolute value of the second operand.
            space.add(smt_b * modproduct + remainder == smt_a)
            if space.smt_fork(smt_b > 0):
                space.add(remainder >= 0)
                space.add(remainder < smt_b)
            else:
                space.add(remainder <= 0)
                space.add(smt_b < remainder)
            return (SymbolicInt(modproduct), RealBasedSymbolicFloat(remainder))

    def _(
        op: BinFn,
        a: Union[NonFiniteFloat, FiniteFloat, RealBasedSymbolicFloat],
        b: Union[NonFiniteFloat, FiniteFloat, RealBasedSymbolicFloat],
    ):
        return _float_divmod(a, b)[1]

    setup_binop(_, {ops.mod})

    def _(
        op: BinFn,
        a: Union[NonFiniteFloat, FiniteFloat, RealBasedSymbolicFloat],
        b: Union[NonFiniteFloat, FiniteFloat, RealBasedSymbolicFloat],
    ):
        return _float_divmod(a, b)[0]

    setup_binop(_, {ops.floordiv})

    # bool
    def _(op: BinFn, a: SymbolicBool, b: SymbolicBool):
        with NoTracing():
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

    def __ne__(self, other):
        return numeric_binop(ops.ne, self, other)

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
        with NoTracing():
            return context_statespace().choose_possible(self.var)

    def __abs__(self):
        with NoTracing():
            return SymbolicInt(z3.If(self.var, 1, 0))

    def __neg__(self):
        with NoTracing():
            return SymbolicInt(z3.If(self.var, -1, 0))

    def __repr__(self):
        return self.__bool__().__repr__()

    def __hash__(self):
        return self.__bool__().__hash__()

    def __index__(self):
        with NoTracing():
            return SymbolicInt(z3.If(self.var, 1, 0))

    def __bool__(self):
        with NoTracing():
            return context_statespace().choose_possible(self.var)

    def __int__(self):
        with NoTracing():
            return SymbolicInt(z3.If(self.var, 1, 0))

    def __float__(self):
        with NoTracing():
            symbolic_type = context_statespace().extra(ModelingDirector).choose(float)
            smt_false = symbolic_type._coerce_to_smt_sort(0)
            smt_true = symbolic_type._coerce_to_smt_sort(1)
            return symbolic_type(z3.If(self.var, smt_true, smt_false))

    def __complex__(self):
        with NoTracing():
            return complex(self.__float__())

    def __round__(self, ndigits=None):
        # This could be smarter, but nobody rounds a bool right?:
        return round(realize(self), realize(ndigits))


class SymbolicInt(SymbolicIntable, AtomicSymbolicValue):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = int):
        assert typ == int
        if (not isinstance(smtvar, str)) and (not smtvar.is_int()):
            raise CrossHairInternal(
                f"non-integer SMT value given to SymbolicInt ({smtvar})"
            )
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
        return context_statespace().find_model_value(self.var)

    def __repr__(self):
        if self < 0:
            return "-" + (-self).__repr__()
        if self < 10:
            return LazyIntSymbolicStr([48 + self])
        codepoints = [48 + (self % 10)]
        cur_divisor = 10
        while True:
            leftover = self // cur_divisor
            if leftover == 0:
                break
            codepoints.append(48 + (leftover % 10))
            cur_divisor *= 10
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
        with NoTracing():
            symbolic_type = context_statespace().extra(ModelingDirector).choose(float)
            if symbolic_type is RealBasedSymbolicFloat:
                return RealBasedSymbolicFloat(z3.ToReal(self.var))
            elif symbolic_type is PreciseIeeeSymbolicFloat:
                # TODO: We can likely do better with: int -> bit vector -> float
                return PreciseIeeeSymbolicFloat(
                    z3.fpRealToFP(
                        z3.RNE(), z3.ToReal(self.var), _PRECISE_IEEE_FLOAT_SORT
                    )
                )
            else:
                raise CrossHairInternal

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
        with NoTracing():
            return SymbolicBool(
                self.var != 0
            ).__bool__()  # TODO: can we leave this symbolic?

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

    if sys.version_info >= (3, 12):

        def is_integer(self):
            return True

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
        with NoTracing():
            intarray = [
                SymbolicInt((self.var / (2 ** (i * 8))) % 256) for i in range(length)
            ]
            if realize(byteorder) == "big":
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


class SymbolicFloat(SymbolicNumberAble, AtomicSymbolicValue):
    @classmethod
    def _pytype(cls) -> Type:
        return float

    def __ch_realize__(self) -> object:
        return context_statespace().find_model_value(self.var).__float__()  # type: ignore

    def __repr__(self):
        return context_statespace().find_model_value(self.var).__repr__()

    def __hash__(self):
        return context_statespace().find_model_value(self.var).__hash__()

    def __bool__(self):
        with NoTracing():
            return SymbolicBool(self.var != 0).__bool__()

    def __float__(self):
        with NoTracing():
            return self.__ch_realize__()

    def __complex__(self):
        with NoTracing():
            return complex(self.__float__())

    def hex(self) -> str:
        return realize(self).hex()


_PRECISE_IEEE_FLOAT_SORT = {
    11: z3.Float16(),
    24: z3.Float32(),
    53: z3.Float64(),
}[sys.float_info.mant_dig]


class PreciseIeeeSymbolicFloat(SymbolicFloat):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = float):
        if not isinstance(smtvar, str) and not z3.is_fp(smtvar):
            raise CrossHairInternal(
                f"non-float SMT value ({name_of_type(type(smtvar))}) given to PreciseIeeeSymbolicFloat"
            )
        SymbolicValue.__init__(self, smtvar, typ)

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return _PRECISE_IEEE_FLOAT_SORT

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, float):
            return z3.FPVal(literal, cls._ch_smt_sort())
        return None

    def __eq__(self, other):
        with NoTracing():
            coerced = type(self)._coerce_to_smt_sort(other)
            if coerced is None:
                return False
            return SymbolicBool(fpEQ(self.var, coerced))

    # __hash__ has to be explicitly reassigned because we define __eq__
    __hash__ = SymbolicFloat.__hash__

    def __ne__(self, other):
        with NoTracing():
            coerced = type(self)._coerce_to_smt_sort(other)
            if coerced is None:
                return True
            return SymbolicBool(z3.Not(fpEQ(self.var, coerced)))

    def __int__(self):
        with NoTracing():
            self._check_finite_convert_to("integer")
            return SymbolicInt(
                z3.ToInt(z3.fpToReal(z3.fpRoundToIntegral(z3.RTZ(), self.var)))
            )

    def __round__(self, ndigits=None):
        self_is_finite = isfinite(self)
        if ndigits is None:
            if not self_is_finite:
                # CPython only errors like this when ndigits is None (for ... reasons)
                if isinf(self):
                    raise OverflowError("cannot convert float infinity to integer")
                else:
                    raise ValueError("cannot convert float NaN to integer")
        elif ndigits != 0:
            if not isinstance(ndigits, int):
                raise TypeError(
                    f"'{name_of_type(type(ndigits))}' object cannot be interpreted as an integer"
                )
            factor = 10**ndigits
            return round(self * factor, 0) / factor
        with NoTracing():
            if self_is_finite:
                smt_rounded_real = z3.fpRoundToIntegral(z3.RNE(), self.var)
                if ndigits is None:
                    return SymbolicInt(z3.ToInt(z3.fpToReal(smt_rounded_real)))
                else:
                    return PreciseIeeeSymbolicFloat(smt_rounded_real)
            # Non-finites returns themselves if you supply ndigits:
            return self

    def _check_finite_convert_to(self, target: str) -> None:
        if isfinite(self):
            return
        elif isinf(self):
            raise OverflowError("cannot convert Infinity to " + target)
        else:
            raise ValueError("cannot convert NaN to " + target)

    def __floor__(self):
        with NoTracing():
            self._check_finite_convert_to("integer")
            return PreciseIeeeSymbolicFloat(z3.fpRoundToIntegral(z3.RTN(), self.var))

    def __floordiv__(self, other):
        r = self / other
        with NoTracing():
            return PreciseIeeeSymbolicFloat(z3.fpRoundToIntegral(z3.RTN(), r.var))

    def __rfloordiv__(self, other):
        r = other / self
        with NoTracing():
            return PreciseIeeeSymbolicFloat(z3.fpRoundToIntegral(z3.RTN(), r.var))

    def __ceil__(self):
        with NoTracing():
            self._check_finite_convert_to("integer")
            return PreciseIeeeSymbolicFloat(z3.fpRoundToIntegral(z3.RTP(), self.var))

    def __pow__(self, other, mod=None):
        # TODO: consider losen-ing a little
        return pow(realize(self), realize(other), realize(mod))

    def __trunc__(self):
        with NoTracing():
            self._check_finite_convert_to("integer")
            return PreciseIeeeSymbolicFloat(z3.fpRoundToIntegral(z3.RTZ(), self.var))

    def as_integer_ratio(self) -> Tuple[Integral, Integral]:
        with NoTracing():
            self._check_finite_convert_to("integer ratio")
            return RealBasedSymbolicFloat(z3.fpToReal(self.var)).as_integer_ratio()

    def is_integer(self) -> SymbolicBool:
        return self == self.__int__()
        # with NoTracing():
        #     return SymbolicBool(z3.IsInt(self.var))


_Z3_ONE_HALF = z3.RealVal("1/2")


class RealBasedSymbolicFloat(SymbolicFloat):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type = float):
        assert (
            typ is float
        ), f"RealBasedSymbolicFloat with unexpected python type ({type(typ)})"
        context_statespace().cap_result_at_unknown()
        SymbolicValue.__init__(self, smtvar, typ)

    @classmethod
    def _ch_smt_sort(cls) -> z3.SortRef:
        return z3.RealSort()

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, float) and isfinite(literal):
            return z3.RealVal(literal)
        return None

    def __int__(self):
        with NoTracing():
            var = self.var
            return SymbolicInt(z3.If(var >= 0, z3.ToInt(var), -z3.ToInt(-var)))

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
        with NoTracing():
            return SymbolicInt(z3.ToInt(self.var))

    def __ceil__(self):
        with NoTracing():
            var, floor = self.var, z3.ToInt(self.var)
            return SymbolicInt(z3.If(var == floor, floor, floor + 1))

    def __trunc__(self):
        with NoTracing():
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
        with NoTracing():
            return SymbolicBool(z3.IsInt(self.var))


class SymbolicDictOrSet(SymbolicValue):
    """
    TODO: Ordering is a challenging issue here.
    Modern pythons have in-order iteration for dictionaries but not sets.
    """

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        self.key_pytype = normalize_pytype(type_arg_of(typ, 0))
        space = context_statespace()
        ch_type = space.extra(ModelingDirector).get(self.key_pytype)
        if ch_type:
            self.ch_key_type: Optional[Type[AtomicSymbolicValue]] = ch_type
            self.smt_key_sort = self.ch_key_type._ch_smt_sort()
        else:
            self.ch_key_type = None
            self.smt_key_sort = HeapRef
        SymbolicValue.__init__(self, smtvar, typ)
        space.add(self._len() >= 0)

    def __ch_realize__(self):
        return origin_of(self.python_type)(self)

    def _arr(self):
        return self.var[0]

    def _len(self):
        return self.var[1]

    def __len__(self):
        with NoTracing():
            return SymbolicInt(self._len())

    def __bool__(self):
        with NoTracing():
            return SymbolicBool(self._len() != 0).__bool__()


# TODO: rename to SymbolicImmutableMap (ShellMutableMap is the real symbolic `dict` class)
class SymbolicDict(SymbolicDictOrSet, collections.abc.Mapping):
    """An immutable symbolic dictionary."""

    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        space = context_statespace()
        self.val_pytype = normalize_pytype(type_arg_of(typ, 1))
        val_ch_type = space.extra(ModelingDirector).get(self.val_pytype)
        if val_ch_type:
            self.ch_val_type: Optional[Type[AtomicSymbolicValue]] = val_ch_type
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
        self._iter_cache: List[z3.Const] = []  # TODO: is this used?
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
        space = context_statespace()
        return (
            z3.Const(varname + "_map" + space.uniq(), arr_smt_sort),
            z3.Const(varname + "_len" + space.uniq(), _SMT_INT_SORT),
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
        # TODO: symbolic repr; something like this?:
        # itemiter = self.items()
        # with NoTracing():
        #     return "{" + ", ".join(f"{repr(deep_realize(k))}: {repr(deep_realize(v))}" for k, v in tracing_iter(itemiter)) + "}"

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
                context_statespace(),
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
            space = context_statespace()
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
                    raise CrossHairInternal()
                if idx == len(iter_cache):
                    iter_cache.append(k)
                else:
                    space.add(k == iter_cache[idx])
                idx += 1
                yieldval = smt_to_ch_value(space, self.snapshot, k, self.key_pytype)
                with ResumedTracing():
                    yield yieldval
                arr_var = remaining
            # In this conditional, we reconcile the parallel symbolic variables for
            # length and contents:
            if space.choose_possible(arr_var != self.empty, probability_true=0.0):
                raise IgnoreAttempt("SymbolicDict in inconsistent state")

    def copy(self):
        with NoTracing():
            return SymbolicDict(self.var, self.python_type)


class SymbolicFrozenSet(SymbolicDictOrSet, FrozenSetBase):
    def __init__(self, smtvar: Union[str, z3.ExprRef], typ: Type):
        if origin_of(typ) != frozenset:
            raise CrossHairInternal
        SymbolicDictOrSet.__init__(self, smtvar, typ)
        self._iter_cache: List[z3.Const] = []
        self.empty = z3.K(self._arr().sort().domain(), False)
        space = context_statespace()
        space.add((self._arr() == self.empty) == (self._len() == 0))
        space.defer_assumption("symbolic set is consistent", self._is_consistent)

    @assert_tracing(True)
    def _is_consistent(self) -> SymbolicBool:
        """
        Checks whether the set size is consistent with the SMT array size

        Realizes the size. (but not the values)
        """
        my_len = len(self)
        with NoTracing():
            target_len = 0
            space = context_statespace()
            comparison_smt_array = self.empty
            items = []
            while True:
                with ResumedTracing():
                    if target_len == my_len:
                        break
                item = z3.Const(f"set_{target_len}_{space.uniq()}", self.smt_key_sort)
                items.append(item)
                comparison_smt_array = z3.Store(comparison_smt_array, item, True)
                target_len += 1
            if len(items) >= 2:
                space.add(z3.Distinct(*items))
            return SymbolicBool(comparison_smt_array == self._arr())

    def __ch_is_deeply_immutable__(self) -> bool:
        return True

    def __ch_realize__(self):
        return python_type(self)(map(realize, self))

    def __repr__(self):
        if self:
            return "frozenset({" + ", ".join(map(repr, self)) + "})"
        else:
            return "frozenset()"

    def __hash__(self):
        return deep_realize(self).__hash__()

    def __eq__(self, other):
        (self_arr, self_len) = self.var
        if isinstance(other, SymbolicFrozenSet):
            (other_arr, other_len) = other.var
            if other_arr.sort() == self_arr.sort():
                # TODO: this is wrong for HeapRef sets (which could customize __eq__)
                return SymbolicBool(
                    z3.And(self_len == other_len, self_arr == other_arr)
                )
        if not isinstance(
            other, (set, frozenset, SymbolicFrozenSet, collections.abc.Set)
        ):
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
        space = context_statespace()
        return (
            z3.Const(
                varname + "_map" + space.uniq(),
                z3.ArraySort(self.smt_key_sort, _SMT_BOOL_SORT),
            ),
            z3.Const(varname + "_len" + space.uniq(), _SMT_INT_SORT),
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
            space = context_statespace()
            idx = 0
            arr_sort = self._arr().sort()
            keys_on_heap = is_heapref_sort(arr_sort.domain())
            already_yielded = []
            while True:
                if idx < len(iter_cache):
                    k = iter_cache[idx]
                elif SymbolicBool(idx < len_var).__bool__():
                    if space.choose_possible(
                        arr_var == self.empty, probability_true=0.0
                    ):
                        raise IgnoreAttempt("SymbolicFrozenSet in inconsistent state")
                    k = z3.Const("k" + str(idx) + space.uniq(), arr_sort.domain())
                else:
                    break
                remaining = z3.Const("remaining" + str(idx) + space.uniq(), arr_sort)
                space.add(arr_var == z3.Store(remaining, k, True))
                # TODO: this seems like it won't work the same for heaprefs which can be distinct but equal:
                space.add(z3.Not(z3.Select(remaining, k)))

                if idx > len(iter_cache):
                    raise CrossHairInternal()
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
                            unequal = ch_value != previous_value
                            with NoTracing():
                                if not prefer_true(unequal):
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
            if space.choose_possible(arr_var != self.empty, probability_true=0.0):
                raise IgnoreAttempt("SymbolicFrozenSet in inconsistent state")

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


def flip_slice_vs_symbolic_len(
    space: StateSpace,
    i: Union[int, slice],
    smt_len: z3.ExprRef,
) -> Union[z3.ExprRef, Tuple[z3.ExprRef, z3.ExprRef]]:
    if is_tracing():
        raise CrossHairInternal("index math while tracing")

    def normalize_symbolic_index(idx) -> z3.ExprRef:
        if type(idx) is int:
            return z3IntVal(idx) if idx >= 0 else (smt_len + z3IntVal(idx))
        else:
            smt_idx = SymbolicInt._coerce_to_smt_sort(idx)
            if space.smt_fork(smt_idx >= 0):  # type: ignore
                return smt_idx
            else:
                return smt_len + smt_idx

    if isinstance(i, Integral):
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
        with NoTracing():
            return SymbolicInt(z3.Length(self.var))

    def __bool__(self):
        with NoTracing():
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
        space = context_statespace()
        val_ch_type = space.extra(ModelingDirector).get(self.val_pytype)
        if val_ch_type:
            self.ch_item_type: Optional[Type[AtomicSymbolicValue]] = val_ch_type
            self.item_smt_sort = self.ch_item_type._ch_smt_sort()
        else:
            self.ch_item_type = None
            self.item_smt_sort = HeapRef

        SymbolicValue.__init__(self, smtvar, typ)
        space.add(self._len() >= 0)

    def __init_var__(self, typ, varname):
        assert typ == self.python_type
        arr_smt_type = z3.ArraySort(_SMT_INT_SORT, self.item_smt_sort)
        space = context_statespace()
        return (
            z3.Const(varname + "_map" + space.uniq(), arr_smt_type),
            z3.Const(varname + "_len" + space.uniq(), _SMT_INT_SORT),
        )

    def _arr(self):
        return self.var[0]

    def _len(self):
        return self.var[1]

    def __len__(self):
        with NoTracing():
            return SymbolicInt(self._len())

    def __bool__(self) -> bool:
        with NoTracing():
            return SymbolicBool(self._len() != 0).__bool__()

    def __eq__(self, other):
        with NoTracing():
            if self is other:
                return True
            (self_arr, self_len) = self.var
            if isinstance(other, SymbolicArrayBasedUniformTuple):
                # TODO: Can these be HeapRefs? If so, we're only doing identity checks:
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
        with NoTracing():
            space = context_statespace()
            arr_var, len_var = self.var
            idx = 0
            while space.smt_fork(idx < len_var):
                val = smt_to_ch_value(
                    space, self.snapshot, z3.Select(arr_var, idx), self.val_pytype
                )
                with ResumedTracing():
                    yield val
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
        space = context_statespace()
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
        space = context_statespace()
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
        self,
        value: object,
        start: int = _LIST_INDEX_START_DEFAULT,
        stop: int = _LIST_INDEX_STOP_DEFAULT,
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
        start = 0 if start is _LIST_INDEX_START_DEFAULT else max(start, 0)
        stop = mylen if stop is _LIST_INDEX_STOP_DEFAULT else min(stop, mylen)
        for idx in range(start, stop):
            if self[idx] == value:
                return idx
        raise ValueError


class SymbolicRange:
    start: Union[int, SymbolicInt]
    stop: Union[int, SymbolicInt]
    step: Union[int, SymbolicInt]

    def __init__(
        self,
        a: Union[int, SymbolicInt],
        b: Union[int, SymbolicInt, _Missing] = _MISSING,
        c: Union[int, SymbolicInt, _Missing] = _MISSING,
    ):
        with NoTracing():
            if not isinstance(a, (int, SymbolicInt)):
                raise TypeError
            if b is _MISSING:
                self.start, self.stop, self.step = 0, a, 1
            else:
                if not isinstance(b, (int, SymbolicInt)):
                    raise TypeError
                if c is _MISSING:
                    c = 1
                else:
                    if not isinstance(c, (int, SymbolicInt)):
                        raise TypeError
                    with ResumedTracing():
                        if c == 0:
                            raise ValueError
                self.start, self.stop, self.step = a, b, c

    def __ch_realize__(self):
        start, stop, step = self.start, self.stop, self.step
        return range(realize(start), realize(stop), realize(step))

    def __ch_pytype__(self):
        return range

    def __getitem__(self, idx_or_slice):
        # TODO: compose ranges (Python does this; e.g. `range(10)[:5] == range(5)`)
        return realize(self).__getitem__(idx_or_slice)

    def __iter__(self):
        start, stop, step = self.start, self.stop, self.step
        if step < 0:
            while start > stop:
                yield start
                start += step
        else:
            while start < stop:
                yield start
                start += step

    def __hash__(self):
        with NoTracing():
            start = realize(self.start)
            stop = realize(self.stop)
            step = realize(self.step)
            return hash(range(start, stop, step))

    def __eq__(self, other):
        if not isinstance(other, range):
            return False
        if len(self) != len(other):
            return False
        for (v1, v2) in zip(self, other):
            if v1 != v2:
                return False
        return True

    def __bool__(self):
        return self.__len__() > 0

    def __len__(self):
        start, stop, step = self.start, self.stop, self.step
        if (step > 0) != (stop > start):
            return 0
        width = abs(stop - start)
        if step != 1:
            step = abs(step)
            return (width // step) + min(1, width % step)
        else:
            return width

    def __repr__(self):
        start, stop, step = self.start, self.stop, self.step
        if step == 1:
            return f"range({start}, {stop})"
        else:
            return f"range({start}, {stop}, {step})"

    def __reversed__(self):
        start, stop, step = self.start, self.stop, self.step
        if step < 1:
            width = start - stop
            if width <= 0:
                return iter(())
            revstep = -step
            ll = (width // revstep) + min(1, width % revstep)
            tail_space = width - ((ll - 1) * revstep)
            return iter(SymbolicRange(stop + tail_space, start + tail_space, revstep))
        else:
            width = stop - start
            if width <= 0:
                return iter(())
            ll = (width // step) + min(1, width % step)
            tail_space = width - ((ll - 1) * step)
            return iter(SymbolicRange(stop - tail_space, start - tail_space, -step))


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

    def __eq__(self, other):
        if not isinstance(other, list):
            return False
        return ShellMutableSequence.__eq__(self, other)

    def __lt__(self, other):
        if not isinstance(other, (list, SymbolicList)):
            raise TypeError
        return super().__lt__(other)

    def __mod__(self, *a):
        raise TypeError


_EAGER_OBJECT_SUBTYPES = with_uniform_probabilities([int, str])


class SymbolicType(AtomicSymbolicValue, SymbolicValue, Untracable):
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
            if isinstance(self.pytype_cap, CrossHairValue):
                raise CrossHairInternal(
                    "Cannot create symbolic type capped at a symbolic type"
                )
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

    @assert_tracing(False)
    def _is_superclass_of_(self, other):
        if self is SymbolicType:
            return False
        if type(other) is SymbolicType:
            # Prefer it this way because only _is_subcless_of_ does the type cap lowering.
            return other._is_subclass_of_(self)
        space = context_statespace()
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
        space = context_statespace()
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
            space = context_statespace()
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
                for pytype, probability_true in iter_types(cap, include_abstract=True):
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


@assert_tracing(True)
def symbolic_obj_binop(symbolic_obj: "SymbolicObject", other, op):
    other_type = type(other)
    with NoTracing():
        mytype = symbolic_obj._typ
        if isinstance(mytype, SymbolicType):
            # This just encourages a useful type realization; we discard the result:
            other_smt_type = SymbolicType._coerce_to_smt_sort(other_type)
            if other_smt_type is not None:
                space = context_statespace()
                space.smt_fork(z3Eq(mytype.var, other_smt_type), probability_true=0.9)

            # The following call then lowers the type cap.
            # TODO: This does more work than is really needed. But it might be good for
            # subclass realizations. We want the equality check above mostly because
            # `object`` realizes to int|str and we don't want to spend lots of time
            # considering (usually enum-based) int and str subclasses.
            mytype._is_subclass_of_(other_type)
        obj_with_known_type = symbolic_obj._wrapped()
    return op(obj_with_known_type, other)


class SymbolicObject(ObjectProxy, CrossHairValue, Untracable):
    """
    An object with an unknown type.
    We lazily create a more specific smt-based value in hopes that an
    isinstance() check will be called before something is accessed on us.
    Note that this class is not an SymbolicValue, but its _typ and _inner
    members can be.
    """

    @assert_tracing(False)
    def __init__(self, smtvar: str, typ: Type):
        if not isinstance(typ, type):
            raise CrossHairInternal(f"Creating SymbolicObject with non-type {typ}")
        if isinstance(typ, CrossHairValue):
            raise CrossHairInternal(f"Creating SymbolicObject with symbolic type {typ}")
        object.__setattr__(self, "_typ", SymbolicType(smtvar + "_type", Type[typ]))
        object.__setattr__(self, "_space", context_statespace())
        object.__setattr__(self, "_varname", smtvar)

    @assert_tracing(False)
    def _realize(self):
        object.__getattribute__(self, "_space")
        varname = object.__getattribute__(self, "_varname")

        pytype = realize(object.__getattribute__(self, "_typ"))
        debug(
            "materializing the type of symbolic", varname, "to be", pytype, ch_stack()
        )
        object.__setattr__(self, "_typ", pytype)
        if pytype is object:
            return object()
        return proxy_for_type(pytype, varname, allow_subtypes=False)

    def _wrapped(self):
        with NoTracing():
            inner = _MISSING
            try:
                inner = object.__getattribute__(self, "_inner")
            except AttributeError:
                pass
            if inner is _MISSING:
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
            # TODO: we should do something else here; realizing the type doesn't
            # seem THAT terrible?
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

    def __eq__(self, other):
        if not isinstance(other, tuple):
            return False
        return SymbolicArrayBasedUniformTuple.__eq__(self, other)


class SymbolicBoundedIntTuple(collections.abc.Sequence):
    def __init__(self, ranges: List[Tuple[int, int]], varname: str):
        assert not is_tracing()
        self._ranges = ranges
        space = context_statespace()
        smtlen = z3.Int(varname + "len" + space.uniq())
        space.add(smtlen >= 0)
        self._varname = varname
        self._len = SymbolicInt(smtlen)
        self._created_vars: List[SymbolicInt] = []

    def _create_up_to(self, size: int) -> None:
        space = context_statespace()
        created_vars = self._created_vars
        for idx in range(len(created_vars), size):
            assert idx == len(created_vars)
            smtval = z3.Int(self._varname + "@" + str(idx))
            constraints = [
                z3And(minval <= smtval, smtval <= maxval)
                for minval, maxval in self._ranges
            ]
            space.add(constraints[0] if len(constraints) == 1 else z3Or(*constraints))
            created_vars.append(SymbolicInt(smtval))
            if idx % 1_000 == 999:
                space.check_timeout()

    def __len__(self):
        return self._len

    def __bool__(self) -> bool:
        with NoTracing():
            return SymbolicBool(self._len.var == 0).__bool__()

    def __eq__(self, other):
        if self is other:
            return True
        if not is_iterable(other):
            return False
        otherlen = len(other)
        if len(self) != otherlen:
            return False
        with NoTracing():
            self._create_up_to(realize(otherlen))
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
                if (
                    stop
                    and stop > 0
                    and space.smt_fork(z3Ge(mylen.var, z3IntVal(stop)))
                ):
                    self._create_up_to(stop)
                elif (
                    stop is None
                    and step is None
                    and (
                        start is None
                        or (
                            0 <= start
                            and space.smt_fork(z3Ge(mylen.var, z3IntVal(start)))
                        )
                    )
                ):
                    return SliceView(self, start, mylen)
                else:
                    self._create_up_to(realize(mylen))
                return self._created_vars[start:stop:step]
            else:
                argument = realize(argument)
                if argument >= 0 and space.smt_fork(
                    z3Gt(self._len.var, z3IntVal(argument))
                ):
                    self._create_up_to(realize(argument) + 1)
                else:
                    self._create_up_to(realize(self._len))
                try:
                    return self._created_vars[argument]
                except IndexError:
                    raise IndexError("string index out of range")

    def index(
        self,
        value: object,
        start: int = _LIST_INDEX_START_DEFAULT,
        stop: int = _LIST_INDEX_STOP_DEFAULT,
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
        start = 0 if start is _LIST_INDEX_START_DEFAULT else max(start, 0)
        stop = mylen if stop is _LIST_INDEX_STOP_DEFAULT else min(stop, mylen)
        for idx in range(start, stop):
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
        return repr(self.__str__())  # TODO symbolic repr'ing should be possible

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
        return firstchar + self[1:].lower()

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
            titlefn = space.extra(UnicodeMaskCache).title()
        if self.__len__() == 0:
            return False
        found_one = False
        for char in self:
            codepoint = ord(char)
            with NoTracing():
                smt_codepoint = SymbolicInt._coerce_to_smt_sort(codepoint)
                if space.smt_fork(upperfn(smt_codepoint)):
                    found_one = True
                elif space.smt_fork(
                    z3.Or(lowerfn(smt_codepoint), titlefn(smt_codepoint))
                ):
                    return False
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
        if sys.version_info < (3, 12):
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
                [(0, maxunicode)], smtvar
            )
        elif isinstance(
            smtvar,
            (
                SymbolicBoundedIntTuple,
                SliceView,
                SequenceConcatenation,
                list,  # TODO: are we sharing mutable state here?
            ),
        ):
            self._codepoints = smtvar
        elif isinstance(smtvar, SymbolicList):
            self._codepoints = smtvar.inner  # use the (immutable) contents
        else:
            raise CrossHairInternal(
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
            if not isinstance(i, (Integral, slice)):
                raise TypeError(type(i))
            i = deep_realize(i)
            with ResumedTracing():
                newcontents = self._codepoints[i]
            if not isinstance(i, slice):
                newcontents = [newcontents]
            return LazyIntSymbolicStr(newcontents)

    # # TODO: let's try overriding __iter__ too for performance:
    # def __iter__(self):
    #     for ch in self._codepoints:
    #         with NoTracing():
    #             ret = LazyIntSymbolicStr([ch])
    #         yield ret

    @classmethod
    def _force_into_codepoints(cls, other):
        assert not is_tracing()
        other = typeable_value(other)
        if isinstance(other, LazyIntSymbolicStr):
            return other._codepoints
        elif isinstance(other, str):
            return list(map(ord, other))
        elif isinstance(other, AnySymbolicStr):
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
        for start in range(1 + len(mypoints) - substrlen):
            # We perform the comparison via `all()` because these are usually concrete lists,
            # and any() will defer all the character comparisons into a single SMT query.
            my_candidate = mypoints[start : start + substrlen]
            if not all(a == b for a, b in zip(my_candidate, subpoints)):
                continue
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

    def _find(self, substr, start=None, end=None, from_right=False):
        if not isinstance(substr, str):
            raise TypeError
        mylen = len(self)
        if start is None:
            start = 0
        elif start < 0:
            start += mylen
        if end is None:
            end = mylen
        elif end < 0:
            end += mylen
        matchstr = self[start:end] if start != 0 or end is not mylen else self
        if len(substr) == 0:
            # Add oddity of CPython. We can find the empty string when over-slicing
            # off the left side of the string, but not off the right:
            # ''.find('', 3, 4) == -1
            # ''.find('', -4, -3) == 0
            if matchstr == "" and start > min(mylen, max(end, 0)):
                return -1
            else:
                if from_right:
                    return max(min(end, mylen), 0)
                else:
                    return max(start, 0)
        else:
            if from_right:
                (prefix, match, _) = LazyIntSymbolicStr.rpartition(matchstr, substr)
            else:
                (prefix, match, _) = LazyIntSymbolicStr.partition(matchstr, substr)
            if match == "":
                return -1
            return start + len(prefix)

    def find(self, substr, start=None, end=None):
        return self._find(substr, start, end, from_right=False)

    def rfind(self, substr, start=None, end=None):
        return self._find(substr, start, end, from_right=True)


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
        return z3.SeqSort(z3.IntSort())

    @classmethod
    def _pytype(cls) -> Type:
        return str

    @classmethod
    def _smt_promote_literal(cls, literal) -> Optional[z3.SortRef]:
        if isinstance(literal, str):
            if len(literal) <= 1:
                if len(literal) == 0:
                    return z3.Empty(z3.SeqSort(z3.IntSort()))
                return z3.Unit(z3IntVal(ord(literal)))
            return z3.Concat([z3.Unit(z3IntVal(ord(ch))) for ch in literal])
        return None

    def __ch_realize__(self) -> object:
        codepoints = context_statespace().find_model_value(self.var)
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
        with NoTracing():
            forced = force_to_smt_sort(other, SeqBasedSymbolicStr)
            return SymbolicBool(z3.Contains(self.var, forced))

    def __getitem__(self, i: Union[int, slice]):
        with NoTracing():
            idx_or_pair = process_slice_vs_symbolic_len(
                context_statespace(), i, z3.Length(self.var)
            )
            if isinstance(idx_or_pair, tuple):
                (start, stop) = idx_or_pair
                smt_result = z3.Extract(self.var, start, stop - start)
            else:
                smt_result = z3.Unit(self.var[idx_or_pair])
            return SeqBasedSymbolicStr(smt_result)

    def endswith(self, substr):
        with NoTracing():
            smt_substr = force_to_smt_sort(substr, SeqBasedSymbolicStr)
            return SymbolicBool(z3.SuffixOf(smt_substr, self.var))

    def find(self, substr, start=None, end=None):
        if not isinstance(substr, str):
            raise TypeError
        with NoTracing():
            space = context_statespace()
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
            space = context_statespace()
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
        with NoTracing():
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
_ORD_OF_ZERO = ord("0")
_ORD_OF_ZERO_PLUS_TEN = ord("0") + 10
_ORD_OF_LOWERCASE_A_MINUS_TEN = ord("a") - 10
_ORD_OF_LOWERCASE_A = ord("a")
_ORD_OF_LOWERCASE_F = ord("f")


def make_hex_digit(value: int) -> int:
    num = value % 16
    if num < 10:
        return _ORD_OF_ZERO + num
    else:
        return _ORD_OF_LOWERCASE_A_MINUS_TEN + num


def parse_hex_digit(char_ordinal: int) -> Optional[int]:
    if all([_ORD_OF_ZERO <= char_ordinal, char_ordinal < _ORD_OF_ZERO_PLUS_TEN]):
        return char_ordinal - _ORD_OF_ZERO
    if all([_ORD_OF_LOWERCASE_A <= char_ordinal, char_ordinal <= _ORD_OF_LOWERCASE_F]):
        return char_ordinal - _ORD_OF_LOWERCASE_A_MINUS_TEN
    return None


def is_ascii_space_ord(char_ord: int):
    return any(
        [
            # source: see PY_CTF_SPACE in Python/pyctype.c
            all(
                [
                    char_ord >= 0x09,  # (\t \n \v \f \r)
                    char_ord <= 0x0D,
                ]
            ),
            char_ord == 0x20,  # (space)
        ]
    )


class BytesLike(BufferAbc, AbcString, CrossHairValue):
    def __eq__(self, other) -> bool:
        if not isinstance(other, _ALL_BYTES_TYPES):
            return False
        if len(self) != len(other):
            return False
        return list(self) == list(other)

    if sys.version_info >= (3, 12):

        def __buffer__(self, flags: int):
            with NoTracing():
                return memoryview(realize(self))

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

    def hex(self, sep=_MISSING, bytes_per_sep=1):
        if not isinstance(bytes_per_sep, Integral):
            raise TypeError
        if sep is _MISSING:
            return LazyIntSymbolicStr(
                [
                    make_hex_digit(byt // 16 if ishigh else byt)
                    for byt in self
                    for ishigh in (True, False)
                ]
            )
        if not isinstance(sep, str):
            raise TypeError
        if len(sep) != 1:
            raise ValueError("sep must be length 1.")
        if not sep.isascii():
            raise ValueError("sep must be ASCII")
        if bytes_per_sep >= 0:
            if bytes_per_sep == 0:
                sep = ""
                start_index = 0
                bytes_per_sep = 1
            else:
                start_index = len(self) % bytes_per_sep
        else:
            bytes_per_sep = -bytes_per_sep
            start_index = 0
        chars = []
        for idx, byt in enumerate(self, start=start_index):
            if idx != start_index and idx % bytes_per_sep == 0:
                chars.append(sep)
            low = make_hex_digit(byt)
            high = make_hex_digit(byt // 16)
            chars.append(chr(high))
            chars.append(chr(low))
        # TODO: optimize by creating a LazyIntSymbolicStr directly
        return "".join(chars)


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

    def __hash__(self):
        return deep_realize(self).__hash__()

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
            retseq = self.inner.__getitem__(i)
            with NoTracing():
                return SymbolicBytes(retseq)
        else:
            return self.inner.__getitem__(i)

    def __iter__(self):
        return self.inner.__iter__()

    def __copy__(self):
        with NoTracing():
            return SymbolicBytes(self.inner)

    def __add__(self, other):
        with NoTracing():
            byte_seq = buffer_to_byte_seq(other)
            if byte_seq is other:
                # plain numeric sequences can't be added to byte-like objects
                raise TypeError
            if byte_seq is None:
                return self.__ch_realize__().__add__(realize(other))
        retseq = self.inner + byte_seq
        with NoTracing():
            return SymbolicBytes(self.inner + byte_seq)

    def decode(self, encoding="utf-8", errors="strict"):
        return codecs.decode(self, encoding, errors=errors)

    @classmethod
    def fromhex(cls, hexstr):
        accumulated = []
        high = None
        if not isinstance(hexstr, str):
            raise TypeError
        for idx, ch in enumerate(hexstr):
            if not ch.isascii():
                raise ValueError(
                    f"non-hexadecimal number found in fromhex() arg at position {idx}"
                )
            chnum = ord(ch)
            if high is None and is_ascii_space_ord(chnum):
                continue
            hexdigit = parse_hex_digit(chnum)
            if hexdigit is None:
                raise ValueError(
                    f"non-hexadecimal number found in fromhex() arg at position {idx}"
                )
            if high is None:
                high = hexdigit * 16
            else:
                accumulated.append(high + hexdigit)
                high = None
        if high is not None:
            raise ValueError(
                f"non-hexadecimal number found in fromhex() arg at position {len(hexstr)}"
            )
        return SymbolicBytes(accumulated)


def make_byte_string(creator: SymbolicFactory):
    return SymbolicBytes(SymbolicBoundedIntTuple([(0, 255)], creator.varname))


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

    @classmethod
    def fromhex(cls, hexstr):
        return SymbolicByteArray(bytes.fromhex(hexstr))


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
        with ResumedTracing():
            objlen = len(obj)
            self.readonly = isinstance(obj, bytes)
        self.obj = obj
        self.nbytes = objlen
        self.shape = (objlen,)
        self._sliced = SliceView(obj, 0, objlen)

    def __ch_realize__(self):
        sliced = self._sliced
        obj, start, stop = self.obj, sliced.start, sliced.stop
        self.obj = obj
        return memoryview(realize(obj))[realize(start) : realize(stop)]

    def __ch_deep_realize__(self, memo):
        sliced = self._sliced
        obj, start, stop = self.obj, sliced.start, sliced.stop
        self.obj = obj
        return memoryview(deep_realize(obj, memo))[realize(start) : realize(stop)]

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
        with NoTracing():
            return SymbolicBytes(self._sliced)

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


_PYTYPE_TO_WRAPPER_TYPE = {
    # These are mappings for AtomicSymbolic values - values that we directly represent
    # as single z3 values.
    bool: ((SymbolicBool, 1.0),),
    int: ((SymbolicInt, 1.0),),
    float: ((RealBasedSymbolicFloat, 0.98), (PreciseIeeeSymbolicFloat, 0.02)),
    type: ((SymbolicType, 1.0),),
}


#
# Symbolic-making helpers
#


def make_union_choice(creator: SymbolicFactory, *pytypes):
    for typ, probability_true in with_uniform_probabilities(pytypes)[:-1]:
        if creator.space.smt_fork(
            probability_true=probability_true,
            desc=f"{creator.varname}_is_{smtlib_typename(typ)}",
        ):
            return creator(typ)
    return creator(pytypes[-1])


def make_concrete_or_symbolic(typ: type):
    def make(creator: SymbolicFactory, *type_args):
        nonlocal typ
        space = context_statespace()
        varname, pytype = creator.varname, creator.pytype
        ret = typ(creator.varname, pytype)

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


@assert_tracing(False)
def make_float(varname: str, pytype: Type):
    if os.environ.get("CROSSHAIR_ONLY_FINITE_FLOATS") == "1":
        warnings.warn(
            "Support for CROSSHAIR_ONLY_FINITE_FLOATS will be removed in CrossHair v0.0.75",
            FutureWarning,
        )
        return RealBasedSymbolicFloat(varname, pytype)
    space = context_statespace()
    chosen_typ = space.extra(ModelingDirector).choose(float)
    if chosen_typ is RealBasedSymbolicFloat:
        if space.smt_fork(desc=f"{varname}_isfinite", probability_true=0.8):
            return RealBasedSymbolicFloat(varname, pytype)
        if space.smt_fork(desc=f"{varname}_isnan", probability_true=0.5):
            return nan
        if space.smt_fork(desc=f"{varname}_neginf", probability_true=0.25):
            return -inf
        return inf
    else:
        return chosen_typ(varname, pytype)


def make_tuple(creator: SymbolicFactory, *type_args):
    if not type_args:
        type_args = (object, ...)  # type: ignore
    if len(type_args) == 2 and type_args[1] == ...:
        return SymbolicUniformTuple(creator.varname, creator.pytype)
    elif len(type_args) == 1 and type_args[0] == ():
        # In python, the type for the empty tuple is written like Tuple[()]
        return ()
    else:
        return tuple(
            proxy_for_type(t, creator.varname + "_at_" + str(idx), allow_subtypes=True)
            for (idx, t) in enumerate(type_args)
        )


def make_range(creator: SymbolicFactory) -> SymbolicRange:
    step = SymbolicInt(creator.varname + "_step")
    creator.space.add(step.var != 0)
    return SymbolicRange(creator(int, "_start"), creator(int, "_stop"), step)


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
# Function Patches
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


def _all(items: Iterable) -> Union[SymbolicBool, bool]:
    with NoTracing():
        smt_prefix: List[z3.ExprRef] = []  # a lazy accumulation of symbolic expressions
        space = context_statespace()
        for item in tracing_iter(items):
            if not isinstance(item, (CrossHairValue, bool)):
                # Discharge accumulated SMT clauses, because we don't know whether
                # our evaluation of `item.__bool__()` will raise an exception.
                if smt_prefix:
                    if not space.smt_fork(z3.And(*smt_prefix)):
                        return False
                    smt_prefix.clear()
            if not isinstance(item, (SymbolicBool, bool)):
                with ResumedTracing():
                    item = bool(item)
            if isinstance(item, SymbolicBool):
                smt_prefix.append(item.var)
            elif not item:
                return False
        if not smt_prefix:
            return True
        return SymbolicBool(z3.And(*smt_prefix))


def _any(items: Iterable) -> Union[SymbolicBool, bool]:
    with NoTracing():
        smt_prefix: List[z3.ExprRef] = []  # a lazy accumulation of symbolic expressions
        space = context_statespace()
        for item in tracing_iter(items):
            if not isinstance(item, (CrossHairValue, bool)):
                # Discharge accumulated SMT clauses, because we don't know whether
                # our evaluation of `item.__bool__()` will raise an exception.
                if smt_prefix:
                    if space.smt_fork(z3.Or(*smt_prefix)):
                        return True
                    smt_prefix.clear()
            if not isinstance(item, (SymbolicBool, bool)):
                with ResumedTracing():
                    item = bool(item)
            if isinstance(item, SymbolicBool):
                smt_prefix.append(item.var)
            elif item:
                return True
        if not smt_prefix:
            return False
        return SymbolicBool(z3.Or(*smt_prefix))


_bin = with_realized_args(bin)


def _bytearray(*a):
    if len(a) <= 1:
        with NoTracing():
            if len(a) == 0:
                return SymbolicByteArray([])
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


def _dict(arg=_MISSING, **kwargs) -> Union[dict, ShellMutableMap]:
    if not optional_context_statespace():
        newdict: Union[dict, ShellMutableMap] = dict() if arg is _MISSING else dict(arg)
    if isinstance(arg, Mapping):
        newdict = ShellMutableMap(SimpleDict(list(arg.items())))
    elif arg is _MISSING:
        newdict = ShellMutableMap(SimpleDict([]))
    elif is_iterable(arg):
        keys: List = []
        key_compares: List = []
        all_items: List = []
        for pair in arg:  # NOTE: `arg` can be an iterator; scan only once
            if len(pair) != 2:
                raise ValueError
            (key, val) = pair
            if not is_hashable(key):
                raise ValueError
            all_items.append(pair)
            key_compares.extend(key == k for k in keys)
            keys.append(key)
        if not any(key_compares):
            simpledict = SimpleDict(all_items)
        else:  # we have one or more key conflicts:
            simpledict = SimpleDict([])
            for key, val in reversed(all_items):
                if key not in simpledict:
                    simpledict[key] = val
        newdict = ShellMutableMap(simpledict)
    else:
        raise TypeError
    newdict.update(kwargs)
    return newdict


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


def _int(val: Any = 0, base=_MISSING):
    with NoTracing():
        if isinstance(val, SymbolicInt):
            if base is not _MISSING:
                raise TypeError("int() can't convert non-string with explicit base")
            return val
        if isinstance(val, AnySymbolicStr):
            with ResumedTracing():
                if base is _MISSING:
                    base = 10
                elif not hasattr(base, "__index__"):
                    raise TypeError(
                        f"{name_of_type(type(base))} object cannot be interpreted as an integer"
                    )
                if any([base < 2, base > 10, not val]):
                    # TODO: bses 11-36 are allowed, but require parsing the a-z and A-Z ranges.
                    # TODO: base can be 0, which means to interpret the string as a literal e.g. '0b100'
                    return int(realize(val), base=realize(base))
                ret = 0
                for ch in val:
                    ch_num = ord(ch) - _ORD_OF_ZERO
                    # Use `any()` to collapse symbolc conditions
                    if any((ch_num < 0, ch_num >= base)):
                        # TODO parse other digits with data from unicodedata.decimal()
                        return int(realize(val))
                    else:
                        ret = (ret * base) + ch_num
                return ret
        if base is _MISSING:
            return int(deep_realize(val))
        else:
            return int(deep_realize(val), base=realize(base))


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
    if fn is None:
        filterfn = lambda x: x
    else:
        filterfn = lambda x: fn(x)
    return filter(filterfn, *iters)


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
        return val.__float__()
    return float(realize(val))


def _frozenset(itr=()) -> Union[set, LinearSet]:
    if isinstance(itr, set):
        return LinearSet(itr)
    else:
        return LinearSet.check_unique_and_create(itr)


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
    return _issubclass(type(obj), types)


# CPython's len() forces the return value to be a native integer.
# Avoid that requirement by making it only call __len__().
def _len(ls):
    if hasattr(ls, "__len__"):
        return ls.__len__()
    else:
        raise TypeError(f"object of type '{name_of_type(type(ls))}' has no len()")


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
    # TODO: we should be able to loosen this up a little.
    # TODO: move this into the __pow__ definitions. (different smt vars will have different needs)
    return pow(realize(base), realize(exp), realize(mod))


def _print(*a: object, **kw: Any) -> None:
    print(*deep_realize(a), **deep_realize(kw))


def _range(*a):
    return SymbolicRange(*a)


def _repr(obj: object) -> str:
    """
    post[]: True
    """
    # Skip the built-in repr if possible, because it requires the output
    # to be a native string:
    return invoke_dunder(obj, "__repr__")


def _set(itr=_MISSING) -> Union[set, ShellMutableSet]:
    with NoTracing():
        return ShellMutableSet() if itr is _MISSING else ShellMutableSet(itr)


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
    if not isinstance(b, Sized):
        if is_iterable(b):
            b = list(b)
        else:
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


def _dict_repr(self):  # de-optimize
    if not isinstance(self, dict):
        raise TypeError
    contents = ", ".join([repr(k) + ": " + repr(v) for k, v in self.items()])
    return "{" + contents + "}"


def _list_index(
    self, value, start=_LIST_INDEX_START_DEFAULT, stop=_LIST_INDEX_STOP_DEFAULT
):
    with NoTracing():
        if not isinstance(self, list):
            raise TypeError
        if (
            start is not _LIST_INDEX_START_DEFAULT
            or stop is not _LIST_INDEX_STOP_DEFAULT
        ):
            self = self[start:stop]
        for idx, self_value in enumerate(self):
            with ResumedTracing():
                isequal = value == self_value
            if isequal:
                return idx
        raise ValueError


def _list_repr(self):
    # de-optimize
    if not isinstance(self, list):
        raise TypeError
    contents = ", ".join([repr(x) for x in self])
    return "[" + contents + "]"


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


def _frozenset_repr(self):
    if not isinstance(self, frozenset):
        raise TypeError
    if not self:
        return "frozenset()"
    contents = ", ".join(map(repr, self))
    return "frozenset({" + contents + "})"


def _set_repr(self):
    if not isinstance(self, set):
        raise TypeError
    if not self:
        return "set()"
    contents = ", ".join(map(repr, self))
    return "{" + contents + "}"


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


def _str_percent_format(self, other):
    # Almost nobody uses percent formatting anymore, so it's
    # probably OK to realize here.
    # TODO: However, collections.namedtuple still uses percent formatting to
    # do reprs, so we should consider handling the special case when there
    # are only "%r" substitutions.
    if not isinstance(self, str):
        raise TypeError
    return self.__mod__(deep_realize(other))


def _bytes_join(self, itr) -> str:
    return _join(self, itr, self_type=bytes, item_type=BufferAbc)


def _bytearray_join(self, itr) -> str:
    return _join(self, itr, self_type=bytearray, item_type=BufferAbc)


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


def _tuple_repr(self):
    if not isinstance(self, tuple):
        raise TypeError
    contents = ", ".join(map(repr, self))
    return "(" + contents + ")"


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
    register_type(bool, make_concrete_or_symbolic(SymbolicBool))
    register_type(int, make_concrete_or_symbolic(SymbolicInt))
    register_type(float, make_concrete_or_symbolic(make_float))
    register_type(str, make_concrete_or_symbolic(LazyIntSymbolicStr))
    register_type(list, make_concrete_or_symbolic(SymbolicList))
    register_type(dict, make_dictionary)
    register_type(range, make_range)
    register_type(tuple, make_tuple)
    register_type(set, make_set)
    register_type(frozenset, make_concrete_or_symbolic(SymbolicFrozenSet))
    register_type(type, make_concrete_or_symbolic(SymbolicType))
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

    register_type(re.Pattern, lambda p, t=None: re.compile(realize(p(str))))
    register_type(re.Match, make_raiser(CrosshairUnsupported))

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

    register_patch(all, _all)
    register_patch(any, _any)
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
    register_patch(dict, _dict)
    register_patch(float, _float)
    register_patch(frozenset, _frozenset)
    register_patch(int, _int)
    register_patch(memoryview, _memoryview)
    register_patch(range, _range)
    register_patch(set, _set)

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
    register_patch(str.__repr__, with_symbolic_self(LazyIntSymbolicStr, str.__repr__))
    register_patch(str.__mod__, _str_percent_format)

    # Patches on bytes
    register_patch(bytes.join, _bytes_join)
    register_patch(bytes.fromhex, SymbolicBytes.fromhex)

    # Patches on bytearrays
    register_patch(bytearray.join, _bytearray_join)
    register_patch(bytearray.fromhex, SymbolicByteArray.fromhex)

    # Patches on list
    register_patch(list.__len__, with_checked_self(list, "__len__"))
    register_patch(list.__repr__, _list_repr)
    register_patch(list.copy, with_checked_self(list, "copy"))
    register_patch(list.index, _list_index)
    register_patch(list.pop, with_checked_self(list, "pop"))

    # Patches on dict
    register_patch(dict.__len__, with_checked_self(dict, "__len__"))
    register_patch(dict.__repr__, _dict_repr)
    register_patch(dict.copy, with_checked_self(dict, "copy"))
    register_patch(dict.items, with_checked_self(dict, "items"))
    register_patch(dict.keys, with_checked_self(dict, "keys"))
    # TODO: dict.update (concrete w/ symbolic argument), __getitem__, & more?
    register_patch(dict.get, _dict_get)
    register_patch(dict.values, with_checked_self(dict, "values"))

    # Patches on set/frozenset
    register_patch(set.__repr__, _set_repr)
    register_patch(frozenset.__repr__, _frozenset_repr)
    register_patch(set.copy, with_checked_self(set, "copy"))
    register_patch(frozenset.copy, with_checked_self(frozenset, "copy"))
    register_patch(set.pop, with_checked_self(set, "pop"))

    # Patches on int
    register_patch(int.__repr__, with_checked_self(int, "__repr__"))
    register_patch(int.as_integer_ratio, with_checked_self(int, "as_integer_ratio"))
    if sys.version_info >= (3, 10):
        register_patch(int.bit_count, with_checked_self(int, "bit_count"))
    register_patch(int.bit_length, with_checked_self(int, "bit_length"))
    register_patch(int.conjugate, with_checked_self(int, "conjugate"))
    register_patch(int.from_bytes, _int_from_bytes)
    if sys.version_info >= (3, 12):
        register_patch(int.is_integer, with_checked_self(int, "is_integer"))
    register_patch(int.to_bytes, with_checked_self(int, "to_bytes"))

    # Patches on float
    register_patch(float.__repr__, with_checked_self(float, "__repr__"))
    register_patch(float.fromhex, with_realized_args(float.fromhex))
    register_patch(float.as_integer_ratio, with_checked_self(float, "as_integer_ratio"))
    register_patch(float.conjugate, with_checked_self(float, "conjugate"))
    register_patch(float.hex, with_checked_self(float, "hex"))
    register_patch(float.is_integer, with_checked_self(float, "is_integer"))

    # Patches on tuples
    register_patch(tuple.__repr__, _tuple_repr)

    setup_binops()
