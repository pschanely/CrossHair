"""
SMT sorts, terms, and term constructors.

Names follow SMT-LIB. Constructors never convert Python values into terms;
use the ``*_val`` functions to build literals.
"""

from typing import Dict, Sequence, Tuple

import z3  # type: ignore
from z3 import (
    Z3_APP_AST,
    Z3_OP_NOT,
    BoolRef,
    ExprRef,
    FPRef,
    FuncDeclRef,
    IntNumRef,
    SortRef,
    Z3_get_app_arg,
    Z3_get_app_decl,
    Z3_get_ast_kind,
    Z3_get_decl_kind,
    Z3_mk_and,
    Z3_mk_app,
    Z3_mk_const,
    Z3_mk_const_array,
    Z3_mk_distinct,
    Z3_mk_eq,
    Z3_mk_exists_const,
    Z3_mk_forall_const,
    Z3_mk_fpa_eq,
    Z3_mk_fpa_is_infinite,
    Z3_mk_fpa_is_nan,
    Z3_mk_fpa_is_negative,
    Z3_mk_fpa_is_zero,
    Z3_mk_fpa_nan,
    Z3_mk_fpa_round_to_integral,
    Z3_mk_fpa_to_fp_real,
    Z3_mk_fpa_to_real,
    Z3_mk_ge,
    Z3_mk_gt,
    Z3_mk_int2real,
    Z3_mk_is_int,
    Z3_mk_ite,
    Z3_mk_le,
    Z3_mk_lt,
    Z3_mk_not,
    Z3_mk_numeral,
    Z3_mk_or,
    Z3_mk_real2int,
    Z3_mk_select,
    Z3_mk_seq_length,
    Z3_mk_store,
    Z3_mk_string_symbol,
    Z3_mk_xor,
    Z3Exception,
)
from z3.z3 import _to_ast_array, _to_expr_ref  # type: ignore

__all__ = [
    "ExprRef",
    "FPRef",
    "FuncDeclRef",
    "SortRef",
    "Z3Exception",
    "bool_sort",
    "int_sort",
    "real_sort",
    "float16_sort",
    "float32_sort",
    "float64_sort",
    "array_sort",
    "declare_sort",
    "declare_datatype",
    "RNE",
    "RTN",
    "RTP",
    "RTZ",
    "true",
    "false",
    "bool_val",
    "int_val",
    "real_val",
    "fp_val",
    "fp_nan",
    "const",
    "declare_fun",
    "apply",
    "not_",
    "and_",
    "or_",
    "xor",
    "eq",
    "distinct",
    "ite",
    "lt",
    "le",
    "gt",
    "ge",
    "to_int",
    "to_real",
    "is_int",
    "select",
    "store",
    "const_array",
    "seq_len",
    "fp_eq",
    "fp_is_nan",
    "fp_is_infinite",
    "fp_is_zero",
    "fp_is_negative",
    "fp_round_to_integral",
    "fp_to_real",
    "real_to_fp",
    "forall",
    "exists",
    "is_not",
    "pop_not",
    "has_fp_sort",
]

ctx = z3.main_ctx()
ctx_ref = ctx.ref()


# Sorts

bool_sort: SortRef = z3.BoolSort(ctx)
int_sort: SortRef = z3.IntSort(ctx)
real_sort: SortRef = z3.RealSort(ctx)
float16_sort: SortRef = z3.Float16(ctx)
float32_sort: SortRef = z3.Float32(ctx)
float64_sort: SortRef = z3.Float64(ctx)
_int_sort_ast = int_sort.ast


def array_sort(domain: SortRef, range: SortRef) -> SortRef:
    return z3.ArraySort(domain, range)


def declare_sort(name: str) -> SortRef:
    return z3.DeclareSort(name, ctx)


def declare_datatype(
    name: str, constructors: Dict[str, Sequence[Tuple[str, SortRef]]]
) -> SortRef:
    """
    Declare an algebraic datatype.

    ``constructors`` maps each constructor name to its (field name, sort) pairs.
    """
    datatype = z3.Datatype(name, ctx)
    for constructor_name, fields in constructors.items():
        datatype.declare(constructor_name, *fields)
    return datatype.create()


# Rounding modes

RNE: ExprRef = z3.RNE(ctx)
RTN: ExprRef = z3.RTN(ctx)
RTP: ExprRef = z3.RTP(ctx)
RTZ: ExprRef = z3.RTZ(ctx)


# Literals

true: BoolRef = z3.BoolVal(True, ctx)
false: BoolRef = z3.BoolVal(False, ctx)


def bool_val(value: bool) -> BoolRef:
    return true if value else false


def int_val(value: int) -> IntNumRef:
    # __index__ produces a plain int for int subtypes (e.g. enums)
    return IntNumRef(
        Z3_mk_numeral(ctx_ref, value.__index__().__str__(), _int_sort_ast), ctx
    )


def real_val(value: object) -> ExprRef:
    """Make a real literal from a Python number or a string like "1/2"."""
    return z3.RealVal(value, ctx)


def fp_val(value: float, sort: SortRef) -> FPRef:
    return z3.FPVal(value, None, sort, ctx)


def fp_nan(sort: SortRef) -> FPRef:
    return _to_expr_ref(Z3_mk_fpa_nan(ctx_ref, sort.ast), ctx)


# Constants and functions


def const(name: str, sort: SortRef) -> ExprRef:
    return _to_expr_ref(
        Z3_mk_const(ctx_ref, Z3_mk_string_symbol(ctx_ref, name), sort.ast), ctx
    )


def declare_fun(name: str, *sorts: SortRef) -> FuncDeclRef:
    """Declare an uninterpreted function; the last sort is the range."""
    return z3.Function(name, *sorts)


def apply(fn: FuncDeclRef, *args: ExprRef) -> ExprRef:
    ast_args, sz = _to_ast_array(args)
    return _to_expr_ref(Z3_mk_app(ctx_ref, fn.ast, sz, ast_args), ctx)


# Core


def is_not(expr: ExprRef) -> bool:
    ast = expr.as_ast()
    if Z3_get_ast_kind(ctx_ref, ast) != Z3_APP_AST:
        return False
    return Z3_get_decl_kind(ctx_ref, Z3_get_app_decl(ctx_ref, ast)) == Z3_OP_NOT


def _first_arg(expr: ExprRef) -> ExprRef:
    return _to_expr_ref(Z3_get_app_arg(ctx_ref, expr.as_ast(), 0), ctx)


def not_(expr: ExprRef) -> BoolRef:
    """Negate ``expr``, unwrapping it instead if it is already a negation."""
    if is_not(expr):
        return _first_arg(expr)
    else:
        return BoolRef(Z3_mk_not(ctx_ref, expr.as_ast()), ctx)


def pop_not(expr: ExprRef) -> Tuple[bool, ExprRef]:
    """Split ``expr`` into a polarity and a non-negated expression."""
    if is_not(expr):
        return (False, _first_arg(expr))
    else:
        return (True, expr)


def and_(*exprs: ExprRef) -> BoolRef:
    args, sz = _to_ast_array(exprs)
    return BoolRef(Z3_mk_and(ctx_ref, sz, args), ctx)


def or_(*exprs: ExprRef) -> BoolRef:
    args, sz = _to_ast_array(exprs)
    return BoolRef(Z3_mk_or(ctx_ref, sz, args), ctx)


def xor(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_xor(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def eq(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_eq(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def distinct(*exprs: ExprRef) -> BoolRef:
    args, sz = _to_ast_array(exprs)
    return BoolRef(Z3_mk_distinct(ctx_ref, sz, args), ctx)


def ite(condition: ExprRef, then: ExprRef, otherwise: ExprRef) -> ExprRef:
    return _to_expr_ref(
        Z3_mk_ite(ctx_ref, condition.as_ast(), then.as_ast(), otherwise.as_ast()),
        ctx,
    )


# Arithmetic


def lt(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_lt(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def le(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_le(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def gt(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_gt(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def ge(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_ge(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def to_int(expr: ExprRef) -> ExprRef:
    return _to_expr_ref(Z3_mk_real2int(ctx_ref, expr.as_ast()), ctx)


def to_real(expr: ExprRef) -> ExprRef:
    return _to_expr_ref(Z3_mk_int2real(ctx_ref, expr.as_ast()), ctx)


def is_int(expr: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_is_int(ctx_ref, expr.as_ast()), ctx)


# Arrays and sequences


def select(array: ExprRef, index: ExprRef) -> ExprRef:
    return _to_expr_ref(Z3_mk_select(ctx_ref, array.as_ast(), index.as_ast()), ctx)


def store(array: ExprRef, index: ExprRef, value: ExprRef) -> ExprRef:
    return _to_expr_ref(
        Z3_mk_store(ctx_ref, array.as_ast(), index.as_ast(), value.as_ast()), ctx
    )


def const_array(domain: SortRef, value: ExprRef) -> ExprRef:
    return _to_expr_ref(Z3_mk_const_array(ctx_ref, domain.ast, value.as_ast()), ctx)


def seq_len(seq: ExprRef) -> ExprRef:
    return _to_expr_ref(Z3_mk_seq_length(ctx_ref, seq.as_ast()), ctx)


# Floating point


def fp_eq(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_eq(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def fp_is_nan(expr: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_is_nan(ctx_ref, expr.as_ast()), ctx)


def fp_is_infinite(expr: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_is_infinite(ctx_ref, expr.as_ast()), ctx)


def fp_is_zero(expr: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_is_zero(ctx_ref, expr.as_ast()), ctx)


def fp_is_negative(expr: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_is_negative(ctx_ref, expr.as_ast()), ctx)


def fp_round_to_integral(rounding_mode: ExprRef, expr: ExprRef) -> FPRef:
    return FPRef(
        Z3_mk_fpa_round_to_integral(ctx_ref, rounding_mode.as_ast(), expr.as_ast()),
        ctx,
    )


def fp_to_real(expr: ExprRef) -> ExprRef:
    return _to_expr_ref(Z3_mk_fpa_to_real(ctx_ref, expr.as_ast()), ctx)


def real_to_fp(rounding_mode: ExprRef, expr: ExprRef, sort: SortRef) -> FPRef:
    return FPRef(
        Z3_mk_fpa_to_fp_real(ctx_ref, rounding_mode.as_ast(), expr.as_ast(), sort.ast),
        ctx,
    )


def has_fp_sort(expr: ExprRef) -> bool:
    return z3.is_fp(expr)


# Quantifiers


def forall(variables: Sequence[ExprRef], body: ExprRef) -> BoolRef:
    bound, num_bound = _to_ast_array(variables)
    return _to_expr_ref(
        Z3_mk_forall_const(ctx_ref, 1, num_bound, bound, 0, None, body.as_ast()), ctx
    )


def exists(variables: Sequence[ExprRef], body: ExprRef) -> BoolRef:
    bound, num_bound = _to_ast_array(variables)
    return _to_expr_ref(
        Z3_mk_exists_const(ctx_ref, 1, num_bound, bound, 0, None, body.as_ast()), ctx
    )
