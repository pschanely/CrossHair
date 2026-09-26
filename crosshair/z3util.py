import z3  # type: ignore
from z3 import (
    Z3_APP_AST,
    Z3_OP_NOT,
    BoolRef,
    BoolSort,
    ExprRef,
    FuncDeclRef,
    IntNumRef,
    IntSort,
    Z3_get_app_arg,
    Z3_get_app_decl,
    Z3_get_ast_kind,
    Z3_get_decl_kind,
    Z3_mk_and,
    Z3_mk_app,
    Z3_mk_eq,
    Z3_mk_ge,
    Z3_mk_gt,
    Z3_mk_le,
    Z3_mk_lt,
    Z3_mk_not,
    Z3_mk_numeral,
    Z3_mk_or,
)
from z3.z3 import _to_ast_array, _to_expr_ref  # type: ignore

ctx = z3.main_ctx()
ctx_ref = ctx.ref()
bool_sort = BoolSort(ctx)
int_sort_ast = IntSort(ctx).ast


def z3Eq(a: ExprRef, b: ExprRef) -> BoolRef:
    # return a == b
    return BoolRef(Z3_mk_eq(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def z3Gt(a: IntNumRef, b: IntNumRef) -> BoolRef:
    # return a > b
    return BoolRef(Z3_mk_gt(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def z3Ge(a: IntNumRef, b: IntNumRef) -> BoolRef:
    # return a >= b
    return BoolRef(Z3_mk_ge(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def z3Le(a: IntNumRef, b: IntNumRef) -> BoolRef:
    # return a <= b
    return BoolRef(Z3_mk_le(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def z3Lt(a: IntNumRef, b: IntNumRef) -> BoolRef:
    # return a < b
    return BoolRef(Z3_mk_lt(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def z3App(fn: FuncDeclRef, *args: ExprRef) -> ExprRef:
    # return fn(*args)
    ast_args, sz = _to_ast_array(args)
    return _to_expr_ref(Z3_mk_app(ctx_ref, fn.ast, sz, ast_args), ctx)


def z3IntVal(x: int) -> z3.IntNumRef:
    # return z3.IntVal(x)
    # Use __index__ to get a regular integer for int subtypes (e.g. enums)
    return IntNumRef(Z3_mk_numeral(ctx_ref, x.__index__().__str__(), int_sort_ast), ctx)


def z3Or(*exprs):
    # return z3.Or(*exprs)
    args, sz = _to_ast_array(exprs)
    return BoolRef(Z3_mk_or(ctx.ref(), sz, args), ctx)


def z3And(*exprs):
    # return z3.And(*exprs)
    args, sz = _to_ast_array(exprs)
    return BoolRef(Z3_mk_and(ctx.ref(), sz, args), ctx)


def z3IsNot(expr: ExprRef) -> bool:
    # return z3.is_not(expr)
    ast = expr.as_ast()
    if Z3_get_ast_kind(ctx_ref, ast) != Z3_APP_AST:
        return False
    return Z3_get_decl_kind(ctx_ref, Z3_get_app_decl(ctx_ref, ast)) == Z3_OP_NOT


def _z3FirstArg(expr: ExprRef) -> ExprRef:
    # return expr.arg(0)
    return _to_expr_ref(Z3_get_app_arg(ctx_ref, expr.as_ast(), 0), ctx)


def z3Not(expr):
    # return z3.Not(expr)
    if z3IsNot(expr):
        return _z3FirstArg(expr)
    else:
        return BoolRef(Z3_mk_not(ctx_ref, expr.as_ast()), ctx)


def z3PopNot(expr):
    if z3IsNot(expr):
        return (False, _z3FirstArg(expr))
    else:
        return (True, expr)
