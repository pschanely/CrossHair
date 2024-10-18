import z3  # type: ignore
from z3 import (
    BoolRef,
    BoolSort,
    ExprRef,
    IntNumRef,
    IntSort,
    Z3_mk_and,
    Z3_mk_eq,
    Z3_mk_ge,
    Z3_mk_gt,
    Z3_mk_not,
    Z3_mk_numeral,
    Z3_mk_or,
    Z3_solver_assert,
)
from z3.z3 import _to_ast_array  # type: ignore

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


def z3IntVal(x: int) -> z3.IntNumRef:
    # return z3.IntVal(x)
    # Use __index__ to get a regular integer for int subtypes (e.g. enums)
    return IntNumRef(Z3_mk_numeral(ctx_ref, x.__index__().__str__(), int_sort_ast), ctx)


def z3Or(*exprs):
    # return z3.Or(*exprs)
    (args, sz) = _to_ast_array(exprs)
    return BoolRef(Z3_mk_or(ctx.ref(), sz, args), ctx)


def z3And(*exprs):
    # return z3.And(*exprs)
    (args, sz) = _to_ast_array(exprs)
    return BoolRef(Z3_mk_and(ctx.ref(), sz, args), ctx)


def z3Aassert(solver, expr):
    # return solver.add(expr)
    assert isinstance(expr, z3.ExprRef)
    Z3_solver_assert(ctx_ref, solver.solver, expr.as_ast())


def z3Not(expr):
    # return z3.Not(expr)
    if z3.is_not(expr):
        return expr.arg(0)
    else:
        return BoolRef(Z3_mk_not(ctx_ref, expr.as_ast()), ctx)


def z3PopNot(expr):
    if z3.is_not(expr):
        return (False, expr.arg(0))
    else:
        return (True, expr)
