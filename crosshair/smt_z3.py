"""SMT backend built on z3."""

import ast
from fractions import Fraction
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, cast

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
    Z3_get_numeral_string,
    Z3_mk_add,
    Z3_mk_and,
    Z3_mk_app,
    Z3_mk_array_sort,
    Z3_mk_const,
    Z3_mk_const_array,
    Z3_mk_distinct,
    Z3_mk_div,
    Z3_mk_eq,
    Z3_mk_exists_const,
    Z3_mk_forall_const,
    Z3_mk_fpa_abs,
    Z3_mk_fpa_add,
    Z3_mk_fpa_div,
    Z3_mk_fpa_eq,
    Z3_mk_fpa_geq,
    Z3_mk_fpa_gt,
    Z3_mk_fpa_is_infinite,
    Z3_mk_fpa_is_nan,
    Z3_mk_fpa_is_negative,
    Z3_mk_fpa_is_positive,
    Z3_mk_fpa_is_zero,
    Z3_mk_fpa_leq,
    Z3_mk_fpa_lt,
    Z3_mk_fpa_mul,
    Z3_mk_fpa_nan,
    Z3_mk_fpa_neg,
    Z3_mk_fpa_round_to_integral,
    Z3_mk_fpa_sub,
    Z3_mk_fpa_to_fp_real,
    Z3_mk_fpa_to_real,
    Z3_mk_fpa_zero,
    Z3_mk_func_decl,
    Z3_mk_ge,
    Z3_mk_gt,
    Z3_mk_int2real,
    Z3_mk_is_int,
    Z3_mk_ite,
    Z3_mk_le,
    Z3_mk_lt,
    Z3_mk_mod,
    Z3_mk_mul,
    Z3_mk_not,
    Z3_mk_numeral,
    Z3_mk_or,
    Z3_mk_power,
    Z3_mk_real2int,
    Z3_mk_select,
    Z3_mk_seq_length,
    Z3_mk_store,
    Z3_mk_string_symbol,
    Z3_mk_sub,
    Z3_mk_unary_minus,
    Z3_mk_xor,
)
from z3 import ModelRef as Model
from z3 import Z3Exception as SolverError
from z3.z3 import Sort, _to_ast_array, _to_expr_ref, _to_sort_ref  # type: ignore

from crosshair.smtlib import format_int_numeral, parse_int_numeral, parse_smtlib_literal
from crosshair.util import (
    MAX_REALIZED_INT_BITS,
    CrosshairUnsupported,
    UnknownSatisfiability,
    ch_stack,
    debug,
)

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


def array_sort(domain: SortRef, codomain: SortRef) -> SortRef:
    return _to_sort_ref(Z3_mk_array_sort(ctx_ref, domain.ast, codomain.ast), ctx)


def declare_sort(name: str) -> SortRef:
    return z3.DeclareSort(name, ctx)


class Datatype(NamedTuple):
    sort: SortRef
    constructors: Dict[str, FuncDeclRef]
    recognizers: Dict[str, FuncDeclRef]
    """Maps each constructor name to its recognizer."""
    accessors: Dict[str, FuncDeclRef]
    """Maps each field name to its accessor."""


def declare_datatype(
    name: str, constructors: Dict[str, Sequence[Tuple[str, SortRef]]]
) -> Datatype:
    """
    Declare an algebraic datatype.

    ``constructors`` maps each constructor name to its (field name, sort) pairs.
    """
    datatype = z3.Datatype(name, ctx)
    for constructor_name, fields in constructors.items():
        datatype.declare(constructor_name, *fields)
    sort = datatype.create()
    accessors = {}
    for idx, fields in enumerate(constructors.values()):
        for field_idx, (field_name, _) in enumerate(fields):
            accessors[field_name] = sort.accessor(idx, field_idx)
    return Datatype(
        sort,
        {n: sort.constructor(i) for i, n in enumerate(constructors)},
        {n: sort.recognizer(i) for i, n in enumerate(constructors)},
        accessors,
    )


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
    numeral = format_int_numeral(value)
    return IntNumRef(Z3_mk_numeral(ctx_ref, numeral, _int_sort_ast), ctx)


def numeral_to_int(numeral: IntNumRef) -> int:
    return parse_int_numeral(Z3_get_numeral_string(ctx_ref, numeral.as_ast()))


def real_val(value: object) -> ExprRef:
    """Make a real literal from a Python number or a string like "1/2"."""
    return z3.RealVal(value, ctx)


def fp_val(value: float, sort: SortRef) -> FPRef:
    return z3.FPVal(value, None, sort, ctx)


def fp_nan(sort: SortRef) -> FPRef:
    return _to_expr_ref(Z3_mk_fpa_nan(ctx_ref, sort.ast), ctx)


def fp_plus_zero(sort: SortRef) -> FPRef:
    return FPRef(Z3_mk_fpa_zero(ctx_ref, sort.ast, False), ctx)


def fp_minus_zero(sort: SortRef) -> FPRef:
    return FPRef(Z3_mk_fpa_zero(ctx_ref, sort.ast, True), ctx)


# Constants and functions


def const(name: str, sort: SortRef) -> ExprRef:
    return _to_expr_ref(
        Z3_mk_const(ctx_ref, Z3_mk_string_symbol(ctx_ref, name), sort.ast), ctx
    )


def declare_fun(name: str, *sorts: SortRef) -> FuncDeclRef:
    """Declare an uninterpreted function; the last sort is the range."""
    arity = len(sorts) - 1
    domain = (Sort * arity)()
    for i in range(arity):
        domain[i] = sorts[i].ast
    return FuncDeclRef(
        Z3_mk_func_decl(
            ctx_ref, Z3_mk_string_symbol(ctx_ref, name), arity, domain, sorts[-1].ast
        ),
        ctx,
    )


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


def add(*exprs: ExprRef) -> ExprRef:
    args, sz = _to_ast_array(exprs)
    return _to_expr_ref(Z3_mk_add(ctx_ref, sz, args), ctx)


def sub(a: ExprRef, b: ExprRef) -> ExprRef:
    args, sz = _to_ast_array((a, b))
    return _to_expr_ref(Z3_mk_sub(ctx_ref, sz, args), ctx)


def mul(*exprs: ExprRef) -> ExprRef:
    args, sz = _to_ast_array(exprs)
    return _to_expr_ref(Z3_mk_mul(ctx_ref, sz, args), ctx)


def neg(expr: ExprRef) -> ExprRef:
    return _to_expr_ref(Z3_mk_unary_minus(ctx_ref, expr.as_ast()), ctx)


def abs_(expr: ExprRef) -> ExprRef:
    """Absolute value of an integer or real."""
    zero = real_val(0) if has_real_sort(expr) else int_val(0)
    return ite(lt(expr, zero), neg(expr), expr)


def div(a: ExprRef, b: ExprRef) -> ExprRef:
    """Integer division; the remainder is non-negative."""
    return _to_expr_ref(Z3_mk_div(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def mod(a: ExprRef, b: ExprRef) -> ExprRef:
    """Integer remainder; always non-negative."""
    return _to_expr_ref(Z3_mk_mod(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def real_div(a: ExprRef, b: ExprRef) -> ExprRef:
    return _to_expr_ref(Z3_mk_div(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def power(base: ExprRef, exponent: ExprRef) -> ExprRef:
    """Exponentiation; the result has real sort."""
    return _to_expr_ref(Z3_mk_power(ctx_ref, base.as_ast(), exponent.as_ast()), ctx)


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


def fp_add(rounding_mode: ExprRef, a: ExprRef, b: ExprRef) -> FPRef:
    return FPRef(
        Z3_mk_fpa_add(ctx_ref, rounding_mode.as_ast(), a.as_ast(), b.as_ast()), ctx
    )


def fp_sub(rounding_mode: ExprRef, a: ExprRef, b: ExprRef) -> FPRef:
    return FPRef(
        Z3_mk_fpa_sub(ctx_ref, rounding_mode.as_ast(), a.as_ast(), b.as_ast()), ctx
    )


def fp_mul(rounding_mode: ExprRef, a: ExprRef, b: ExprRef) -> FPRef:
    return FPRef(
        Z3_mk_fpa_mul(ctx_ref, rounding_mode.as_ast(), a.as_ast(), b.as_ast()), ctx
    )


def fp_div(rounding_mode: ExprRef, a: ExprRef, b: ExprRef) -> FPRef:
    return FPRef(
        Z3_mk_fpa_div(ctx_ref, rounding_mode.as_ast(), a.as_ast(), b.as_ast()), ctx
    )


def fp_abs(expr: ExprRef) -> FPRef:
    return FPRef(Z3_mk_fpa_abs(ctx_ref, expr.as_ast()), ctx)


def fp_neg(expr: ExprRef) -> FPRef:
    return FPRef(Z3_mk_fpa_neg(ctx_ref, expr.as_ast()), ctx)


def fp_eq(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_eq(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def fp_neq(a: ExprRef, b: ExprRef) -> BoolRef:
    return not_(fp_eq(a, b))


def fp_lt(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_lt(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def fp_le(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_leq(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def fp_gt(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_gt(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def fp_ge(a: ExprRef, b: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_geq(ctx_ref, a.as_ast(), b.as_ast()), ctx)


def fp_is_nan(expr: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_is_nan(ctx_ref, expr.as_ast()), ctx)


def fp_is_infinite(expr: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_is_infinite(ctx_ref, expr.as_ast()), ctx)


def fp_is_zero(expr: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_is_zero(ctx_ref, expr.as_ast()), ctx)


def fp_is_negative(expr: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_is_negative(ctx_ref, expr.as_ast()), ctx)


def fp_is_positive(expr: ExprRef) -> BoolRef:
    return BoolRef(Z3_mk_fpa_is_positive(ctx_ref, expr.as_ast()), ctx)


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


# Term inspection


def sort_of(expr: ExprRef) -> SortRef:
    return expr.sort()


def has_bool_sort(expr: ExprRef) -> bool:
    return z3.is_bool(expr)


def has_int_sort(expr: ExprRef) -> bool:
    return z3.is_int(expr)


def has_real_sort(expr: ExprRef) -> bool:
    return z3.is_real(expr)


def has_fp_sort(expr: ExprRef) -> bool:
    return z3.is_fp(expr)


def is_same_term(a: ExprRef, b: ExprRef) -> bool:
    return a.eq(b)


def is_uninterpreted_const(expr: ExprRef) -> bool:
    return z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED


def is_true(expr: ExprRef) -> bool:
    return z3.is_true(expr)


def is_false(expr: ExprRef) -> bool:
    return z3.is_false(expr)


def _ground_int_bits(term: ExprRef) -> Optional[int]:
    """
    Bound the bit length of a ground integer term.

    Recognizes numerals, and powers (with natural exponents) combined by +, -, *,
    and to_int. Returns None for any other term.
    """
    if z3.is_int_value(term):
        return numeral_to_int(term).bit_length()
    if z3.is_rational_value(term):
        return max(
            numeral_to_int(term.numerator()).bit_length(),
            numeral_to_int(term.denominator()).bit_length(),
        )
    if z3.is_app(term):
        kind, kids = term.decl().kind(), term.children()
        if (
            kind == z3.Z3_OP_POWER
            and z3.is_int_value(kids[1])
            and kids[1].as_long() >= 0
        ):
            base_bits = _ground_int_bits(kids[0])
            if base_bits is not None:
                return base_bits * max(kids[1].as_long(), 1)
        elif kind in (z3.Z3_OP_MUL, z3.Z3_OP_ADD, z3.Z3_OP_SUB):
            parts = [_ground_int_bits(k) for k in kids]
            if all(p is not None for p in parts):
                bits = cast(List[int], parts)
                return sum(bits) if kind == z3.Z3_OP_MUL else max(bits, default=0) + 1
        elif kind in (z3.Z3_OP_UMINUS, z3.Z3_OP_TO_INT, z3.Z3_OP_TO_REAL) and kids:
            return _ground_int_bits(kids[0])
    return None


def _realize_capped_int(value: ExprRef) -> Optional[int]:
    """
    Evaluate a ground integer term that z3 left unsimplified.

    z3 caps power expansion at max_degree=64; this refolds with the cap lifted.
    Raises CrosshairUnsupported for a result too large to hold in memory, and
    returns None for a term that isn't ground.
    """
    bits = _ground_int_bits(value)
    if bits is None:
        return None
    if bits > MAX_REALIZED_INT_BITS:
        raise CrosshairUnsupported("integer too large to realize", value.sexpr()[:60])
    folded = z3.simplify(value, max_degree=MAX_REALIZED_INT_BITS)
    if not z3.is_int_value(folded):
        raise CrosshairUnsupported("could not realize integer", value.sexpr()[:60])
    return numeral_to_int(folded)


def to_python(value: ExprRef) -> object:
    """Convert a ground term (such as a model value) to a Python value."""
    if z3.is_real(value):
        if isinstance(value, z3.AlgebraicNumRef):
            # Force irrational values to be rational:
            value = value.approx(precision=20)
        return float(
            Fraction(
                numeral_to_int(value.numerator()),
                numeral_to_int(value.denominator()),
            )
        )
    elif z3.is_seq(value):
        ret = []
        while value.num_args() == 2:
            ret.append(to_python(value.arg(0).arg(0)))
            value = value.arg(1)
        if value.num_args() == 1:
            ret.append(to_python(value.arg(0)))
        return ret
    elif z3.is_fp(value):
        return parse_smtlib_literal(value.sexpr())
    elif z3.is_int_value(value):
        return numeral_to_int(value)
    elif hasattr(value, "py_value"):
        if z3.is_int(value):
            realized = _realize_capped_int(value)
            if realized is not None:
                return realized
        return value.py_value()
    elif z3.is_int(value):  # z3 versions without py_value
        return value.as_long()
    else:
        return ast.literal_eval(repr(value))


# Models


def model_eval(model: Model, expr: ExprRef) -> ExprRef:
    """Evaluate ``expr`` in ``model``, choosing values for unconstrained terms."""
    return model.eval(expr, model_completion=True)


def model_satisfies(model: Model, expr: ExprRef) -> bool:
    return z3.is_true(model.eval(expr, model_completion=True))


# Solvers


def _make_z3_solver() -> z3.Solver:
    smt_tactic = z3.Tactic("smt", ctx)
    solver = z3.Solver(z3.Z3_mk_solver_from_tactic(ctx_ref, smt_tactic.tactic), ctx)
    solver.set("mbqi", True)
    # turn off every randomization thing we can think of:
    solver.set("random-seed", 42)
    solver.set("smt.random-seed", 42)
    return solver


class Solver:
    """
    An incremental, deterministic solver.

    ``assertion_version`` increments whenever assertions are added.
    """

    assertion_version: int = 0

    def __init__(self) -> None:
        self._solver = _make_z3_solver()

    def assert_(self, *exprs: ExprRef) -> None:
        self.assertion_version += 1
        solver = self._solver.solver
        for expr in exprs:
            z3.Z3_solver_assert(ctx_ref, solver, expr.as_ast())

    def is_sat(self, *assumptions: ExprRef) -> bool:
        """
        Check whether the assertions and ``assumptions`` are satisfiable.

        Raises UnknownSatisfiability when the solver cannot decide.
        """
        solver = self._solver
        ret = solver.check(*assumptions)
        if ret == z3.unknown:
            reason = solver.reason_unknown()
            debug("Z3 Unknown satisfiability. Reason:", reason)
            debug("Call stack at time of unknown sat:", ch_stack())
            if reason == "interrupted from keyboard":
                raise KeyboardInterrupt
            if assumptions:
                debug("While attempting to assert\n", *(e.sexpr() for e in assumptions))
            debug("Solver state follows:\n", solver.sexpr())
            raise UnknownSatisfiability
        return ret == z3.sat

    def model(self) -> Model:
        """Get a model from the last successful ``is_sat`` check."""
        return self._solver.model()

    def assertions(self) -> List[ExprRef]:
        return list(self._solver.assertions())

    def set_timeout(self, milliseconds: int) -> None:
        """Limit the wall-clock time of each check."""
        self._solver.set(timeout=milliseconds)

    def set_rlimit(self, rlimit: int) -> None:
        """Limit the deterministic (abstract) work of each check."""
        self._solver.set("rlimit", rlimit)

    def __str__(self) -> str:
        return self._solver.sexpr()


def unsat_core(exprs: Sequence[ExprRef]) -> Optional[List[ExprRef]]:
    """
    Find a minimal unsatisfiable subset of ``exprs`` using a fresh solver.

    Returns None when the fresh solver does not find ``exprs`` unsatisfiable.
    """
    solver = _make_z3_solver()
    solver.set(unsat_core=True)
    by_tracker: Dict[str, ExprRef] = {}
    for i, expr in enumerate(exprs):
        name = f"__ch_track_{i}"
        by_tracker[name] = expr
        solver.assert_and_track(expr, const(name, bool_sort))
    if solver.check() != z3.unsat:
        return None
    return [by_tracker[t.decl().name()] for t in solver.unsat_core()]
