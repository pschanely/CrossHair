"""SMT backend built on cvc5."""

import struct
from fractions import Fraction
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import cvc5  # type: ignore
from cvc5 import Kind, RoundingMode

from crosshair.smtlib import format_int_numeral, parse_int_numeral
from crosshair.util import CrosshairUnsupported, UnknownSatisfiability, ch_stack, debug

if TYPE_CHECKING:
    ExprRef = Any
    SortRef = Any
else:
    ExprRef = cvc5.Term
    SortRef = cvc5.Sort
BoolRef = ExprRef
FPRef = ExprRef
IntNumRef = ExprRef
SolverError = RuntimeError

tm = cvc5.TermManager()


class FuncDeclRef(NamedTuple):
    application_kind: Kind
    term: ExprRef


# Sorts

bool_sort: SortRef = tm.getBooleanSort()
int_sort: SortRef = tm.getIntegerSort()
real_sort: SortRef = tm.getRealSort()
float16_sort: SortRef = tm.mkFloatingPointSort(5, 11)
float32_sort: SortRef = tm.mkFloatingPointSort(8, 24)
float64_sort: SortRef = tm.mkFloatingPointSort(11, 53)

_PACK_FORMATS = {16: "e", 32: "f", 64: "d"}


def array_sort(domain: SortRef, codomain: SortRef) -> SortRef:
    return tm.mkArraySort(domain, codomain)


def declare_sort(name: str) -> SortRef:
    return tm.mkUninterpretedSort(name)


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
    decl = tm.mkDatatypeDecl(name)
    for constructor_name, fields in constructors.items():
        constructor_decl = tm.mkDatatypeConstructorDecl(constructor_name)
        for field_name, field_sort in fields:
            constructor_decl.addSelector(field_name, field_sort)
        decl.addConstructor(constructor_decl)
    sort = tm.mkDatatypeSort(decl)
    datatype = sort.getDatatype()
    ctors, testers, accessors = {}, {}, {}
    for constructor_name, fields in constructors.items():
        ctor = datatype[constructor_name]
        ctors[constructor_name] = FuncDeclRef(Kind.APPLY_CONSTRUCTOR, ctor.getTerm())
        testers[constructor_name] = FuncDeclRef(Kind.APPLY_TESTER, ctor.getTesterTerm())
        for field_name, _ in fields:
            accessors[field_name] = FuncDeclRef(
                Kind.APPLY_SELECTOR, ctor[field_name].getTerm()
            )
    return Datatype(sort, ctors, testers, accessors)


# Rounding modes

RNE: ExprRef = tm.mkRoundingMode(RoundingMode.ROUND_NEAREST_TIES_TO_EVEN)
RTN: ExprRef = tm.mkRoundingMode(RoundingMode.ROUND_TOWARD_NEGATIVE)
RTP: ExprRef = tm.mkRoundingMode(RoundingMode.ROUND_TOWARD_POSITIVE)
RTZ: ExprRef = tm.mkRoundingMode(RoundingMode.ROUND_TOWARD_ZERO)


# Literals

true: BoolRef = tm.mkTrue()
false: BoolRef = tm.mkFalse()


def bool_val(value: bool) -> BoolRef:
    return true if value else false


def int_val(value: int) -> IntNumRef:
    return tm.mkInteger(format_int_numeral(value))


def numeral_to_int(numeral: IntNumRef) -> int:
    return parse_int_numeral(str(numeral).replace("(- ", "-").rstrip(")"))


def real_val(value: object) -> ExprRef:
    """Make a real literal from a Python number or a string like "1/2"."""
    if isinstance(value, float):
        value = Fraction(value.__repr__())
    return tm.mkReal(str(value))


def _fp_dimensions(sort: SortRef) -> Tuple[int, int]:
    return (
        sort.getFloatingPointExponentSize(),
        sort.getFloatingPointSignificandSize(),
    )


def fp_val(value: float, sort: SortRef) -> FPRef:
    exponent, significand = _fp_dimensions(sort)
    width = exponent + significand
    fmt = _PACK_FORMATS[width]
    bits = int.from_bytes(struct.pack(">" + fmt, value), "big")
    return tm.mkFloatingPoint(exponent, significand, tm.mkBitVector(width, bits))


def fp_nan(sort: SortRef) -> FPRef:
    return tm.mkFloatingPointNaN(*_fp_dimensions(sort))


def fp_plus_zero(sort: SortRef) -> FPRef:
    return tm.mkFloatingPointPosZero(*_fp_dimensions(sort))


def fp_minus_zero(sort: SortRef) -> FPRef:
    return tm.mkFloatingPointNegZero(*_fp_dimensions(sort))


# Constants and functions

_CONSTS: Dict[Tuple[str, SortRef], ExprRef] = {}


def const(name: str, sort: SortRef) -> ExprRef:
    key = (name, sort)
    term = _CONSTS.get(key)
    if term is None:
        term = tm.mkConst(sort, name)
        _CONSTS[key] = term
    return term


def declare_fun(name: str, *sorts: SortRef) -> FuncDeclRef:
    """Declare an uninterpreted function; the last sort is the range."""
    *domain, codomain = sorts
    return FuncDeclRef(Kind.APPLY_UF, const(name, tm.mkFunctionSort(domain, codomain)))


def apply(fn: FuncDeclRef, *args: ExprRef) -> ExprRef:
    return tm.mkTerm(fn.application_kind, fn.term, *args)


def _mk(kind: Kind, *args: ExprRef) -> ExprRef:
    return tm.mkTerm(kind, *args)


def _mk_nary(kind: Kind, args: Sequence[ExprRef], empty: ExprRef) -> ExprRef:
    if not args:
        return empty
    if len(args) == 1:
        return args[0]
    return tm.mkTerm(kind, *args)


# Core


def is_not(expr: ExprRef) -> bool:
    return expr.getKind() == Kind.NOT


def not_(expr: ExprRef) -> BoolRef:
    """Negate ``expr``, unwrapping it instead if it is already a negation."""
    if is_not(expr):
        return expr[0]
    return _mk(Kind.NOT, expr)


def pop_not(expr: ExprRef) -> Tuple[bool, ExprRef]:
    """Split ``expr`` into a polarity and a non-negated expression."""
    if is_not(expr):
        return (False, expr[0])
    return (True, expr)


def and_(*exprs: ExprRef) -> BoolRef:
    return _mk_nary(Kind.AND, exprs, true)


def or_(*exprs: ExprRef) -> BoolRef:
    return _mk_nary(Kind.OR, exprs, false)


def xor(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.XOR, a, b)


def eq(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.EQUAL, a, b)


def distinct(*exprs: ExprRef) -> BoolRef:
    if len(exprs) < 2:
        return true
    return _mk(Kind.DISTINCT, *exprs)


def ite(condition: ExprRef, then: ExprRef, otherwise: ExprRef) -> ExprRef:
    return _mk(Kind.ITE, condition, then, otherwise)


# Arithmetic


def add(*exprs: ExprRef) -> ExprRef:
    return _mk_nary(Kind.ADD, exprs, tm.mkInteger(0))


def sub(a: ExprRef, b: ExprRef) -> ExprRef:
    return _mk(Kind.SUB, a, b)


def mul(*exprs: ExprRef) -> ExprRef:
    return _mk_nary(Kind.MULT, exprs, tm.mkInteger(1))


def neg(expr: ExprRef) -> ExprRef:
    return _mk(Kind.NEG, expr)


def abs_(expr: ExprRef) -> ExprRef:
    """Absolute value of an integer or real."""
    return _mk(Kind.ABS, expr)


def div(a: ExprRef, b: ExprRef) -> ExprRef:
    """Integer division; the remainder is non-negative."""
    return _mk(Kind.INTS_DIVISION, a, b)


def mod(a: ExprRef, b: ExprRef) -> ExprRef:
    """Integer remainder; always non-negative."""
    return _mk(Kind.INTS_MODULUS, a, b)


def real_div(a: ExprRef, b: ExprRef) -> ExprRef:
    return _mk(Kind.DIVISION, a, b)


def _as_real(expr: ExprRef) -> ExprRef:
    return to_real(expr) if expr.getSort().isInteger() else expr


_MAX_EXPANDED_EXPONENT = 64


def power(base: ExprRef, exponent: ExprRef) -> ExprRef:
    """Exponentiation; the result has real sort."""
    base = _as_real(base)
    if exponent.isIntegerValue():
        count = numeral_to_int(_as_int_value(exponent))
        if 0 <= count <= _MAX_EXPANDED_EXPONENT:
            return _mk_nary(Kind.MULT, [base] * count, tm.mkReal(1))
    return _mk(Kind.POW, base, _as_real(exponent))


def _as_int_value(value: ExprRef) -> ExprRef:
    if value.getSort().isInteger():
        return value
    return tm.mkInteger(str(value.getRealValue().numerator))


def lt(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.LT, a, b)


def le(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.LEQ, a, b)


def gt(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.GT, a, b)


def ge(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.GEQ, a, b)


def to_int(expr: ExprRef) -> ExprRef:
    return _mk(Kind.TO_INTEGER, expr)


def to_real(expr: ExprRef) -> ExprRef:
    return _mk(Kind.TO_REAL, expr)


def is_int(expr: ExprRef) -> BoolRef:
    return _mk(Kind.IS_INTEGER, expr)


# Arrays and sequences


def select(array: ExprRef, index: ExprRef) -> ExprRef:
    return _mk(Kind.SELECT, array, index)


def store(array: ExprRef, index: ExprRef, value: ExprRef) -> ExprRef:
    return _mk(Kind.STORE, array, index, value)


def const_array(domain: SortRef, value: ExprRef) -> ExprRef:
    return tm.mkConstArray(tm.mkArraySort(domain, value.getSort()), value)


def seq_len(seq: ExprRef) -> ExprRef:
    return _mk(Kind.SEQ_LENGTH, seq)


# Floating point


def fp_add(rounding_mode: ExprRef, a: ExprRef, b: ExprRef) -> FPRef:
    return _mk(Kind.FLOATINGPOINT_ADD, rounding_mode, a, b)


def fp_sub(rounding_mode: ExprRef, a: ExprRef, b: ExprRef) -> FPRef:
    return _mk(Kind.FLOATINGPOINT_SUB, rounding_mode, a, b)


def fp_mul(rounding_mode: ExprRef, a: ExprRef, b: ExprRef) -> FPRef:
    return _mk(Kind.FLOATINGPOINT_MULT, rounding_mode, a, b)


def fp_div(rounding_mode: ExprRef, a: ExprRef, b: ExprRef) -> FPRef:
    return _mk(Kind.FLOATINGPOINT_DIV, rounding_mode, a, b)


def fp_abs(expr: ExprRef) -> FPRef:
    return _mk(Kind.FLOATINGPOINT_ABS, expr)


def fp_neg(expr: ExprRef) -> FPRef:
    return _mk(Kind.FLOATINGPOINT_NEG, expr)


def fp_eq(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.FLOATINGPOINT_EQ, a, b)


def fp_neq(a: ExprRef, b: ExprRef) -> BoolRef:
    return not_(fp_eq(a, b))


def fp_lt(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.FLOATINGPOINT_LT, a, b)


def fp_le(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.FLOATINGPOINT_LEQ, a, b)


def fp_gt(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.FLOATINGPOINT_GT, a, b)


def fp_ge(a: ExprRef, b: ExprRef) -> BoolRef:
    return _mk(Kind.FLOATINGPOINT_GEQ, a, b)


def fp_is_nan(expr: ExprRef) -> BoolRef:
    return _mk(Kind.FLOATINGPOINT_IS_NAN, expr)


def fp_is_infinite(expr: ExprRef) -> BoolRef:
    return _mk(Kind.FLOATINGPOINT_IS_INF, expr)


def fp_is_zero(expr: ExprRef) -> BoolRef:
    return _mk(Kind.FLOATINGPOINT_IS_ZERO, expr)


def fp_is_negative(expr: ExprRef) -> BoolRef:
    return _mk(Kind.FLOATINGPOINT_IS_NEG, expr)


def fp_is_positive(expr: ExprRef) -> BoolRef:
    return _mk(Kind.FLOATINGPOINT_IS_POS, expr)


def fp_round_to_integral(rounding_mode: ExprRef, expr: ExprRef) -> FPRef:
    return _mk(Kind.FLOATINGPOINT_RTI, rounding_mode, expr)


def fp_to_real(expr: ExprRef) -> ExprRef:
    return _mk(Kind.FLOATINGPOINT_TO_REAL, expr)


def real_to_fp(rounding_mode: ExprRef, expr: ExprRef, sort: SortRef) -> FPRef:
    op = tm.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_REAL, *_fp_dimensions(sort))
    return tm.mkTerm(op, rounding_mode, expr)


# Quantifiers


def _quantify(kind: Kind, variables: Sequence[ExprRef], body: ExprRef) -> BoolRef:
    bound = [tm.mkVar(v.getSort(), str(v)) for v in variables]
    return tm.mkTerm(
        kind,
        tm.mkTerm(Kind.VARIABLE_LIST, *bound),
        body.substitute(list(variables), bound),
    )


def forall(variables: Sequence[ExprRef], body: ExprRef) -> BoolRef:
    return _quantify(Kind.FORALL, variables, body)


def exists(variables: Sequence[ExprRef], body: ExprRef) -> BoolRef:
    return _quantify(Kind.EXISTS, variables, body)


# Term inspection


def sort_of(expr: ExprRef) -> SortRef:
    return expr.getSort()


def has_bool_sort(expr: ExprRef) -> bool:
    return expr.getSort().isBoolean()


def has_int_sort(expr: ExprRef) -> bool:
    return expr.getSort().isInteger()


def has_real_sort(expr: ExprRef) -> bool:
    return expr.getSort().isReal()


def has_fp_sort(expr: ExprRef) -> bool:
    return expr.getSort().isFloatingPoint()


def is_same_term(a: ExprRef, b: ExprRef) -> bool:
    return a == b


def is_uninterpreted_const(expr: ExprRef) -> bool:
    return expr.getKind() == Kind.CONSTANT


def is_true(expr: ExprRef) -> bool:
    return expr.isBooleanValue() and expr.getBooleanValue()


def is_false(expr: ExprRef) -> bool:
    return expr.isBooleanValue() and not expr.getBooleanValue()


def to_python(value: ExprRef) -> object:
    """Convert a ground term (such as a model value) to a Python value."""
    if value.isBooleanValue():
        return value.getBooleanValue()
    if value.getSort().isReal() and value.isRealValue():
        return float(value.getRealValue())
    if value.isRealAlgebraicNumber():
        return float(_approximate_algebraic(value))
    if value.isIntegerValue():
        return numeral_to_int(value)
    if value.isFloatingPointValue():
        exponent, significand, bits = value.getFloatingPointValue()
        width = exponent + significand
        raw = int(bits.getBitVectorValue(2), 2).to_bytes(width // 8, "big")
        return struct.unpack(">" + _PACK_FORMATS[width], raw)[0]
    if value.getSort().isInteger():
        raise CrosshairUnsupported("could not realize integer", str(value)[:60])
    raise CrosshairUnsupported("cannot convert SMT value", str(value)[:60])


_ALGEBRAIC_BISECTIONS = 64


def _approximate_algebraic(value: ExprRef) -> Fraction:
    """Bisect the isolating interval of a real algebraic number."""
    variable = tm.mkVar(real_sort, "v")
    polynomial = value.getRealAlgebraicNumberDefiningPolynomial(variable)
    low = value.getRealAlgebraicNumberLowerBound().getRealValue()
    high = value.getRealAlgebraicNumberUpperBound().getRealValue()

    def sign_at(point: Fraction) -> int:
        at_point = polynomial.substitute(variable, tm.mkReal(str(point)))
        return _evaluator.simplify(at_point).getRealOrIntegerValueSign()

    low_sign = sign_at(low)
    for _ in range(_ALGEBRAIC_BISECTIONS):
        mid = (low + high) / 2
        mid_sign = sign_at(mid)
        if mid_sign == 0:
            return mid
        if mid_sign == low_sign:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _free_consts(terms: Iterable[ExprRef], seen: Set[ExprRef]) -> List[ExprRef]:
    """Collect constants in ``terms`` not already in ``seen`` (which is updated)."""
    found = []
    stack = list(terms)
    while stack:
        term = stack.pop()
        if term in seen:
            continue
        seen.add(term)
        if term.getKind() == Kind.CONSTANT:
            found.append(term)
        stack.extend(term)
    return found


# Models


def _new_solver(**options: str) -> cvc5.Solver:
    solver = cvc5.Solver(tm)
    solver.setOption("arrays-exp", "true")
    solver.setOption("incremental", "true")
    for name, value in options.items():
        solver.setOption(name.replace("_", "-"), value)
    return solver


_evaluator = _new_solver(produce_models="true")


class Model:
    """Values for constants, as found by one satisfiable check."""

    def __init__(self, consts: List[ExprRef], values: List[ExprRef]) -> None:
        self._consts = consts
        self._values = values
        self._value_of: Dict[ExprRef, ExprRef] = dict(zip(consts, values))

    def _complete(self, expr: ExprRef) -> None:
        free = _free_consts([expr], set())
        missing = [c for c in free if c not in self._value_of]
        if not missing:
            return
        _evaluator.checkSat()
        values = _evaluator.getValue(missing)
        self._consts.extend(missing)
        self._values.extend(values)
        self._value_of.update(zip(missing, values))

    def eval(self, expr: ExprRef) -> ExprRef:
        self._complete(expr)
        value = self._value_of.get(expr)
        if value is not None:
            return value
        ground = expr.substitute(self._consts, self._values)
        simplified = _evaluator.simplify(ground)
        if _is_value(simplified):
            return simplified
        _evaluator.checkSat()
        return _evaluator.getValue(ground)


def _is_value(term: ExprRef) -> bool:
    return (
        term.isBooleanValue()
        or term.isIntegerValue()
        or term.isRealValue()
        or term.isFloatingPointValue()
        or term.isUninterpretedSortValue()
        or term.getKind() == Kind.APPLY_CONSTRUCTOR
    )


def model_eval(model: Model, expr: ExprRef) -> ExprRef:
    """Evaluate ``expr`` in ``model``, choosing values for unconstrained terms."""
    return model.eval(expr)


def model_satisfies(model: Model, expr: ExprRef) -> bool:
    return is_true(model.eval(expr))


# Solvers


class Solver:
    """
    An incremental, deterministic solver.

    ``assertion_version`` increments whenever assertions are added.
    """

    assertion_version: int = 0

    def __init__(self) -> None:
        self._solver = _new_solver(produce_models="true", seed="42")
        self._seen: Set[ExprRef] = set()
        self._consts: List[ExprRef] = []

    def _track(self, exprs: Iterable[ExprRef]) -> None:
        self._consts.extend(_free_consts(exprs, self._seen))

    def assert_(self, *exprs: ExprRef) -> None:
        self.assertion_version += 1
        self._track(exprs)
        for expr in exprs:
            self._solver.assertFormula(expr)

    def is_sat(self, *assumptions: ExprRef) -> bool:
        """
        Check whether the assertions and ``assumptions`` are satisfiable.

        Raises UnknownSatisfiability when the solver cannot decide.
        """
        self._track(assumptions)
        if assumptions:
            result = self._solver.checkSatAssuming(*assumptions)
        else:
            result = self._solver.checkSat()
        if result.isUnknown():
            debug(
                "cvc5 Unknown satisfiability. Reason:", result.getUnknownExplanation()
            )
            debug("Call stack at time of unknown sat:", ch_stack())
            raise UnknownSatisfiability
        return result.isSat()

    def model(self) -> Model:
        """Get a model from the last successful ``is_sat`` check."""
        consts = list(self._consts)
        values = self._solver.getValue(consts) if consts else []
        return Model(consts, list(values))

    def assertions(self) -> List[ExprRef]:
        return list(self._solver.getAssertions())

    def set_timeout(self, milliseconds: int) -> None:
        """Limit the wall-clock time of each check."""
        self._solver.setOption("tlimit-per", str(milliseconds))

    def set_rlimit(self, rlimit: int) -> None:
        """Ignored: cvc5 resource units are not comparable to z3's."""

    def __str__(self) -> str:
        return "\n".join(str(a) for a in self._solver.getAssertions())


def unsat_core(exprs: Sequence[ExprRef]) -> Optional[List[ExprRef]]:
    """
    Find a minimal unsatisfiable subset of ``exprs`` using a fresh solver.

    Returns None when the fresh solver does not find ``exprs`` unsatisfiable.
    """
    solver = _new_solver(produce_unsat_cores="true", minimal_unsat_cores="true")
    for expr in exprs:
        solver.assertFormula(expr)
    if not solver.checkSat().isUnsat():
        return None
    return list(solver.getUnsatCore())
