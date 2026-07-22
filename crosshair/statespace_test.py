import time

import pytest
import z3  # type: ignore

from crosshair.core import Patched, proxy_for_type
from crosshair.statespace import (
    HeapRef,
    ModelValueNode,
    RootNode,
    SimpleStateSpace,
    SnapshotRef,
    StateSpace,
    StateSpaceContext,
    _resolve_real_model_value,
    model_value_to_python,
)
from crosshair.tracers import COMPOSITE_TRACER
from crosshair.util import CrossHairInternal, IgnoreAttempt, UnknownSatisfiability

_HEAD_SNAPSHOT = SnapshotRef(-1)


def test_find_key_in_heap():
    space = SimpleStateSpace()
    listref = z3.Const("listref", HeapRef)
    listval1 = space.find_key_in_heap(listref, list, lambda t: [], _HEAD_SNAPSHOT)
    assert isinstance(listval1, list)
    listval2 = space.find_key_in_heap(listref, list, lambda t: [], _HEAD_SNAPSHOT)
    assert listval1 is listval2
    dictref = z3.Const("dictref", HeapRef)
    dictval = space.find_key_in_heap(dictref, dict, lambda t: {}, _HEAD_SNAPSHOT)
    assert dictval is not listval1
    assert isinstance(dictval, dict)


def test_timeout() -> None:
    num_ints = 100
    space = StateSpace(time.process_time() + 60_000, 0.1, RootNode())
    with pytest.raises(UnknownSatisfiability):
        with Patched(), StateSpaceContext(space), COMPOSITE_TRACER:
            ints = [proxy_for_type(int, f"i{i}") for i in range(num_ints)]
            for i in range(num_ints - 2):
                t0 = time.process_time()
                if ints[i] * ints[i + 1] == ints[i + 2]:
                    pass
                ints[i + 1] += ints[i]
    solve_time = time.process_time() - t0
    assert 0.01 < solve_time < 0.5, f"solve_time={solve_time} outside expected range"


def test_infinite_timeout() -> None:
    space = StateSpace(time.monotonic() + 1000, float("+inf"), RootNode())
    assert space.solver.check(True) == z3.sat


def test_checkpoint() -> None:
    space = SimpleStateSpace()
    ref = z3.Const("ref", HeapRef)

    def find_key(snapshot):
        return space.find_key_in_heap(ref, list, lambda t: [], snapshot)

    orig_snapshot = space.current_snapshot()
    listval = find_key(_HEAD_SNAPSHOT)
    space.checkpoint()

    head_listval = find_key(_HEAD_SNAPSHOT)
    head_listval.append(42)
    assert len(head_listval) == 1
    assert listval is not head_listval
    assert len(listval) == 0

    listval_again = find_key(orig_snapshot)
    assert listval_again is listval
    head_listval_again = find_key(_HEAD_SNAPSHOT)
    assert head_listval_again is head_listval


def test_model_value_to_python_AlgebraicNumRef():
    # Tests that z3.AlgebraicNumRef is handled properly.
    # See https://github.com/pschanely/CrossHair/issues/242
    rt2 = z3.simplify(z3.Sqrt(2))
    assert type(rt2) == z3.AlgebraicNumRef
    model_value_to_python(rt2)


def test_model_value_to_python_ArithRef():
    # Tests that a plain z3.ArithRef can be exported as Python
    # See https://github.com/pschanely/CrossHair/issues/381
    rt2 = z3.ToInt(2 ** z3.Int("x"))
    print("type(rt2)", type(rt2))
    assert type(rt2) == z3.ArithRef
    model_value_to_python(rt2)


def test_resolve_real_model_value_rebinds_to_a_representable_float():
    # Regression test for https://github.com/pschanely/CrossHair/issues/491:
    # a RealBasedSymbolicFloat constrained to be `!= 1` could realize to an
    # exact rational like `1 + 2**-1064` that silently rounds to 1.0 as a
    # float, contradicting the very constraint that produced it.
    space = SimpleStateSpace()
    f = z3.Real("f")
    space.add(f != 1)
    huge_denominator = 2**1064
    unrepresentable_model_value = z3.RealVal(
        f"{huge_denominator + 1}/{huge_denominator}"
    )
    bind_value = _resolve_real_model_value(space.solver, f, unrepresentable_model_value)
    assert model_value_to_python(bind_value) != 1.0
    space.add(f == bind_value)
    assert space.solver.check() == z3.sat


def test_model_value_node_gives_up_when_no_float_fits():
    space = SimpleStateSpace()
    f = z3.Real("f")
    # An interval far narrower than one float ULP near 1.0: no representable
    # float exists inside it, so whatever value the solver's model first
    # picks for `f` will need (and fail to find) a nearby fixup.
    lower = z3.RealVal(
        "999999999999999999999999999999999999/1000000000000000000000000000000000000"
    )
    space.add(f > lower)
    space.add(f < 1)
    space.solver.check()
    with pytest.raises(IgnoreAttempt):
        ModelValueNode(space._random, f, space.solver)


def test_smt_fanout(space: SimpleStateSpace):
    option1 = z3.Bool("option1")
    option2 = z3.Bool("option2")
    space.add(z3.Xor(option1, option2))  # Ensure exactly one option can be set
    exprs_and_results = [(option1, "result1"), (option2, "result2")]

    result = space.smt_fanout(exprs_and_results, desc="choose_one")
    assert result in ("result1", "result2")
    if result == "result1":
        assert space.is_possible(option1)
        assert not space.is_possible(option2)
    else:
        assert not space.is_possible(option1)
        assert space.is_possible(option2)


def test_realization_unsat_debug_reports_core(space: SimpleStateSpace):
    x = z3.Int("x")
    space.solver.add(x > 5)
    space.solver.add(x < 3)
    with pytest.raises(CrossHairInternal) as exc_info:
        space._raise_unexpected_realization_unsat(x, None)
    message = str(exc_info.value)
    assert "fresh same-config solver.check(): unsat" in message
    assert "minimal unsat core:" in message
    assert "x > 5" in message
    assert "x < 3" in message
