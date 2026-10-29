import time

import pytest
import z3  # type: ignore

from crosshair import smt
from crosshair.core import Patched, proxy_for_type
from crosshair.statespace import (
    HeapRef,
    RootNode,
    SimpleStateSpace,
    SnapshotRef,
    StateSpace,
    StateSpaceContext,
    model_value_to_python,
)
from crosshair.tracers import COMPOSITE_TRACER, NoTracing
from crosshair.util import (
    CrossHairInternal,
    CrosshairUnsupported,
    UnknownSatisfiability,
)

_HEAD_SNAPSHOT = SnapshotRef(-1)


def test_find_key_in_heap():
    space = SimpleStateSpace()
    listref = smt.Const("listref", HeapRef)
    listval1 = space.find_key_in_heap(listref, list, lambda t: [], _HEAD_SNAPSHOT)
    assert isinstance(listval1, list)
    listval2 = space.find_key_in_heap(listref, list, lambda t: [], _HEAD_SNAPSHOT)
    assert listval1 is listval2
    dictref = smt.Const("dictref", HeapRef)
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
    ref = smt.Const("ref", HeapRef)

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
    # Tests that smt.AlgebraicNumRef is handled properly.
    # See https://github.com/pschanely/CrossHair/issues/242
    rt2 = smt.simplify(smt.Sqrt(2))
    assert type(rt2) == smt.AlgebraicNumRef
    model_value_to_python(rt2)


def test_model_value_to_python_ArithRef():
    # Tests that a plain smt.ArithRef can be exported as Python
    # See https://github.com/pschanely/CrossHair/issues/381
    rt2 = smt.ToInt(2 ** smt.Int("x"))
    print("type(rt2)", type(rt2))
    assert type(rt2) == smt.ArithRef
    model_value_to_python(rt2)


def test_model_value_to_python_unfolded_power():
    # z3's simplifier leaves a ground power above max_degree=64 unevaluated;
    # realization must refold it rather than return None.
    unfolded = smt.simplify(smt.ToInt(smt.IntVal(7) ** smt.IntVal(200)))
    assert not smt.is_int_value(unfolded)
    assert model_value_to_python(unfolded) == 7**200


def test_model_value_to_python_refuses_astronomical_int():
    huge = smt.ToInt(smt.IntVal(2) ** smt.IntVal(5_000_000))
    with pytest.raises(CrosshairUnsupported):
        model_value_to_python(huge)


def test_smt_fanout(space: SimpleStateSpace):
    option1 = smt.Bool("option1")
    option2 = smt.Bool("option2")
    space.add(smt.Xor(option1, option2))  # Ensure exactly one option can be set
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
    x = smt.Int("x")
    space.solver.add(x > 5)
    space.solver.add(x < 3)
    with pytest.raises(CrossHairInternal) as exc_info:
        space._raise_unexpected_realization_unsat(x, None)
    message = str(exc_info.value)
    assert "fresh same-config solver.check(): unsat" in message
    assert "minimal unsat core:" in message
    assert "x > 5" in message
    assert "x < 3" in message


def _run_in_space(space, fn):
    with Patched(), StateSpaceContext(space), COMPOSITE_TRACER, NoTracing():
        return fn()


@pytest.mark.parametrize(
    "add_constraint",
    [
        lambda space, x, v: space.add(x != v),
        lambda space, x, v: space.solver.add(x != v),
        lambda space, x, v: space.solver.from_string(
            f"(declare-const x Int) (assert (not (= x {v})))"
        ),
    ],
)
def test_nondeterministic_realization_is_detected(add_constraint):
    root = RootNode()
    x, y = smt.Int("x"), smt.Int("y")

    def first_iteration():
        return space1.find_model_value(x)

    space1 = StateSpace(time.monotonic() + 1000, float("+inf"), root)
    realized_x = _run_in_space(space1, first_iteration)

    def second_iteration():
        assert space2.witness() is not None
        add_constraint(space2, x, realized_x)
        assert space2.find_model_value(x) == realized_x
        space2.find_model_value(y)

    space2 = StateSpace(time.monotonic() + 1000, float("+inf"), root)
    with pytest.raises(CrossHairInternal, match="Unexpected unsat"):
        _run_in_space(space2, second_iteration)


def test_is_possible_does_not_affect_realization():
    x = smt.Int("x")

    def realize(check_first):
        space = SimpleStateSpace()

        def run():
            if check_first:
                assert space.is_possible(x == 7)
            return space.find_model_value(x)

        return _run_in_space(space, run)

    assert realize(check_first=True) == realize(check_first=False)
