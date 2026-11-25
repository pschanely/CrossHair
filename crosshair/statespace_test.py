import copy
import re
import time

import pytest

from crosshair import smt
from crosshair.core import Patched, proxy_for_type
from crosshair.statespace import (
    CallAnalysis,
    HeapRef,
    RootNode,
    SimpleStateSpace,
    StateSpace,
    StateSpaceContext,
    SymbolicHeap,
    VerificationStatus,
    prefer_true,
)
from crosshair.tracers import COMPOSITE_TRACER, NoTracing
from crosshair.util import (
    CrossHairInternal,
    UnknownSatisfiability,
)


def test_find_key_in_heap():
    space = SimpleStateSpace()
    listref = smt.const("listref", HeapRef)
    listval1 = space.find_key_in_heap(listref, list, lambda t: [], space.heap)
    assert isinstance(listval1, list)
    listval2 = space.find_key_in_heap(listref, list, lambda t: [], space.heap)
    assert listval1 is listval2
    dictref = smt.const("dictref", HeapRef)
    dictval = space.find_key_in_heap(dictref, dict, lambda t: {}, space.heap)
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
    assert space.solver.is_sat(smt.true)


def test_heap_deepcopy() -> None:
    space = SimpleStateSpace()
    ref = smt.const("ref", HeapRef)

    def find_key(heap):
        return space.find_key_in_heap(ref, list, lambda t: [], heap)

    orig_heap = space.heap
    listval = find_key(orig_heap)
    listval.append(42)
    with NoTracing():
        heap_copy = copy.deepcopy(orig_heap)

    copied_listval = find_key(heap_copy)
    assert copied_listval == [42]
    assert copied_listval is not listval
    assert find_key(orig_heap) is listval
    assert find_key(heap_copy) is copied_listval


def test_heap_lookup_after_mutation_in_other_heap() -> None:
    space = SimpleStateSpace()
    ref = smt.const("ref", HeapRef)

    def find_key(heap):
        return space.find_key_in_heap(ref, list, lambda t: [], heap)

    other_heap = SymbolicHeap()
    find_key(other_heap).append(42)
    assert find_key(space.heap) == []


def test_smt_fanout(space: SimpleStateSpace):
    option1 = smt.const("option1", smt.bool_sort)
    option2 = smt.const("option2", smt.bool_sort)
    space.add(smt.xor(option1, option2))  # Ensure exactly one option can be set
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
    x = smt.const("x", smt.int_sort)
    space.solver.assert_(x > 5)
    space.solver.assert_(x < 3)
    with pytest.raises(CrossHairInternal) as exc_info:
        space._raise_unexpected_realization_unsat(x, None)
    message = str(exc_info.value)
    assert "fresh solver: unsat; minimal unsat core:" in message
    assert "x > 5" in message
    assert "x < 3" in message


def _run_in_space(space, fn):
    with Patched(), StateSpaceContext(space), COMPOSITE_TRACER, NoTracing():
        return fn()


@pytest.mark.parametrize(
    "add_constraint",
    [
        lambda space, x, v: space.add(x != v),
        lambda space, x, v: space.solver.assert_(x != v),
    ],
)
def test_nondeterministic_realization_is_detected(add_constraint):
    root = RootNode()
    x, y = smt.const("x", smt.int_sort), smt.const("y", smt.int_sort)

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
    x = smt.const("x", smt.int_sort)

    def realize(check_first):
        space = SimpleStateSpace()

        def run():
            if check_first:
                assert space.is_possible(x == 7)
            return space.find_model_value(x)

        return _run_in_space(space, run)

    assert realize(check_first=True) == realize(check_first=False)


def test_approximate_path_prevents_confirmation() -> None:
    root = RootNode()
    x = smt.const("x", smt.bool_sort)
    exhausted = False
    num_paths = 0
    while not exhausted:
        space = StateSpace(time.process_time() + 60, 10, root)
        with StateSpaceContext(space):
            space.choose_possible(x)
            if num_paths == 0:
                space.mark_approximate()
            top, exhausted = space.bubble_status(
                CallAnalysis(VerificationStatus.CONFIRMED)
            )
        num_paths += 1
    assert num_paths == 2
    assert top is not None
    assert top.verification_status == VerificationStatus.UNKNOWN


def test_prefer_true_on_concrete_values_without_bool() -> None:
    assert prefer_true(re.match("a", "a")) is True
    assert prefer_true([]) is False
