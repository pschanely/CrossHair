import time

import pytest
import z3  # type: ignore

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
from crosshair.tracers import COMPOSITE_TRACER
from crosshair.util import UnknownSatisfiability

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
    space = StateSpace(time.monotonic() + 60_000, 0.1, RootNode())
    with pytest.raises(UnknownSatisfiability):
        with Patched(), StateSpaceContext(space), COMPOSITE_TRACER:
            ints = [proxy_for_type(int, f"i{i}") for i in range(num_ints)]
            for i in range(num_ints - 2):
                t0 = time.monotonic()
                if ints[i] * ints[i + 1] == ints[i + 2]:
                    pass
                ints[i + 1] += ints[i]
    solve_time = time.monotonic() - t0
    assert 0.05 < solve_time < 0.2


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
