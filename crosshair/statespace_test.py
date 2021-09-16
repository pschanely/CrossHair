import z3  # type: ignore

from crosshair.statespace import (
    StateSpace,
    SimpleStateSpace,
    HeapRef,
    SinglePathNode,
    SnapshotRef,
)


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


def TODO_test_checkpoint():
    space = SimpleStateSpace()
    listref = z3.Const("listref", HeapRef)
    listval = space.find_key_in_heap(listref, list, lambda t: [], _HEAD_SNAPSHOT)
    space.checkpoint()
    listval_at_head = space.find_key_in_heap(
        listref, list, lambda t: [], _HEAD_SNAPSHOT
    )
    assert listval is not listval_at_head
