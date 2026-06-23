import copy
import itertools
import re
import time
from typing import Callable, List, Sequence, Tuple

import pytest  # type: ignore

from crosshair.core import _PATCH_REGISTRATIONS
from crosshair.core_and_libs import standalone_statespace
from crosshair.inputgen import valid_inputs
from crosshair.test_util import summarize_execution
from crosshair.util import debug

# Curated edge / invalid inputs -- shared across all patches to exercise
# failure-mode parity (wrong types, error paths). These deliberately mostly
# DON'T apply to any given patch; per-patch *valid* inputs come from
# ``valid_inputs`` below.
edge_args = [
    (),
    (0,),
    (None,),
    ("a",),
    ("ab", "b"),  # str endwith, index
    (b"a",),
    (b"ab", "little"),  # int.from_bytes
    ("ǔ", "latin-1", "replace"),  # encode, bytes constructor
    (b"ab", b"b"),  # bytes endwith, index
    (b"x", [b"a", b"b"]),  # bytes join
    (bytearray(b"x"), [b"a", bytearray(b"b")]),  # mixed bytearray join
    ([2, 1],),  # min, max
    (1, 2),
    (int, object),  # issubclass
    (int, (str, (tuple, list))),  # wacky multiply-nested issubclass checking
    (int, 42),  # issubclass error
    (42, int),  # isinstance
    (re.compile("(ab|a|b)"), r"\n", ""),  # re methods
    (bool, [1, 1, 0]),  # itertools.takewhile and friends
    ([(1, 2), (3, 4)]),  # key-value pairs
    ([(1, 2), ([], 4)]),  # key-value pairs w/ unhashable key
]

untested_patches = {
    itertools.groupby,  # the return value has nested iterators that break comparisons
    # CrossHair intentionally no-ops sleep; it doesn't replicate CPython's
    # OverflowError for out-of-range durations.
    time.sleep,
}


def _params() -> List:
    params = []
    for native_fn, patched_fn in _PATCH_REGISTRATIONS.items():
        if native_fn in untested_patches:
            continue
        name = getattr(native_fn, "__qualname__", native_fn.__name__)
        try:
            typed = valid_inputs(
                native_fn, k=3
            )  # per-patch valid inputs (deterministic)
        except Exception:
            typed = []
        for i, args in enumerate(edge_args + typed):
            argid = re.sub(r"[\W_]+", "_", str(args))[:40]
            params.append(
                pytest.param(native_fn, patched_fn, args, id=f"{name}-{argid}")
            )
    return params


@pytest.mark.parametrize("native_fn,patched_fn,args", _params())
def test_patch(native_fn: Callable, patched_fn: Callable, args: Sequence[object]):
    """
    Tests that the builtin and standard library patches behave like their
    counterparts, when given concrete python input values.
    Equivalence under symbolic inputs is tested in "_ch_test.py" files.
    "Behavior" includes values returned and exceptions thrown, so you'll
    need to make sure the patch does argument checking in the same order
    as CPython.

    Inputs come from two sources: a shared curated set of edge/invalid tuples
    (``edge_args``, for failure-mode parity) and per-patch valid inputs
    synthesized from typeshed signatures (``crosshair.inputgen.valid_inputs``).
    """
    debug("Patch test:", native_fn, patched_fn)
    debug("Args:", args)
    native_result = summarize_execution(
        native_fn, copy.deepcopy(args), {}, detach_path=False
    )
    debug("Native result:  ", native_result)
    with standalone_statespace:
        patched_result = summarize_execution(
            patched_fn, copy.deepcopy(args), {}, detach_path=False
        )
    debug("Patched result: ", patched_result)
    assert native_result == patched_result
