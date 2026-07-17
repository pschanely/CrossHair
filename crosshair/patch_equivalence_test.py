import codecs
import copy
import itertools
import re
import time
from typing import Callable, Sequence

import pytest  # type: ignore

from crosshair.behavior_compare import summarize_execution
from crosshair.core import _PATCH_REGISTRATIONS
from crosshair.core_and_libs import standalone_statespace
from crosshair.inputgen import valid_inputs
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
    # Order-dependent output: CrossHair realizes a set/frozenset in solver order,
    # which need not match CPython's hash order, so repr strings and the popped
    # element legitimately differ.  (Value-comparable set ops -- union, difference,
    # membership -- are still tested; these three are not value functions.)
    set.pop,
    set.__repr__,
    frozenset.__repr__,
    # CrossHair intentionally no-ops sleep; it doesn't replicate CPython's
    # OverflowError for out-of-range durations.
    time.sleep,
    # All three return codec callables (getencoder/getdecoder a bare encode/decode
    # function; lookup a CodecInfo OF callables) where native is a C builtin and
    # patched is CrossHair's Python reimpl -- behaviorally equivalent but unequal
    # by identity, and not fixable with __eq__ (the patched lookup result is a
    # plain CodecInfo, sometimes from the crosshair_<enc> codec and always
    # flattened by deep_realize, so a subclass __eq__ never applies).  "Both return
    # a callable" is a vacuous check; real coverage comes from the codecs.encode /
    # codecs.decode op cells.  TODO(equivalence): check lookup by codec *name* via
    # a CodecInfo-aware rule in flexible_equal (covers both plain + realized).
    codecs.getencoder,
    codecs.getdecoder,
    codecs.lookup,
}


def _fn_sortkey(kv):
    fn = kv[0]
    return (
        getattr(fn, "__module__", "") or "",
        getattr(fn, "__qualname__", None) or getattr(fn, "__name__", ""),
    )


def _params():
    params = []
    # sorted() so collection order is deterministic across processes: pytest-xdist
    # aborts if its workers collect tests in different orders. (_PATCH_REGISTRATIONS
    # is insertion-ordered, but sort by a stable function id defensively.)
    for native_fn, patched_fn in sorted(_PATCH_REGISTRATIONS.items(), key=_fn_sortkey):
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
