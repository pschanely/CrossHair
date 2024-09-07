import copy
import itertools
import re
import sys
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import pytest  # type: ignore

from crosshair.core import _PATCH_REGISTRATIONS
from crosshair.core_and_libs import standalone_statespace
from crosshair.test_util import ExecutionResult, summarize_execution
from crosshair.util import ch_stack, debug

"""
Tests that the builtin and standard library patches behave like their
counterparts, for native python input values.
Equivalence under symbolic inputs is tested in "_ch_test.py" files.
"""


possible_args = [
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
}

comparisons: List[Tuple[Callable, Callable]] = []
for native_fn, patched_fn in _PATCH_REGISTRATIONS.items():
    if native_fn in untested_patches:
        continue
    patch_name = native_fn.__name__
    comparisons.append(
        pytest.param(native_fn, patched_fn, id=patch_name)  # type: ignore
    )


@pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="Awaiting CPython fix in https://github.com/python/cpython/issues/122888",
)
@pytest.mark.parametrize("native_fn,patched_fn", comparisons)
@pytest.mark.parametrize(
    "args", possible_args, ids=lambda t: re.sub(r"[\W_]", "_", str(t))
)
def test_patch(native_fn: Callable, patched_fn: Callable, args: Sequence[object]):
    debug("Patch test:", native_fn, patched_fn)
    debug("Args:", args)
    args2 = copy.deepcopy(args)
    native_result = summarize_execution(native_fn, args, {}, detach_path=False)
    debug("Native result:  ", native_result)
    with standalone_statespace:
        patched_result = summarize_execution(patched_fn, args2, {}, detach_path=False)
    debug("Patched result: ", patched_result)
    assert native_result == patched_result
