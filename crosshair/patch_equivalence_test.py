import copy
import itertools
import re
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import pytest  # type: ignore

from crosshair.core import _PATCH_REGISTRATIONS
from crosshair.core_and_libs import standalone_statespace
from crosshair.test_util import ExecutionResult, summarize_execution
from crosshair.util import debug, test_stack

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


@dataclass(init=False)
class ExecutionResultWithTb:
    ret: object  # return value
    exc: Optional[BaseException]
    tb: Optional[str]
    post_args: Sequence
    post_kwargs: Mapping[str, object]

    def __init__(self, result: ExecutionResult):
        self.ret = result.ret
        self.exc = result.exc
        self.tb = test_stack(self.exc.__traceback__) if self.exc else None
        self.post_args = result.post_args
        self.post_kwargs = result.post_kwargs


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
    if native_result != patched_result:
        assert ExecutionResultWithTb(native_result) == ExecutionResultWithTb(
            patched_result
        )
