from dataclasses import dataclass
from dataclasses import astuple
import re
from typing import *

import pytest  # type: ignore

from crosshair.core import _PATCH_REGISTRATIONS
import crosshair.core_and_libs
from crosshair.test_util import summarize_execution
from crosshair.test_util import ExecutionResult
from crosshair.util import debug
from crosshair.util import test_stack

'''
Tests that the builtin and standard library patches behave like their
counterparts, for native python input values.
Equivalence under symbolic inputs is tested separately.
'''


possible_args = [
    (),
    (0,),
    (None,),
    ('a',),
    ('ab', 'b'), # str endwith, index
    ([2, 1],), # min, max
    (1, 2),
    (int, object), # issubclass
    (int, 42), # issubclass error
    (42, int), # isinstance
]

comparisons: List[Tuple[Callable, Callable]] = []
for id_wrapper, attrmap in _PATCH_REGISTRATIONS.items():
    for attr_name, patched_fn in attrmap.items():
        native_fn = getattr(id_wrapper.get(), attr_name)
        patch_name = id_wrapper.get().__name__ + '.' + attr_name
        comparisons.append(pytest.param(native_fn, patched_fn, id=patch_name))


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
@pytest.mark.parametrize("args", possible_args, ids=lambda t: re.sub(r'[\W_]', '_', str(t)))
def test_patch(native_fn: Callable,
               patched_fn: Callable,
               args: Sequence[object]):
    debug('String patch test:', native_fn, patched_fn)
    debug('Computing native result on args:', args)
    native_result = summarize_execution(native_fn, args, {})
    debug('Computing patched result on args:', args)
    patched_result = summarize_execution(patched_fn, args, {})
    if native_result != patched_result:
        assert (ExecutionResultWithTb(native_result) ==
                ExecutionResultWithTb(patched_result))

