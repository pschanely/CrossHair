import sys
from copy import copy, deepcopy

from crosshair import NoTracing, register_patch
from crosshair.util import CrossHairValue


def _copy(x):
    with NoTracing():
        if isinstance(x, CrossHairValue):
            return copy(x)
    return copy(x)


# Mirror CPython's deepcopy signature so argument-count errors match: the
# private ``_nil`` sentinel parameter was removed in Python 3.15.
if sys.version_info >= (3, 15):

    def _deepcopy(x, memo=None):
        with NoTracing():
            if isinstance(x, CrossHairValue):
                return deepcopy(x, memo)
        return deepcopy(x, memo)

else:

    def _deepcopy(x, memo=None, _nil=[]):
        with NoTracing():
            if isinstance(x, CrossHairValue):
                return deepcopy(x, memo)
        return deepcopy(x, memo)


def make_registrations() -> None:
    register_patch(copy, _copy)
    register_patch(deepcopy, _deepcopy)
