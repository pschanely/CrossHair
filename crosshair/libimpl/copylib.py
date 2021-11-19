from copy import copy
from copy import deepcopy
from crosshair.core import CrossHairValue

from crosshair import register_patch
from crosshair import NoTracing


def _copy(x):
    with NoTracing():
        if isinstance(x, CrossHairValue):
            return copy(x)
    return copy(x)


def _deepcopy(x, memo=None, _nil=[]):
    with NoTracing():
        if isinstance(x, CrossHairValue):
            return deepcopy(x, memo)
    return deepcopy(x, memo)


def make_registrations() -> None:
    register_patch(copy, _copy)
    register_patch(deepcopy, _deepcopy)
