from crosshair.libimpl import builtinslib
from crosshair.libimpl import collectionslib
from crosshair.libimpl import datetimelib
from crosshair.libimpl import mathlib
from crosshair.libimpl import randomlib
from crosshair.libimpl import relib


def make_registrations():
    builtinslib.make_registrations()
    collectionslib.make_registrations()
    datetimelib.make_registrations()
    mathlib.make_registrations()
    randomlib.make_registrations()
    relib.make_registrations()

    # We monkey patch icontract below to prevent it from enforcing contracts.
    # (we want to control how and when they run)
    # TODO: consider a better home for this code
    try:
        import icontract

        icontract._checkers._assert_invariant = lambda *a, **kw: None
        icontract._checkers._assert_preconditions = lambda *a, **kw: None
        icontract._checkers._assert_postconditions = lambda *a, **kw: None
    except ImportError:
        pass
