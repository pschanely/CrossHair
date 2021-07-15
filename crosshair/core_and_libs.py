"""Register all type handlers and exports core functionality."""

# These imports are just for exporting functionality:
from crosshair.core import analyze_function
from crosshair.core import analyze_any
from crosshair.core import analyze_class
from crosshair.core import analyze_module
from crosshair.core import run_checkables
from crosshair.core import proxy_for_type
from crosshair.core import standalone_statespace
from crosshair.core import AnalysisMessage
from crosshair.core import MessageType
from crosshair.options import AnalysisKind
from crosshair.options import AnalysisOptions
from crosshair.tracers import NoTracing
from crosshair.tracers import ResumedTracing

# Modules with registrations:
from crosshair.libimpl import builtinslib
from crosshair.libimpl import collectionslib
from crosshair.libimpl import datetimelib
from crosshair.libimpl import mathlib
from crosshair.libimpl import randomlib
from crosshair.libimpl import relib
from crosshair import opcode_intercept


def _make_registrations():
    builtinslib.make_registrations()
    collectionslib.make_registrations()
    datetimelib.make_registrations()
    mathlib.make_registrations()
    randomlib.make_registrations()
    relib.make_registrations()
    opcode_intercept.make_registrations()

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


_make_registrations()
