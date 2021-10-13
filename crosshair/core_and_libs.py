"""Register all type handlers and exports core functionality."""

from packaging import version
import sys
from typing import List

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
from crosshair.util import debug

# Modules with registrations:
from crosshair.libimpl import arraylib
from crosshair.libimpl import builtinslib
from crosshair.libimpl import collectionslib
from crosshair.libimpl import datetimelib
from crosshair.libimpl import jsonlib
from crosshair.libimpl import mathlib
from crosshair.libimpl import randomlib
from crosshair.libimpl import relib
from crosshair import opcode_intercept

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


installed_plugins: List[str] = []  # We record these for diagnostic purposes


def _make_registrations():
    arraylib.make_registrations()
    builtinslib.make_registrations()
    collectionslib.make_registrations()
    datetimelib.make_registrations()
    jsonlib.make_registrations()
    mathlib.make_registrations()
    randomlib.make_registrations()
    relib.make_registrations()
    opcode_intercept.make_registrations()

    plugin_entries = entry_points(group="crosshair.plugin")
    for plugin_entry in plugin_entries:
        installed_plugins.append(plugin_entry.name)
        plugin_entry.load()

    # We monkey patch icontract below to prevent it from enforcing contracts.
    # (we want to control how and when they run)
    # TODO: consider a better home for this code
    try:
        import icontract

        if version.parse(icontract.__version__) < version.parse("2.4.0"):
            raise Exception("CrossHair requires icontract version >= 2.4.0")

        icontract._checkers._assert_invariant = lambda *a, **kw: None
        icontract._checkers._assert_preconditions = lambda *a, **kw: None
        icontract._checkers._assert_postconditions = lambda *a, **kw: None
    except ImportError:
        pass

    try:
        import deal

        if version.parse(deal.__version__) < version.parse("4.13.0"):
            raise Exception("CrossHair requires deal version >= 4.13.0")
        deal.disable()
    except ImportError:
        pass

    # Set hypothesis to run in a minimal mode.
    # (auditwall will yell if hypothesis tries to write to disk)
    # TODO: figure out some other way to set options via fuzz_one_input.
    try:
        from hypothesis import settings, Phase

        settings.register_profile("ch", database=None, phases=[Phase.generate])
        settings.load_profile("ch")
    except ImportError:
        pass


_make_registrations()
