"""Register all type handlers and exports core functionality."""

import sys
from typing import List

from packaging import version

from crosshair import opcode_intercept

# These imports are just for exporting functionality:
from crosshair.core import (
    AnalysisMessage,
    MessageType,
    _reset_all_registrations,
    analyze_any,
    analyze_class,
    analyze_function,
    analyze_module,
    proxy_for_type,
    run_checkables,
    standalone_statespace,
)

# Modules with registrations:
from crosshair.libimpl import (
    arraylib,
    binasciilib,
    builtinslib,
    codecslib,
    collectionslib,
    copylib,
    datetimelib,
    decimallib,
    functoolslib,
    hashliblib,
    heapqlib,
    importliblib,
    iolib,
    ipaddresslib,
    itertoolslib,
    jsonlib,
    mathlib,
    oslib,
    randomlib,
    relib,
    timelib,
    typeslib,
    unicodedatalib,
    urlliblib,
    zliblib,
)
from crosshair.options import AnalysisKind, AnalysisOptions
from crosshair.tracers import NoTracing, ResumedTracing
from crosshair.util import debug

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


installed_plugins: List[str] = []  # We record these for diagnostic purposes
__all__ = [
    "analyze_function",
    "analyze_any",
    "analyze_class",
    "analyze_module",
    "run_checkables",
    "proxy_for_type",
    "standalone_statespace",
    "AnalysisMessage",
    "MessageType",
    "AnalysisKind",
    "AnalysisOptions",
    "NoTracing",
    "ResumedTracing",
    "debug",
]


def _make_registrations():
    _reset_all_registrations()
    arraylib.make_registrations()
    binasciilib.make_registrations()
    builtinslib.make_registrations()
    codecslib.make_registrations()
    collectionslib.make_registrations()
    copylib.make_registrations()
    datetimelib.make_registrations()
    decimallib.make_registrations()
    functoolslib.make_registrations()
    hashliblib.make_registrations()
    heapqlib.make_registrations()
    jsonlib.make_registrations()
    importliblib.make_registrations()
    iolib.make_registrations()
    ipaddresslib.make_registrations()
    itertoolslib.make_registrations()
    mathlib.make_registrations()
    oslib.make_registrations()
    randomlib.make_registrations()
    relib.make_registrations()
    timelib.make_registrations()
    typeslib.make_registrations()
    unicodedatalib.make_registrations()
    urlliblib.make_registrations()
    opcode_intercept.make_registrations()
    zliblib.make_registrations()

    plugin_entries = entry_points(group="crosshair.plugin")
    for plugin_entry in plugin_entries:
        installed_plugins.append(plugin_entry.name)
        plugin_entry.load()

    # We monkey patch icontract below to prevent it from enforcing contracts.
    # (we want to control how and when they run)
    # TODO: consider a better home for this code
    try:
        import icontract  # type: ignore

        if version.parse(icontract.__version__) < version.parse("2.4.0"):
            raise Exception("CrossHair requires icontract version >= 2.4.0")

        icontract._checkers._assert_invariant = lambda *a, **kw: None
        icontract._checkers._assert_preconditions = lambda *a, **kw: None
        icontract._checkers._assert_postconditions = lambda *a, **kw: None
    except ImportError:
        pass

    try:
        import deal  # type: ignore

        deal_version = version.parse(deal.__version__)
        if deal_version < version.parse("4.13.0"):
            raise Exception("CrossHair requires deal version >= 4.13.0")
        if deal_version < version.parse("4.21.2"):
            deal.disable()
        else:
            # deal >= 4.21.2 throws runtime warnings.
            deal.disable(warn=False)
    except ImportError:
        pass


_make_registrations()
