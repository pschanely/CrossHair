"""Register all type handlers and exports core functionality."""

from crosshair.core import analyze_function
from crosshair.core import analyze_any
from crosshair.core import analyze_class
from crosshair.core import analyze_module
from crosshair.core import run_checkables
from crosshair.core import AnalysisMessage
from crosshair.core import MessageType
from crosshair.options import AnalysisKind
from crosshair.options import AnalysisOptions

from crosshair.libimpl import make_registrations as _make_registrations

_make_registrations()
