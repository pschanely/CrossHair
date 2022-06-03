import pytest
from pygls.lsp.types import Diagnostic, DiagnosticSeverity, Position, Range

from crosshair.lsp_server import LocalState, create_lsp_server, get_diagnostic
from crosshair.options import AnalysisOptionSet
from crosshair.statespace import AnalysisMessage, MessageType
from crosshair.watcher import Watcher


@pytest.fixture
def state():
    server = create_lsp_server(AnalysisOptionSet())
    return LocalState(Watcher([]), server)


def test_get_diagnostic():
    msg = AnalysisMessage(MessageType.POST_ERR, "exception raised", __file__, 2, 0, "")
    assert get_diagnostic(msg, ["one", "  two  ", "three"]) == Diagnostic(
        range=Range(
            start=Position(line=1, character=2),
            end=Position(line=1, character=5),
        ),
        severity=DiagnosticSeverity.Error,
        message="exception raised",
        source="CrossHair",
    )


def test_watch_loop(state: LocalState):
    state.run_watch_loop(max_watch_iterations=1)
