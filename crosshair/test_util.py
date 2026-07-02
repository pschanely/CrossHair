import pathlib
import sys
from dataclasses import replace
from typing import Callable, Iterable, List, Tuple

from crosshair.core import (
    AnalysisMessage,
    Checkable,
    MessageType,
    analyze_function,
    run_checkables,
)
from crosshair.options import AnalysisOptionSet
from crosshair.util import in_debug

ComparableLists = Tuple[List, List]


def simplefs(path: pathlib.Path, files: dict) -> None:
    for name, contents in files.items():
        subpath = path / name
        if isinstance(contents, str):
            with open(subpath, "w") as fh:
                fh.write(contents)
        elif isinstance(contents, dict):
            subpath.mkdir()
            simplefs(subpath, contents)
        else:
            raise Exception("bad input to simplefs")


def check_states(
    fn: Callable,
    expected: MessageType,
    optionset: AnalysisOptionSet = AnalysisOptionSet(),
) -> None:
    if expected == MessageType.POST_FAIL:
        local_opts = AnalysisOptionSet(
            per_condition_timeout=16,
            max_uninteresting_iterations=sys.maxsize,
        )
    elif expected == MessageType.CONFIRMED:
        local_opts = AnalysisOptionSet(
            per_condition_timeout=60,
            per_path_timeout=20,
            max_uninteresting_iterations=sys.maxsize,
        )
    elif expected == MessageType.POST_ERR:
        local_opts = AnalysisOptionSet(max_iterations=20)
    elif expected == MessageType.CANNOT_CONFIRM:
        local_opts = AnalysisOptionSet(
            max_uninteresting_iterations=40,
            per_condition_timeout=3,
        )
    else:
        local_opts = AnalysisOptionSet(
            max_uninteresting_iterations=40,
            per_condition_timeout=5,
        )
    options = local_opts.overlay(optionset)
    found = set([m.state for m in run_checkables(analyze_function(fn, options))])
    assertmsg = f"Got {','.join(map(str, found))} instead of {expected}"
    if not in_debug():
        assertmsg += " (use `pytest -v` to show trace)"
    assert found == {expected}, assertmsg


def check_exec_err(
    fn: Callable, message_prefix="", optionset: AnalysisOptionSet = AnalysisOptionSet()
) -> ComparableLists:
    local_opts = AnalysisOptionSet(max_iterations=20)
    options = local_opts.overlay(optionset)
    messages = run_checkables(analyze_function(fn, options))
    if all(m.message.startswith(message_prefix) for m in messages):
        return ([m.state for m in messages], [MessageType.EXEC_ERR])
    else:
        return (
            [(m.state, m.message) for m in messages],
            [(MessageType.EXEC_ERR, message_prefix)],
        )


def check_messages(checkables: Iterable[Checkable], **kw) -> ComparableLists:
    msgs = run_checkables(checkables)
    if kw.get("state") != MessageType.CONFIRMED:
        # Normally, ignore confirmation messages:
        msgs = [m for m in msgs if m.state != MessageType.CONFIRMED]
    else:
        # When we want CONFIRMED, take the message with the worst status:
        msgs = [max(msgs, key=lambda m: m.state)]
    default_msg = AnalysisMessage(MessageType.CANNOT_CONFIRM, "", "", 0, 0, "")
    msg = msgs[0] if msgs else replace(default_msg)
    fields = (
        "state",
        "message",
        "filename",
        "line",
        "column",
        "traceback",
        "test_fn",
        "condition_src",
    )
    for k in fields:
        if k not in kw:
            default_val = getattr(default_msg, k)
            msg = replace(msg, **{k: default_val})
            kw[k] = default_val
    if msgs:
        msgs[0] = msg
    return (msgs, [AnalysisMessage(**kw)])
