from dataclasses import replace
import types

from crosshair.core import analyze_function
from crosshair.core import AnalysisMessage
from crosshair.core import MessageType
from typing import *

ComparableLists = Tuple[List, List]

def check_fail(fn: Callable) -> ComparableLists:
    return ([m.state for m in analyze_function(fn)], [MessageType.POST_FAIL])


def check_exec_err(fn: Callable, message_prefix='') -> ComparableLists:
    messages = analyze_function(fn)
    if all(m.message.startswith(message_prefix) for m in messages):
        return ([m.state for m in messages], [MessageType.EXEC_ERR])
    else:
        return ([(m.state, m.message) for m in messages], [(MessageType.EXEC_ERR, message_prefix)])


def check_post_err(fn: Callable) -> ComparableLists:
    return ([m.state for m in analyze_function(fn)], [MessageType.POST_ERR])


def check_unknown(fn: Callable) -> ComparableLists:
    return ([(m.state, m.message, m.traceback) for m in analyze_function(fn)],
            [(MessageType.CANNOT_CONFIRM, 'I cannot confirm this', '')])


def check_ok(fn: Callable) -> ComparableLists:
    return (analyze_function(fn), [])


def check_messages(msgs: List[AnalysisMessage], **kw) -> ComparableLists:
    default_msg = AnalysisMessage(MessageType.CANNOT_CONFIRM, '', '', 0, 0, '')
    msg = msgs[0] if msgs else replace(default_msg)
    fields = ('state', 'message', 'filename', 'line', 'column', 'traceback',
              'execution_log', 'test_fn', 'condition_src')
    for k in fields:
        if k not in kw:
            default_val = getattr(default_msg, k)
            msg = replace(msg, **{k: default_val})
            kw[k] = default_val
    if msgs:
        msgs[0] = msg
    return (msgs, [AnalysisMessage(**kw)])
