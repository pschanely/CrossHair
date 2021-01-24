from dataclasses import replace
import types

from crosshair.core import analyze_function
from crosshair.core import AnalysisMessage
from crosshair.core import AnalysisOptions
from crosshair.core import MessageType
from crosshair.core import DEFAULT_OPTIONS
from typing import *

ComparableLists = Tuple[List, List]

def check_fail(fn: Callable, options: AnalysisOptions=DEFAULT_OPTIONS) -> ComparableLists:
    if options.max_iterations == DEFAULT_OPTIONS.max_iterations:
        options = replace(options, max_iterations=20)
    if options.per_condition_timeout == DEFAULT_OPTIONS.per_condition_timeout:
        options = replace(options, per_condition_timeout=5)
    messages = analyze_function(fn, options) if options else analyze_function(fn)
    return ([m.state for m in messages], [MessageType.POST_FAIL])


def check_exec_err(fn: Callable,
                   message_prefix='',
                   options: AnalysisOptions=DEFAULT_OPTIONS) -> ComparableLists:
    if options.max_iterations == DEFAULT_OPTIONS.max_iterations:
        options = replace(DEFAULT_OPTIONS, max_iterations=20)
    messages = analyze_function(fn, options)
    if all(m.message.startswith(message_prefix) for m in messages):
        return ([m.state for m in messages], [MessageType.EXEC_ERR])
    else:
        return ([(m.state, m.message) for m in messages], [(MessageType.EXEC_ERR, message_prefix)])


def check_post_err(fn: Callable,
                   options: AnalysisOptions=DEFAULT_OPTIONS) -> ComparableLists:
    if options.max_iterations == DEFAULT_OPTIONS.max_iterations:
        options = replace(DEFAULT_OPTIONS, max_iterations=20)
    return ([m.state for m in analyze_function(fn, options)], [MessageType.POST_ERR])


def check_unknown(fn: Callable,
                  options: AnalysisOptions=DEFAULT_OPTIONS) -> ComparableLists:
    if options.max_iterations == DEFAULT_OPTIONS.max_iterations:
        options = replace(DEFAULT_OPTIONS, max_iterations=40)
    return ([(m.state, m.message, m.traceback) for m in analyze_function(fn, options)],
            [(MessageType.CANNOT_CONFIRM, 'Not confirmed.', '')])


def check_ok(fn: Callable, options: AnalysisOptions=DEFAULT_OPTIONS) -> ComparableLists:
    if options.per_condition_timeout == DEFAULT_OPTIONS.per_condition_timeout:
        options = replace(options, per_condition_timeout=5)
    messages = analyze_function(fn, options) if options else analyze_function(fn)
    messages = [m for m in messages if m.state != MessageType.CONFIRMED]
    return (messages, [])


def check_messages(msgs: List[AnalysisMessage], **kw) -> ComparableLists:
    if kw.get('state') != MessageType.CONFIRMED:
        # Normally, ignore confirmation messages:
        msgs = [m for m in msgs if m.state != MessageType.CONFIRMED]
    else:
        # When we are checking confirmation, just check one:
        msgs = [msgs[0]]
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
