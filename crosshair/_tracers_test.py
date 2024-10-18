import dis
import gc
import sys
from typing import List

import pytest

from _crosshair_tracers import (  # type: ignore
    CTracer,
    code_stack_depths,
    frame_stack_read,
)
from crosshair.util import mem_usage_kb


class ExampleModule:
    opcodes_wanted = frozenset([42, 255])


def test_CTracer_module_refcounts_dont_leak():
    mod = ExampleModule()
    assert sys.getrefcount(mod) == 2
    tracer = CTracer()
    tracer.push_module(mod)
    assert sys.getrefcount(mod) == 3
    tracer.push_module(mod)
    tracer.start()
    tracer.stop()
    assert sys.getrefcount(mod) == 4
    tracer.pop_module(mod)
    assert sys.getrefcount(mod) == 3
    del tracer
    gc.collect()
    assert sys.getrefcount(mod) == 2


def _get_depths(fn):
    # dis.dis(fn)
    depths = code_stack_depths(fn.__code__)
    for instr in dis.Bytecode(fn):
        wordpos = instr.offset // 2
        depth = depths[wordpos]
        if depth != -9:
            assert depth >= 0
            assert depth + dis.stack_effect(instr.opcode, instr.arg) >= 0
    return depths


class RawNull:
    def __repr__(self):
        return "_RAW_NULL"


_RAW_NULL = RawNull()


def _log_execution_stacks(fn, *a, **kw):
    depths = _get_depths(fn)
    stacks = []

    def _tracer(frame, event, arg):
        if event == "opcode":
            lasti = frame.f_lasti
            opcode = frame.f_code.co_code[lasti]
            oparg = frame.f_code.co_code[lasti + 1]  # TODO: account for EXTENDED_ARG
            opname = dis.opname[opcode]
            entry: List = [f"{opname}({oparg})"]
            for i in range(-depths[lasti // 2], 0, 1):
                try:
                    entry.append(frame_stack_read(frame, i))
                except ValueError:
                    entry.append(_RAW_NULL)
            stacks.append(tuple(entry))
        frame.f_trace = _tracer
        frame.f_trace_opcodes = True
        return _tracer

    old_tracer = sys.gettrace()
    # Caller needs opcode tracing since Python 3.12; see https://github.com/python/cpython/issues/103615
    sys._getframe().f_trace_opcodes = True

    sys.settrace(_tracer)
    try:
        result = (fn(*a, **kw), None)
    except Exception as exc:
        result = (None, exc)
    finally:
        sys.settrace(old_tracer)
    return stacks


@pytest.mark.skipif(sys.version_info < (3, 12), reason="stack depth on 3.12+")
def test_one_function_stack_depth():
    _E = (TypeError, KeyboardInterrupt)

    def a(x):
        return {k for k in (35, x)}

    # just enure no crashes:
    _log_execution_stacks(a, 4)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="stack depth on 3.12+")
def test_stack_get():
    def to_be_traced(x):
        r = 8 - x
        return 9 - r

    stacks = _log_execution_stacks(to_be_traced, 3)
    assert ("BINARY_OP(10)", 8, 3) in stacks
    assert ("BINARY_OP(10)", 9, 5) in stacks


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="getrusage not available on windows"
)
def test_CTracer_does_not_leak_memory():
    import resource  # (available only on unix; delay import)

    for i in range(1_000):
        tracer = CTracer()
        tracer.start()
        mods = [ExampleModule() for _ in range(6)]
        for mod in mods:
            tracer.push_module(mod)
        for mod in reversed(mods):
            tracer.pop_module(mod)
        tracer.stop()
        if i == 100:
            usage = mem_usage_kb()
    usage_increase = mem_usage_kb() - usage
    assert usage_increase < 200
