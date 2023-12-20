"""Provide access to and overrides for functions as they are called."""

import ctypes
import dataclasses
import dis
import sys
from collections import defaultdict
from sys import _getframe
from types import CodeType
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple

import opcode

from _crosshair_tracers import CTracer, TraceSwap

USE_C_TRACER = True

PyObjPtr = ctypes.POINTER(ctypes.py_object)
Py_IncRef = ctypes.pythonapi.Py_IncRef
Py_DecRef = ctypes.pythonapi.Py_DecRef


_debug_header: Tuple[Tuple[str, type], ...] = (
    (
        ("_ob_next", PyObjPtr),
        ("_ob_prev", PyObjPtr),
    )
    if sys.flags.debug
    else ()
)


from _crosshair_tracers import frame_stack_read, frame_stack_write

CALL_FUNCTION = dis.opmap.get("CALL_FUNCTION", 256)
CALL_FUNCTION_KW = dis.opmap.get("CALL_FUNCTION_KW", 256)
CALL_FUNCTION_EX = dis.opmap.get("CALL_FUNCTION_EX", 256)
CALL_METHOD = dis.opmap.get("CALL_METHOD", 256)
BUILD_TUPLE_UNPACK_WITH_CALL = dis.opmap.get("BUILD_TUPLE_UNPACK_WITH_CALL", 256)
CALL = dis.opmap.get("CALL", 256)

NULL_POINTER = object()


def handle_build_tuple_unpack_with_call(frame) -> Optional[Tuple[int, Callable]]:
    idx = -(
        frame.f_code.co_code[frame.f_lasti + 1] + 1
    )  # TODO: account for EXTENDED_ARG, here and elsewhere
    try:
        return (idx, frame_stack_read(frame, idx))
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call(frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 1)
    try:
        ret = (idx - 1, frame_stack_read(frame, idx - 1))
    except ValueError:
        ret = (idx, frame_stack_read(frame, idx))
    return ret


def handle_call_function(frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 1)
    try:
        return (idx, frame_stack_read(frame, idx))
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call_function_kw(frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 2)
    try:
        return (idx, frame_stack_read(frame, idx))
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call_function_ex(frame) -> Optional[Tuple[int, Callable]]:
    idx = -((frame.f_code.co_code[frame.f_lasti + 1] & 1) + 2)
    try:
        return (idx, frame_stack_read(frame, idx))
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call_method(frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 2)
    try:
        return (idx, frame_stack_read(frame, idx))
    except ValueError:
        # not a sucessful method lookup; no call happens here
        idx += 1
        return (idx, frame_stack_read(frame, idx))


_CALL_HANDLERS: Dict[int, Callable[[object], Optional[Tuple[int, Callable]]]] = {
    BUILD_TUPLE_UNPACK_WITH_CALL: handle_build_tuple_unpack_with_call,
    CALL: handle_call,
    CALL_FUNCTION: handle_call_function,
    CALL_FUNCTION_KW: handle_call_function_kw,
    CALL_FUNCTION_EX: handle_call_function_ex,
    CALL_METHOD: handle_call_method,
}


class Untracable:
    pass


class TraceException(BaseException):
    # We extend BaseException instead of Exception, because it won't be considered a
    # user-level exception by CrossHair. (this is for internal assertions)
    pass


class TracingModule:
    # override these!:
    opcodes_wanted = frozenset(_CALL_HANDLERS.keys())

    def __call__(self, frame, codeobj, opcodenum, extra):
        return self.trace_op(frame, codeobj, opcodenum, extra)

    def trace_op(self, frame, codeobj, opcodenum, extra):
        if is_tracing():
            raise TraceException
        if extra is None:
            call_handler = _CALL_HANDLERS.get(opcodenum)
            if not call_handler:
                return None
            maybe_call_info = call_handler(frame)
            if maybe_call_info is None:
                # TODO: this cannot happen?
                return None
            (fn_idx, target) = maybe_call_info
            binding_target = None

            try:
                __self = object.__getattribute__(target, "__self__")
            except AttributeError:
                pass
            else:
                try:
                    __func = object.__getattribute__(target, "__func__")
                except AttributeError:
                    # The implementation is likely in C.
                    # Attempt to get a function via the type:
                    typelevel_target = getattr(type(__self), target.__name__, None)
                    if typelevel_target is not None:
                        binding_target = __self
                        target = typelevel_target
                else:
                    binding_target = __self
                    target = __func

        else:
            (fn_idx, target, binding_target) = extra
        if isinstance(target, Untracable):
            return None
        replacement = self.trace_call(frame, target, binding_target)
        if replacement is not None:
            target = replacement
            if binding_target is None:
                overwrite_target = target
            else:
                # re-bind a function object if it was originally a bound method
                # on the stack.
                overwrite_target = target.__get__(binding_target, binding_target.__class__)  # type: ignore
            frame_stack_write(frame, fn_idx, overwrite_target)
            return (fn_idx, target, binding_target)
        return extra

    def trace_call(
        self,
        frame: Any,
        fn: Callable,
        binding_target: object,
    ) -> Optional[Callable]:
        return None


TracerConfig = Tuple[Tuple[TracingModule, ...], DefaultDict[int, List[TracingModule]]]


class CompositeTracer:
    def __init__(self):
        self.ctracer = CTracer()

    def push_module(self, module: TracingModule) -> None:
        self.ctracer.push_module(module)

    def trace_caller(self):
        # Frame 0 is the trace_caller method itself
        # Frame 1 is the frame requesting its caller be traced
        # Frame 2 is the caller that we're targeting
        frame = _getframe(2)
        frame.f_trace_opcodes = True
        frame.f_trace = self.ctracer

    def pop_config(self, module) -> None:
        self.ctracer.pop_module(module)

    def set_postop_callback(self, callback, frame):
        self.ctracer.push_postop_callback(frame, callback)

    def __enter__(self) -> object:
        self.old_traceobj = sys.gettrace()
        # Enable opcode tracing before setting trace function, since Python 3.12; see https://github.com/python/cpython/issues/103615
        sys._getframe().f_trace_opcodes = True
        self.ctracer.start()
        return self

    def __exit__(self, _etype, exc, _etb):
        self.ctracer.stop()
        sys.settrace(self.old_traceobj)


# We expect the composite tracer to be used like a singleton.
# (you can only have one tracer active at a time anyway)
# TODO: Thread-unsafe global. Make this a thread local?
COMPOSITE_TRACER = CompositeTracer()


class PatchingModule(TracingModule):
    """Hot-swap functions on the interpreter stack."""

    def __init__(
        self,
        overrides: Optional[Dict[Callable, Callable]] = None,
        fn_type_overrides: Optional[Dict[type, Callable]] = None,
    ):
        self.overrides: Dict[Callable, Callable] = {}
        self.nextfn: Dict[object, Callable] = {}  # code object to next, lower layer
        if overrides:
            self.add(overrides)
        self.fn_type_overrides = fn_type_overrides or {}

    def add(self, new_overrides: Dict[Callable, Callable]):
        for orig, new_override in new_overrides.items():
            prev_override = self.overrides.get(orig, orig)
            self.nextfn[(new_override.__code__, orig)] = prev_override
            self.overrides[orig] = new_override

    def __repr__(self):
        return f"PatchingModule({list(self.overrides.keys())})"

    def trace_call(
        self,
        frame: Any,
        fn: Callable,
        binding_target: object,
    ) -> Optional[Callable]:
        try:
            target = self.overrides.get(fn)
        except TypeError:
            return None
        if target is None:
            fn_type_override = self.fn_type_overrides.get(type(fn))
            if fn_type_override is None:
                return None
            else:
                target = fn_type_override(fn)
        caller_code = frame.f_code
        if caller_code.co_name == "_crosshair_wrapper":
            return None
        target_name = getattr(fn, "__name__", "")
        if target_name.endswith("_crosshair_wrapper"):
            return None
        nextfn = self.nextfn.get((caller_code, fn))
        if nextfn is not None:
            if nextfn is fn:
                return None
            return nextfn
        return target


@dataclasses.dataclass
class CoverageResult:
    offsets_covered: Set[int]
    all_offsets: Set[int]
    opcode_coverage: float


class CoverageTracingModule(TracingModule):
    opcodes_wanted = frozenset(opcode.opmap.values())

    def __init__(self, *fns: Callable):
        assert not is_tracing()
        self.fns = fns
        self.codeobjects = set(fn.__code__ for fn in fns)
        self.opcode_offsets = {
            code: set(i.offset for i in dis.get_instructions(code))
            for code in self.codeobjects
        }
        self.offsets_seen: Dict[CodeType, Set[int]] = defaultdict(set)

    def trace_op(self, frame, codeobj, opcodenum, extra):
        code = frame.f_code
        if code not in self.codeobjects:
            return
        lasti = frame.f_lasti
        assert lasti in self.opcode_offsets[code]
        self.offsets_seen[code].add(lasti)

    def get_results(self, fn: Optional[Callable] = None):
        if fn is None:
            assert len(self.fns) == 1
            fn = self.fns[0]
        possible = self.opcode_offsets[fn.__code__]
        seen = self.offsets_seen[fn.__code__]
        return CoverageResult(
            offsets_covered=seen,
            all_offsets=possible,
            opcode_coverage=len(seen) / len(possible),
        )


class PushedModule:
    def __init__(self, module: TracingModule):
        self.module = module

    def __enter__(self):
        COMPOSITE_TRACER.push_module(self.module)

    def __exit__(self, *a):
        COMPOSITE_TRACER.pop_config(self.module)
        return False


def is_tracing():
    return COMPOSITE_TRACER.ctracer.enabled()


def NoTracing():
    return TraceSwap(COMPOSITE_TRACER.ctracer, True)


def ResumedTracing():
    return TraceSwap(COMPOSITE_TRACER.ctracer, False)
