"""Provide access to and overrides for functions as they are called."""

import ctypes
import dataclasses
import dis
import sys
import traceback
from collections import defaultdict
from sys import _getframe
from types import CodeType, FrameType
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple

import opcode

from _crosshair_tracers import CTracer, TraceSwap  # type: ignore

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

if sys.version_info >= (3, 11):

    class PyInterpreterFrame(ctypes.Structure):
        _fields_: Tuple[Tuple[str, type], ...] = (
            ("f_func", ctypes.py_object),
            ("f_globals", ctypes.c_void_p),
            ("f_builtins", ctypes.c_void_p),
            ("f_locals", ctypes.c_void_p),
            ("f_code", ctypes.py_object),
            ("frame_obj", ctypes.c_void_p),
            ("previous", ctypes.c_void_p),
            ("prev_instr", ctypes.c_void_p),
            ("stacktop", ctypes.c_int),
            ("is_entry", ctypes.c_bool),
            ("owner", ctypes.c_char),
            ("localsplus", ctypes.py_object * 10_000_000),
        )

    def addrof(ptr):
        return ctypes.cast(ptr, ctypes.c_void_p).value

    class CFrame(ctypes.Structure):
        _fields_: Tuple[Tuple[str, type], ...] = _debug_header + (
            ("ob_refcnt", ctypes.c_ssize_t),
            ("ob_type", ctypes.c_void_p),
            ("f_back", ctypes.c_void_p),
            ("f_frame", ctypes.POINTER(PyInterpreterFrame)),
            ("f_trace", ctypes.c_void_p),
            ("f_lineno", ctypes.c_int),
            ("f_trace_lines", ctypes.c_char),
            ("f_trace_opcodes", ctypes.c_char),
            ("f_fast_as_locals", ctypes.c_char),
            ("_f_frame_data", PyObjPtr),
        )

        def stackread(self, idx: int) -> object:
            frame = self.f_frame.contents
            return frame.localsplus[frame.stacktop + idx]

        def stackwrite(self, idx: int, val: object):
            frame = self.f_frame.contents
            frame.localsplus[frame.stacktop + idx] = val

elif sys.version_info >= (3, 10):

    class CFrame(ctypes.Structure):
        _fields_: Tuple[Tuple[str, type], ...] = _debug_header + (
            ("ob_refcnt", ctypes.c_ssize_t),
            ("ob_type", ctypes.c_void_p),
            ("ob_size", ctypes.c_ssize_t),
            ("f_back", ctypes.c_void_p),
            ("f_code", ctypes.c_void_p),
            ("f_builtins", ctypes.py_object),
            ("f_globals", ctypes.py_object),
            ("f_locals", ctypes.py_object),
            ("f_valuestack", PyObjPtr),
            ("f_trace", ctypes.c_void_p),
            ("f_stackdepth", ctypes.c_int),
        )

        def stackread(self, idx: int) -> object:
            return self.f_valuestack[(self.f_stackdepth) + idx]

        def stackwrite(self, idx: int, val: object):
            self.f_valuestack[(self.f_stackdepth) + idx] = val

else:  # Python < 3.10

    class CFrame(ctypes.Structure):
        _fields_: Tuple[Tuple[str, type], ...] = _debug_header + (
            ("ob_refcnt", ctypes.c_ssize_t),
            ("ob_type", ctypes.c_void_p),
            ("ob_size", ctypes.c_ssize_t),
            ("f_back", ctypes.c_void_p),
            ("f_code", ctypes.c_void_p),
            ("f_builtins", ctypes.py_object),
            ("f_globals", ctypes.py_object),
            ("f_locals", ctypes.py_object),
            ("f_valuestack", PyObjPtr),
            ("f_stacktop", PyObjPtr),
        )

        def stackread(self, idx: int) -> object:
            return self.f_stacktop[idx]

        def stackwrite(self, idx: int, val: object) -> None:
            self.f_stacktop[idx] = val


def frame_stack_read(frame, idx) -> Any:
    c_frame = CFrame.from_address(id(frame))
    val = c_frame.stackread(idx)
    Py_IncRef(ctypes.py_object(val))
    return val


def frame_stack_write(frame, idx, val):
    c_frame = CFrame.from_address(id(frame))
    old_val = c_frame.stackread(idx)
    try:
        Py_IncRef(ctypes.py_object(val))
    except ValueError:  # (PyObject is NULL) - no incref required
        pass
    c_frame.stackwrite(idx, val)
    try:
        Py_DecRef(ctypes.py_object(old_val))
    except ValueError:  # (PyObject is NULL) - no decref required
        pass


CALL_FUNCTION = dis.opmap.get("CALL_FUNCTION", 131)
CALL_FUNCTION_KW = dis.opmap.get("CALL_FUNCTION_KW", 141)
CALL_FUNCTION_EX = dis.opmap["CALL_FUNCTION_EX"]
CALL_METHOD = dis.opmap.get("CALL_METHOD", 161)
BUILD_TUPLE_UNPACK_WITH_CALL = dis.opmap.get("BUILD_TUPLE_UNPACK_WITH_CALL", 158)
CALL = dis.opmap.get("CALL", 171)
NULL_POINTER = object()


def handle_build_tuple_unpack_with_call(
    frame, c_frame
) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 1)
    try:
        return (idx, c_frame.stackread(idx))
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call(frame, c_frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 1)
    try:
        return (idx - 1, c_frame.stackread(idx - 1))
    except ValueError:
        pass
    return (idx, c_frame.stackread(idx))


def handle_call_function(frame, c_frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 1)
    try:
        return (idx, c_frame.stackread(idx))
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call_function_kw(frame, c_frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 2)
    try:
        return (idx, c_frame.stackread(idx))
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call_function_ex(frame, c_frame) -> Optional[Tuple[int, Callable]]:
    idx = -((frame.f_code.co_code[frame.f_lasti + 1] & 1) + 2)
    try:
        return (idx, c_frame.stackread(idx))
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call_method(frame, c_frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 2)
    try:
        return (idx, c_frame.stackread(idx))
    except ValueError:
        # not a sucessful method lookup; no call happens here
        idx += 1
        return (idx, c_frame.stackread(idx))


_CALL_HANDLERS: Dict[
    int, Callable[[object, object], Optional[Tuple[int, Callable]]]
] = {
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
            maybe_call_info = call_handler(frame, CFrame.from_address(id(frame)))
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
        frame.f_trace = self.ctracer
        frame.f_trace_opcodes = True

    def pop_config(self, module) -> None:
        self.ctracer.pop_module(module)

    def set_postop_callback(self, codeobj, callback, frame):
        self.ctracer.push_postop_callback(frame, callback)

    def __enter__(self) -> object:
        self.old_traceobj = sys.gettrace()
        self.ctracer.start()
        return self

    def __exit__(self, _etype, exc, _etb):
        self.ctracer.stop()
        sys.settrace(self.old_traceobj)


# We expect the composite tracer to be used like a singleton.
# (you can only have one tracer active at a time anyway)
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
