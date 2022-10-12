"""Provide access to and overrides for functions as they are called."""

import ctypes
import dis
import sys
from collections import defaultdict
from sys import _getframe
from types import FrameType
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple

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


class TraceException(BaseException):
    # We extend BaseException instead of Exception, because it won't be considered a
    # user-level exception by CrossHair. (this is for internal assertions)
    pass


class TracingModule:
    # override these!:
    opcodes_wanted = frozenset(_CALL_HANDLERS.keys())

    def trace_op(self, frame, codeobj, opcodenum):
        pass

    def trace_call(
        self,
        frame: Any,
        fn: Callable,
        binding_target: object,
    ) -> Optional[Callable]:
        raise NotImplementedError


def set_up_tracing(tracer: Callable, parent_frame: FrameType) -> Callable[[], None]:
    old_frame_trace = parent_frame.f_trace
    old_frame_optrace = parent_frame.f_trace_opcodes
    old_sys_trace = sys.gettrace()
    parent_frame.f_trace = tracer
    parent_frame.f_trace_opcodes = True
    sys.settrace(tracer)

    def undo():
        parent_frame.f_trace = old_frame_trace
        parent_frame.f_trace_opcodes = old_frame_optrace
        sys.settrace(old_sys_trace)

    return undo


TracerConfig = Tuple[Tuple[TracingModule, ...], DefaultDict[int, List[TracingModule]]]


class CompositeTracer:
    config_stack: List[TracerConfig]
    modules: Tuple[TracingModule, ...]
    enabled_modules: DefaultDict[int, List[TracingModule]]

    def __init__(self):
        self.config_stack = []
        self.modules = ()
        self.enabled_modules = defaultdict(list)
        self.postop_callbacks = {}

    def has_any(self) -> bool:
        return bool(self.modules)

    def push_empty_config(self) -> None:
        self.config_stack.append((self.modules, self.enabled_modules))
        self.modules = ()
        self.enabled_modules = defaultdict(list)

    def push_module(self, module: TracingModule) -> None:
        old_modules, old_enabled_modules = self.modules, self.enabled_modules
        if module in old_modules:
            raise TraceException("Module already installed")
        self.config_stack.append((old_modules, old_enabled_modules))
        self.modules = old_modules + (module,)
        new_enabled_modules = defaultdict(list)
        for mod in self.modules:
            for opcode in mod.opcodes_wanted:
                new_enabled_modules[opcode].append(mod)
        self.enabled_modules = new_enabled_modules

    def trace_caller(self):
        if sys.gettrace() is not self:
            raise TraceException("Tracing is not set up")
        # Frame 0 is the trace_caller method itself
        # Frame 1 is the frame requesting its caller be traced
        # Frame 2 is the caller that we're targeting
        frame = _getframe(2)
        frame.f_trace = self
        frame.f_trace_opcodes = True

    def push_config(self, config: TracerConfig) -> None:
        self.config_stack.append((self.modules, self.enabled_modules))
        self.modules, self.enabled_modules = config

    def pop_config(self) -> TracerConfig:
        old_config = (self.modules, self.enabled_modules)
        self.modules, self.enabled_modules = self.config_stack.pop()
        return old_config

    def set_postop_callback(self, codeobj, callback, frame):
        self.postop_callbacks[frame] = (codeobj, callback)

    def __call__(self, frame, event, arg):
        codeobj = frame.f_code
        if event != "opcode":
            if event != "call":
                return None
            if codeobj.co_filename.endswith(("z3types.py", "z3core.py", "z3.py")):
                return None
            if not self.modules:
                return None
            # print("TRACED CALL FROM ", codeobj.co_filename, codeobj.co_firstlineno, codeobj.co_name)
            frame.f_trace_lines = False
            frame.f_trace_opcodes = True
            return self
        postop_callback = self.postop_callbacks.pop(frame, None)
        if postop_callback:
            (expected_codeobj, callback) = postop_callback
            assert codeobj is expected_codeobj
            callback()
        codenum = codeobj.co_code[frame.f_lasti]
        modules = self.enabled_modules[codenum]
        if not modules:
            return None
        call_handler = _CALL_HANDLERS.get(codenum)
        if call_handler:
            maybe_call_info = call_handler(frame, CFrame.from_address(id(frame)))
            if maybe_call_info is None:
                # TODO: this cannot happen?
                return
            (fn_idx, target) = maybe_call_info
            replace_target = False
            binding_target = None
            if hasattr(target, "__self__"):
                if hasattr(target, "__func__"):
                    binding_target = target.__self__
                    target = target.__func__
                    if hasattr(target, "__func__"):
                        raise TraceException("Double method is not expected")
                else:
                    # The implementation is likely in C.
                    # Attempt to get a function via the type:
                    typelevel_target = getattr(
                        type(target.__self__), target.__name__, None
                    )
                    if typelevel_target is not None:
                        binding_target = target.__self__
                        target = typelevel_target
            for mod in modules:
                replacement = mod.trace_call(frame, target, binding_target)
                if replacement is not None:
                    target = replacement
                    replace_target = True
            if replace_target:
                if binding_target is None:
                    overwrite_target = target
                else:
                    # re-bind a function object if it was originally a bound method
                    # on the stack.
                    overwrite_target = target.__get__(binding_target, binding_target.__class__)  # type: ignore
                frame_stack_write(frame, fn_idx, overwrite_target)
        else:
            for module in modules:
                module.trace_op(frame, codeobj, codenum)

    def __enter__(self) -> object:
        self.initial_config_stack_size = len(self.config_stack)
        calling_frame = _getframe(1)
        self.undo = set_up_tracing(self, calling_frame)
        return self

    def __exit__(self, _etype, exc, _etb):
        if len(self.config_stack) != self.initial_config_stack_size:
            raise TraceException("Unexpected configuration stack change while tracing.")
        self.undo()
        assert len(self.postop_callbacks) == 0  # No leaked post-op callbacks


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


def is_tracing():
    return COMPOSITE_TRACER.has_any()


class TracingOnly:
    def __init__(self, module: TracingModule):
        self.module = module

    def __enter__(self):
        if sys.gettrace() is not COMPOSITE_TRACER:
            raise TraceException("Can't reset modules with tracing isn't installed")
        COMPOSITE_TRACER.push_empty_config()
        COMPOSITE_TRACER.push_module(self.module)

    def __exit__(self, *a):
        COMPOSITE_TRACER.pop_config()
        COMPOSITE_TRACER.pop_config()


class NoTracing:
    """
    A context manager that disables tracing.

    While tracing, CrossHair intercepts many builtin and standard library calls.
    Use this context manager to disable those intercepts.
    It's useful, for example, when you want to check the real type of a symbolic
    variable.
    """

    def __enter__(self):
        # Immediately disable tracing so that we don't trace this body either
        _getframe(0).f_trace = None
        if COMPOSITE_TRACER.modules:
            # Equivalent to self.push_empty_config(), but inlined for performance:
            COMPOSITE_TRACER.config_stack.append(
                (COMPOSITE_TRACER.modules, COMPOSITE_TRACER.enabled_modules)
            )
            COMPOSITE_TRACER.modules = ()
            COMPOSITE_TRACER.enabled_modules = defaultdict(list)
            self.had_tracing = True
        else:
            self.had_tracing = False
        calling_frame = _getframe(1)
        self.undo = set_up_tracing(None, calling_frame)

    def __exit__(self, *a):
        if self.had_tracing:
            COMPOSITE_TRACER.pop_config()
        self.undo()


class ResumedTracing:
    """A context manager that re-enables tracing while inside :class:`NoTracing`."""

    _old_config: Optional[TracerConfig] = None

    def __enter__(self):
        if sys.gettrace() is COMPOSITE_TRACER:
            raise TraceException("Can't resume tracing when already installed")
        if self._old_config is not None:
            raise TraceException("Can't resume tracing when modules are present")
        self._old_config = COMPOSITE_TRACER.pop_config()
        if not COMPOSITE_TRACER.has_any():
            raise TraceException("Resumed tracing, but revealed config is empty")
        calling_frame = sys._getframe(1)
        self.undo = set_up_tracing(COMPOSITE_TRACER, calling_frame)

    def __exit__(self, *a):
        if self._old_config is None:
            raise TraceException("Leaving ResumedTracing, but no underlying config")
        COMPOSITE_TRACER.push_config(self._old_config)
        self.undo()
