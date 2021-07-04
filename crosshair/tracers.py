"""Provide access to and overrides for functions as they are called."""

import contextlib
import ctypes
import dis
from functools import wraps
import inspect
import itertools
import sys
from collections import defaultdict
from collections.abc import Mapping
from typing import *


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


class CFrame(ctypes.Structure):
    _fields_: Tuple[Tuple[str, type], ...] = _debug_header + (
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("ob_size", ctypes.c_ssize_t),
        ("f_back", ctypes.c_void_p),
        ("f_code", ctypes.c_void_p),
        ("f_builtins", PyObjPtr),
        ("f_globals", PyObjPtr),
        ("f_locals", PyObjPtr),
        ("f_valuestack", PyObjPtr),
        ("f_stacktop", PyObjPtr),
    )


def cframe_stack_write(c_frame, idx, val):
    stacktop = c_frame.f_stacktop
    old_val = stacktop[idx]
    try:
        Py_IncRef(ctypes.py_object(val))
    except ValueError:  # (PyObject is NULL) - no incref required
        pass
    stacktop[idx] = val
    try:
        Py_DecRef(ctypes.py_object(old_val))
    except ValueError:  # (PyObject is NULL) - no decref required
        pass


CALL_FUNCTION = dis.opmap["CALL_FUNCTION"]
CALL_FUNCTION = dis.opmap["CALL_FUNCTION"]
CALL_FUNCTION_KW = dis.opmap["CALL_FUNCTION_KW"]
CALL_FUNCTION_EX = dis.opmap["CALL_FUNCTION_EX"]
CALL_METHOD = dis.opmap["CALL_METHOD"]
# BUILD_TUPLE_UNPACK_WITH_CALL does not exist in all python versions:
BUILD_TUPLE_UNPACK_WITH_CALL = dis.opmap.get("BUILD_TUPLE_UNPACK_WITH_CALL", 158)
NULL_POINTER = object()


def handle_build_tuple_unpack_with_call(
    frame, c_frame
) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 1)
    try:
        return (idx, c_frame.f_stacktop[idx])
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call_function(frame, c_frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 1)
    try:
        return (idx, c_frame.f_stacktop[idx])
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call_function_kw(frame, c_frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 2)
    try:
        return (idx, c_frame.f_stacktop[idx])
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call_function_ex(frame, c_frame) -> Optional[Tuple[int, Callable]]:
    idx = -((frame.f_code.co_code[frame.f_lasti + 1] & 1) + 2)
    try:
        return (idx, c_frame.f_stacktop[idx])
    except ValueError:
        return (idx, NULL_POINTER)  # type: ignore


def handle_call_method(frame, c_frame) -> Optional[Tuple[int, Callable]]:
    idx = -(frame.f_code.co_code[frame.f_lasti + 1] + 2)
    try:
        return (idx, c_frame.f_stacktop[idx])
    except ValueError:
        # not a sucessful method lookup; no call happens here
        idx += 1
        return (idx, c_frame.f_stacktop[idx])


_CALL_HANDLERS: Dict[
    int, Callable[[object, object], Optional[Tuple[int, Callable]]]
] = {
    BUILD_TUPLE_UNPACK_WITH_CALL: handle_build_tuple_unpack_with_call,
    CALL_FUNCTION: handle_call_function,
    CALL_FUNCTION_KW: handle_call_function_kw,
    CALL_FUNCTION_EX: handle_call_function_ex,
    CALL_METHOD: handle_call_method,
}


class TracingModule:
    def __init__(self):
        self.codeobj_cache: Dict[object, bool] = {}

    def cached_wants_codeobj(self, codeobj) -> bool:
        cache = self.codeobj_cache
        cachedval = cache.get(codeobj)
        if cachedval is None:
            cachedval = self.wants_codeobj(codeobj)
            cache[codeobj] = cachedval
        return cachedval

    # override these!:
    opcodes_wanted = frozenset(_CALL_HANDLERS.keys())

    def wants_codeobj(self, codeobj) -> bool:
        return True

    def trace_op(self, frame, codeobj, opcodenum):
        pass

    def trace_call(
        self,
        frame: Any,
        fn: Callable,
        binding_target: object,
    ) -> Optional[Callable]:
        raise NotImplementedError


TracerConfig = Tuple[Tuple[TracingModule, ...], DefaultDict[int, List[TracingModule]]]


class CompositeTracer:
    config_stack: List[TracerConfig]
    modules: Tuple[TracingModule, ...]
    enabled_modules: DefaultDict[int, List[TracingModule]]

    def __init__(self):
        self.config_stack = []
        self.modules = ()
        self.enabled_modules = defaultdict(list)

    def has_any(self) -> bool:
        return bool(self.modules)

    def push_empty_config(self) -> None:
        self.config_stack.append((self.modules, self.enabled_modules))
        self.modules = ()
        self.enabled_modules = defaultdict(list)

    def push_module(self, module: TracingModule) -> None:
        old_modules, old_enabled_modules = self.modules, self.enabled_modules
        assert module not in old_modules
        self.config_stack.append((old_modules, old_enabled_modules))
        self.modules = old_modules + (module,)
        new_enabled_modules = defaultdict(list)
        for mod in self.modules:
            for opcode in mod.opcodes_wanted:
                new_enabled_modules[opcode].append(mod)
        self.enabled_modules = new_enabled_modules
        height = 1
        while True:
            try:
                frame = sys._getframe(height)
            except ValueError:
                break
            if frame.f_trace == None:
                frame.f_trace = self
                frame.f_trace_opcodes = True
            else:
                break
            height += 1

    def push_config(self, config: TracerConfig) -> None:
        self.config_stack.append((self.modules, self.enabled_modules))
        self.modules, self.enabled_modules = config

    def pop_config(self) -> TracerConfig:
        old_config = (self.modules, self.enabled_modules)
        self.modules, self.enabled_modules = self.config_stack.pop()
        return old_config

    def __call__(self, frame, event, arg):
        codeobj = frame.f_code
        scall = "call"  # exists just to avoid SyntaxWarning
        sopcode = "opcode"  # exists just to avoid SyntaxWarning
        if event is scall:  # identity compare for performance
            for mod in self.modules:
                if mod.cached_wants_codeobj(codeobj):
                    frame.f_trace_lines = False
                    frame.f_trace_opcodes = True
                    return self
            return None
        if event is not sopcode:  # identity compare for performance
            return None
        codenum = codeobj.co_code[frame.f_lasti]
        modules = self.enabled_modules[codenum]
        if not modules:
            return None
        replace_target = False
        # will hold (self, function) or (None, function)
        target: Optional[Tuple[object, Callable]] = None
        binding_target = None
        for mod in modules:
            if not mod.cached_wants_codeobj(codeobj):
                continue
            if target is None:
                call_handler = _CALL_HANDLERS.get(codenum)
                if not call_handler:
                    return
                maybe_call_info = call_handler(frame, CFrame.from_address(id(frame)))
                if maybe_call_info is None:
                    return
                (fn_idx, target) = maybe_call_info
                if hasattr(target, "__self__"):
                    if hasattr(target, "__func__"):
                        binding_target = target.__self__
                        target = target.__func__
                        assert not hasattr(target, "__func__")
                    else:
                        # The implementation is likely in C.
                        # Attempt to get a function via the type:
                        typelevel_target = getattr(
                            type(target.__self__), target.__name__, None
                        )
                        if typelevel_target is not None:
                            binding_target = target.__self__
                            target = typelevel_target
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
            cframe_stack_write(CFrame.from_address(id(frame)), fn_idx, overwrite_target)

    def __enter__(self) -> object:
        self.initial_config_stack_size = len(self.config_stack)
        calling_frame = sys._getframe(1)
        self.prev_tracer = sys.gettrace()
        self.calling_frame_trace = calling_frame.f_trace
        self.calling_frame_trace_opcodes = calling_frame.f_trace_opcodes
        assert self.prev_tracer is not self
        sys.settrace(self)
        calling_frame.f_trace = self
        calling_frame.f_trace_opcodes = True
        self.calling_frame = calling_frame
        return self

    def __exit__(self, *a):
        assert len(self.config_stack) == self.initial_config_stack_size
        sys.settrace(self.prev_tracer)
        self.calling_frame.f_trace = self.calling_frame_trace
        self.calling_frame.f_trace_opcodes = self.calling_frame_trace_opcodes
        return False


# We expect the composite tracer to be used like a singleton.
# (you can only have one tracer active at a time anyway)
COMPOSITE_TRACER = CompositeTracer()


# TODO merge this with core.py's "Patched" class.
class PatchingModule(TracingModule):
    """Hot-swap functions on the interpreter stack."""

    def __init__(self, overrides: Optional[Dict[Callable, Callable]] = None):
        self.overrides: Dict[Callable, Callable] = {}
        self.nextfn: Dict[object, Callable] = {}  # code object to next, lower layer
        if overrides:
            self.add(overrides)

    def add(self, new_overrides: Dict[Callable, Callable]):
        for orig, new_override in new_overrides.items():
            prev_override = self.overrides.get(orig, orig)
            self.nextfn[new_override.__code__] = prev_override
            self.overrides[orig] = new_override

    def cached_wants_codeobj(self, codeobj) -> bool:
        return True

    def trace_call(
        self,
        frame: Any,
        fn: Callable,
        binding_target: object,
    ) -> Optional[Callable]:
        target = self.overrides.get(fn)
        # print('call detected', fn, target, frame.f_code.co_name)
        if target is None:
            return None
        # print("Patching call to", fn)
        nextfn = self.nextfn.get(frame.f_code)
        if nextfn is not None:
            return nextfn
        return target


def is_tracing():
    return COMPOSITE_TRACER.has_any()


class NoTracing:
    """
    A context manager that disables tracing.

    While tracing, CrossHair intercepts many builtin and standard library calls.
    Use this context manager to disable those intercepts.
    It's useful, for example, when you want to check the real type of a symbolic
    variable.
    """

    def __enter__(self):
        had_tracing = COMPOSITE_TRACER.has_any()
        # print("enter NoTracing (had_tracing=", had_tracing, ")")
        if had_tracing:
            COMPOSITE_TRACER.push_empty_config()
        self.had_tracing = had_tracing

    def __exit__(self, *a):
        if self.had_tracing:
            COMPOSITE_TRACER.pop_config()
        # print("exit NoTracing (had_tracing=", self.had_tracing, "), now", COMPOSITE_TRACER.has_any())


class ResumedTracing:
    """A context manager that re-enables tracing while inside :class:`NoTracing`."""

    _old_config: Optional[TracerConfig] = None

    def __enter__(self):
        assert self._old_config is None
        self._old_config = COMPOSITE_TRACER.pop_config()
        assert COMPOSITE_TRACER.has_any()

    def __exit__(self, *a):
        assert self._old_config is not None
        COMPOSITE_TRACER.push_config(self._old_config)
