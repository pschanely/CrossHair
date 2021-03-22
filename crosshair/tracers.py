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
BUILD_TUPLE_UNPACK_WITH_CALL = dis.opmap["BUILD_TUPLE_UNPACK_WITH_CALL"]
CALL_FUNCTION = dis.opmap["CALL_FUNCTION"]
CALL_FUNCTION_KW = dis.opmap["CALL_FUNCTION_KW"]
CALL_FUNCTION_EX = dis.opmap["CALL_FUNCTION_EX"]
CALL_METHOD = dis.opmap["CALL_METHOD"]
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


_EMPTY_OPCODES: Set[int] = set()


class CompositeTracer:
    modules: Tuple[TracingModule, ...] = ()
    enable_flags: Tuple[bool, ...] = ()
    config_stack: List[Tuple[Tuple[TracingModule, ...], Tuple[bool, ...]]] = []
    # regenerated:
    enabled_modules: List[TracingModule]
    enabled_codes: Set[int]

    def __init__(self, modules: Sequence[TracingModule]):
        for module in modules:
            self.add(module)
        self.regen()

    def add(self, module: TracingModule, enabled: bool = True):
        self.modules = (module,) + self.modules
        self.enable_flags = (enabled,) + self.enable_flags
        self.regen()

    def remove(self, module: TracingModule):
        modules = self.modules

        idx = modules.index(module)
        self.modules = modules[:idx] + modules[idx + 1 :]
        self.enable_flags = self.enable_flags[:idx] + self.enable_flags[idx + 1 :]
        self.regen()

    def set_enabled(self, module, enabled):
        for idx, cur_module in enumerate(self.modules):
            if module is cur_module:
                flags = list(self.enable_flags)
                flags[idx] = enabled
                self.enable_flags = tuple(flags)
                self.regen()
                return
        raise Exception

    def has_any(self) -> bool:
        return bool(self.modules)

    def push_empty_config(self) -> None:
        self.config_stack.append((self.modules, self.enable_flags))
        self.modules = ()
        self.enable_flags = ()
        self.enabled_codes = _EMPTY_OPCODES
        self.enabled_modules = []

    def push_last_config(self) -> None:
        last_config = self.config_stack[-1]
        self.config_stack.append((self.modules, self.enable_flags))
        self.modules, self.enable_flags = last_config
        self.regen()

    def pop_config(self) -> None:
        self.modules, self.enable_flags = self.config_stack.pop()
        self.regen()

    def regen(self) -> None:
        enable_flags = self.enable_flags
        ops: Set[int] = set()
        for (idx, module) in enumerate(self.modules):
            if enable_flags[idx]:
                ops |= module.opcodes_wanted
        self.enabled_codes = ops
        self.enabled_modules = [
            mod for (idx, mod) in enumerate(self.modules) if enable_flags[idx]
        ]
        if self.enabled_modules:
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

    def __call__(self, frame, event, arg):
        codeobj = frame.f_code
        scall = "call"  # exists just to avoid SyntaxWarning
        sopcode = "opcode"  # exists just to avoid SyntaxWarning
        if event is scall:  # identity compare for performance
            for mod in self.enabled_modules:
                if mod.cached_wants_codeobj(codeobj):
                    frame.f_trace_lines = False
                    frame.f_trace_opcodes = True
                    return self
            return None
        if event is not sopcode:  # identity compare for performance
            return None
        codenum = codeobj.co_code[frame.f_lasti]
        if codenum not in self.enabled_codes:
            return None
        replace_target = False
        # will hold (self, function) or (None, function)
        target: Optional[Tuple[object, Callable]] = None
        binding_target = None
        for mod in self.enabled_modules:
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
        sys.settrace(self.prev_tracer)
        self.calling_frame.f_trace = self.calling_frame_trace
        self.calling_frame.f_trace_opcodes = self.calling_frame_trace_opcodes
        return False


# We expect the composite tracer to be used like a singleton.
# (you can only have one tracer active at a time anyway)
COMPOSITE_TRACER = CompositeTracer([])


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


class NoTracing:
    def __enter__(self):
        COMPOSITE_TRACER.push_empty_config()

    def __exit__(self, *a):
        COMPOSITE_TRACER.pop_config()


class ResumedTracing:
    def __enter__(self):
        COMPOSITE_TRACER.push_last_config()

    def __exit__(self, *a):
        COMPOSITE_TRACER.pop_config()
