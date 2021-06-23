import builtins
import contextlib
import copy
import inspect
import functools
import os
import sys
import time
import traceback
from types import CodeType
from types import FrameType
from typing import *
from crosshair.condition_parser import fn_globals
from crosshair.condition_parser import get_current_parser
from crosshair.condition_parser import Conditions
from crosshair.condition_parser import ClassConditions
from crosshair.condition_parser import ConditionParser
from crosshair.condition_parser import NoEnforce
from crosshair.fnutil import FunctionInfo
from crosshair.statespace import prefer_true
from crosshair.tracers import CFrame
from crosshair.tracers import cframe_stack_write
from crosshair.tracers import TracingModule
from crosshair.tracers import COMPOSITE_TRACER
from crosshair.util import CrosshairInternal
from crosshair.util import IdentityWrapper
from crosshair.util import AttributeHolder
from crosshair.util import debug


class PreconditionFailed(BaseException):
    pass


class PostconditionFailed(BaseException):
    pass


def is_singledispatcher(fn: Callable) -> bool:
    return hasattr(fn, "registry") and isinstance(fn.registry, Mapping)  # type: ignore


def WithEnforcement(fn: Callable) -> Callable:
    """
    Ensure conditions are enforced on the given callable.

    Enforcement is normally disabled when calling from some internal files, for
    performance reasons. Use WithEnforcement to ensure it is enabled anywhere.
    """
    # This local function has a special name that we look for while tracing
    # (see the wants_codeobj method below):
    def _crosshair_with_enforcement(*a, **kw):
        return fn(*a, **kw)

    return _crosshair_with_enforcement


def manually_construct(typ: type, *a, **kw):
    obj = WithEnforcement(typ.__new__)(typ, *a, **kw)  # object.__new__(typ)
    WithEnforcement(obj.__init__)(*a, **kw)
    return obj


_MISSING = object()


def EnforcementWrapper(
    fn: Callable,
    conditions: Conditions,
    enforced: "EnforcedConditions",
    first_arg: object,
) -> Callable:
    signature = conditions.sig

    def _crosshair_wrapper(*a, **kw):
        fns_enforcing = enforced.fns_enforcing
        if fns_enforcing is None or fn in fns_enforcing:
            return fn(*a, **kw)
        with enforced.currently_enforcing(fn):
            # debug('Calling enforcement wrapper ', fn, signature, 'with', a, kw)
            bound_args = signature.bind(*a, **kw)
            bound_args.apply_defaults()
            old = {}
            mutable_args = conditions.mutable_args
            mutable_args_remaining = (
                set(mutable_args) if mutable_args is not None else set()
            )
            for argname, argval in bound_args.arguments.items():
                try:
                    old[argname] = copy.copy(argval)
                except TypeError as exc:  # for uncopyables
                    pass
                if argname in mutable_args_remaining:
                    mutable_args_remaining.remove(argname)
            if mutable_args_remaining:
                raise PostconditionFailed(
                    'Unrecognized mutable argument(s) in postcondition: "{}"'.format(
                        ",".join(mutable_args_remaining)
                    )
                )
            for precondition in conditions.pre:
                # debug(' precondition eval ', precondition.expr_source)
                # TODO: is fn_globals required here?
                if not precondition.evaluate(bound_args.arguments):
                    raise PreconditionFailed(
                        f'Precondition "{precondition.expr_source}" was not satisfied '
                        f'before calling "{fn.__name__}"'
                    )
        ret = fn(*a, **kw)
        with enforced.currently_enforcing(fn):
            lcls = {
                **bound_args.arguments,
                "__return__": ret,
                "_": ret,
                "__old__": AttributeHolder(old),
            }
            args = {**fn_globals(fn), **lcls}
            for postcondition in conditions.post:
                # debug('Checking postcondition ', postcondition.expr_source, ' on ', fn)
                if postcondition.evaluate and not prefer_true(
                    postcondition.evaluate(args)
                ):
                    raise PostconditionFailed(
                        "Postcondition failed at {}:{}".format(
                            postcondition.filename, postcondition.line
                        )
                    )
        # debug('Completed enforcement wrapper ', fn)
        return ret

    functools.update_wrapper(_crosshair_wrapper, fn)
    return _crosshair_wrapper


_MISSING = object()


_FILE_SUFFIXES_WITHOUT_ENFORCEMENT: Tuple[str, ...] = (
    "/ast.py",
    "/crosshair/libimpl/builtinslib.py",
    "/crosshair/core.py",
    "/crosshair/condition_parser.py",
    "/crosshair/enforce.py",
    "/crosshair/util.py",
    "/crosshair/fnutil.py",
    "/crosshair/statespace.py",
    "/crosshair/tracers.py",
    "/z3.py",
    "/z3core.py",
    "/z3printer.py",
    "/z3types.py",
    "/copy.py",
    "/inspect.py",
    "/re.py",
    "/copyreg.py",
    "/sre_parse.py",
    "/sre_compile.py",
    "/traceback.py",
    "/contextlib.py",
    "/linecache.py",
    "/collections/__init__.py",
    "/enum.py",
    "/typing.py",
)

if os.name == "nt":
    # Hacky platform-independence for performance reasons.
    # (not sure whether there are landmines here?)
    _FILE_SUFFIXES_WITHOUT_ENFORCEMENT = tuple(
        p.replace("/", "\\") for p in _FILE_SUFFIXES_WITHOUT_ENFORCEMENT
    )


class EnforcedConditions(TracingModule):
    def __init__(
        self,
        condition_parser: Optional[ConditionParser] = None,
        interceptor=lambda x: x,
    ):
        super().__init__()
        self.condition_parser = (
            get_current_parser() if condition_parser is None else condition_parser
        )
        self.interceptor = interceptor
        self.fns_enforcing: Optional[Set[Callable]] = None

    def __enter__(self):
        COMPOSITE_TRACER.add(self, enabled=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        COMPOSITE_TRACER.remove(self)
        return False

    @contextlib.contextmanager
    def currently_enforcing(self, fn: Callable):
        if self.fns_enforcing is None:
            yield None
        else:
            self.fns_enforcing.add(fn)
            try:
                yield None
            finally:
                self.fns_enforcing.remove(fn)

    @contextlib.contextmanager
    def enabled_enforcement(self):
        prev = self.fns_enforcing
        assert prev is None
        self.fns_enforcing = set()
        if not COMPOSITE_TRACER.set_enabled(self, True):
            raise CrosshairInternal("Cannot enable enforcement")

        try:
            yield None
        finally:
            self.fns_enforcing = prev
            if not COMPOSITE_TRACER.set_enabled(self, prev):
                raise CrosshairInternal("Tracing handler stack is inconsistent")

    def wants_codeobj(self, codeobj) -> bool:
        if codeobj.co_name == "_crosshair_with_enforcement":
            return True
        fname = codeobj.co_filename
        return not fname.endswith(_FILE_SUFFIXES_WITHOUT_ENFORCEMENT)

    def trace_call(
        self,
        frame: FrameType,
        fn: Callable,
        binding_target: object,
    ) -> Optional[Callable]:
        caller_code = frame.f_code
        if caller_code.co_name == "_crosshair_wrapper":
            return None
        target_name = getattr(fn, "__name__", "")
        if target_name.endswith((">", "_crosshair_wrapper")):
            return None
        if isinstance(fn, NoEnforce):
            return fn.fn
        if type(fn) is type and fn not in (super, type):
            return functools.partial(manually_construct, fn)
        condition_parser = self.condition_parser
        # TODO: Is doing this a problem? A method's function's conditions depend on the
        # class of self.
        ctxfn = FunctionInfo(None, "", fn)  # type: ignore
        conditions = condition_parser.get_fn_conditions(ctxfn)
        if conditions is not None and not conditions.has_any():
            conditions = None
        if conditions is None:
            return None
        # debug("Enforcing conditions on", fn, 'binding', binding_target)
        fn = self.interceptor(conditions.fn)  # type: ignore
        is_bound = binding_target is not None
        wrapper = EnforcementWrapper(fn, conditions, self, binding_target)  # type: ignore
        return wrapper
