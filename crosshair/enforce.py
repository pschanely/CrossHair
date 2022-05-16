import contextlib
import copy
import functools
import os
from types import FrameType
from typing import Callable, Dict, Mapping, Optional, Set, Tuple

from crosshair.condition_parser import (
    ConditionParser,
    Conditions,
    NoEnforce,
    fn_globals,
    get_current_parser,
)
from crosshair.fnutil import FunctionInfo
from crosshair.statespace import prefer_true
from crosshair.tracers import COMPOSITE_TRACER, NoTracing, ResumedTracing, TracingModule
from crosshair.util import AttributeHolder


class PreconditionFailed(BaseException):
    pass


class PostconditionFailed(BaseException):
    pass


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
    with NoTracing():
        # Python does not invoke __init__ if __new__ returns an object of another type
        # https://docs.python.org/3/reference/datamodel.html#object.__new__
        if isinstance(obj, typ):
            with ResumedTracing():
                WithEnforcement(obj.__init__)(*a, **kw)  # type: ignore
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
            # debug("Calling enforcement wrapper ", fn.__name__, signature)
            bound_args = signature.bind(*a, **kw)
            bound_args.apply_defaults()
            old = {}
            mutable_args = conditions.mutable_args
            mutable_args_remaining = (
                set(mutable_args) if mutable_args is not None else set()
            )
            for argname, argval in bound_args.arguments.items():
                try:
                    # TODO: reduce the type realizations when argval is a SymbolicObject
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
            if fn.__name__ in ("__init__", "__new__"):
                old["self"] = a[0]
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
        # debug("Completed enforcement wrapper ", fn)
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
        self.codeobj_cache: Dict[object, bool] = {}

    def __enter__(self):  # TODO: no longer used as a context manager; remove
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        COMPOSITE_TRACER.push_module(self)

        try:
            yield None
        finally:
            self.fns_enforcing = prev
            COMPOSITE_TRACER.pop_config()

    def wants_codeobj(self, codeobj) -> bool:
        name = codeobj.co_name
        if name == "_crosshair_with_enforcement":
            return True
        fname = codeobj.co_filename
        if fname.endswith(_FILE_SUFFIXES_WITHOUT_ENFORCEMENT):
            return False
        if name == "_crosshair_wrapper":
            return False
        return True

    def cached_wants_codeobj(self, codeobj) -> bool:
        cache = self.codeobj_cache
        cachedval = cache.get(codeobj)
        if cachedval is None:
            cachedval = self.wants_codeobj(codeobj)
            cache[codeobj] = cachedval
        return cachedval

    def trace_call(
        self,
        frame: FrameType,
        fn: Callable,
        binding_target: object,
    ) -> Optional[Callable]:
        caller_code = frame.f_code
        if not self.cached_wants_codeobj(caller_code):
            return None
        target_name = getattr(fn, "__name__", "")
        if target_name.endswith((">", "_crosshair_wrapper")):
            return None
        if isinstance(fn, NoEnforce):
            return fn.fn
        if isinstance(fn, type) and fn not in (super, type):
            return functools.partial(manually_construct, fn)

        parser = self.condition_parser
        conditions = None
        if binding_target is None:
            conditions = parser.get_fn_conditions(FunctionInfo(None, "", fn))  # type: ignore
        else:
            # Method call.
            # We normally expect to look up contracts on `type(binding_target)`, but
            # if it's a `@classmethod`, we'll find it directly on `binding_target`.
            # TODO: test contracts on metaclass methods
            if isinstance(binding_target, type):
                instance_methods = parser.get_class_conditions(binding_target).methods
                conditions = instance_methods.get(target_name)
            if conditions is None or not conditions.has_any():
                type_methods = parser.get_class_conditions(type(binding_target)).methods
                conditions = type_methods.get(target_name)

        if conditions is not None and not conditions.has_any():
            conditions = None
        if conditions is None:
            return None
        # debug("Enforcing conditions on", fn, " type(binding)=", type(binding_target))
        fn = self.interceptor(fn)  # conditions.fn)  # type: ignore
        wrapper = EnforcementWrapper(fn, conditions, self, binding_target)  # type: ignore
        return wrapper
