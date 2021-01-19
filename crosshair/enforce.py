import builtins
import contextlib
import copy
import inspect
import functools
import sys
import traceback
import types
from typing import *
from crosshair.condition_parser import fn_globals
from crosshair.condition_parser import Conditions
from crosshair.condition_parser import ClassConditions
from crosshair.condition_parser import ConditionParser
from crosshair.fnutil import FunctionInfo
from crosshair.util import IdentityWrapper
from crosshair.util import AttributeHolder
from crosshair.util import debug

class PreconditionFailed(BaseException):
    pass


class PostconditionFailed(BaseException):
    pass


def is_singledispatcher(fn: Callable) -> bool:
    return hasattr(fn, 'registry') and isinstance(fn.registry, Mapping)  # type: ignore


def EnforcementWrapper(fn: Callable, conditions: Conditions, enforced: 'EnforcedConditions') -> Callable:
    signature = conditions.sig

    def wrapper(*a, **kw):
        fns_enforcing = enforced.fns_enforcing
        if fns_enforcing is None or fn in fns_enforcing:
            return fn(*a, **kw)
        #debug('Calling enforcement wrapper ', fn, signature, 'with', a, kw)
        bound_args = signature.bind(*a, **kw)
        bound_args.apply_defaults()
        old = {}
        mutable_args = conditions.mutable_args
        mutable_args_remaining = set(mutable_args) if mutable_args is not None else set()
        for argname, argval in bound_args.arguments.items():
            try:
                old[argname] = copy.copy(argval)
            except Exception as exc:
                pass
            if argname in mutable_args_remaining:
                mutable_args_remaining.remove(argname)
        if mutable_args_remaining:
            raise PostconditionFailed('Unrecognized mutable argument(s) in postcondition: "{}"'.format(
                ','.join(mutable_args_remaining)))
        with enforced.currently_enforcing(fn):
            for precondition in conditions.pre:
                #debug(' precondition eval ', precondition.expr_source)
                # TODO: is fn_globals required here?
                args = {**fn_globals(fn), **bound_args.arguments}
                if not precondition.evaluate(args):
                    raise PreconditionFailed(
                        f'Precondition "{precondition.expr_source}" was not satisfied '
                        f'before calling "{fn.__name__}"')
        ret = fn(*a, **kw)
        with enforced.currently_enforcing(fn):
            lcls = {**bound_args.arguments, '__return__': ret,
                    '_': ret, '__old__': AttributeHolder(old)}
            args = {**fn_globals(fn), **lcls}
            for postcondition in conditions.post:
                #debug(' postcondition eval ', postcondition.expr_source, fn, lcls['_'])
                if postcondition.evaluate and not postcondition.evaluate(args):
                    raise PostconditionFailed('Postcondition failed at {}:{}'.format(
                        postcondition.filename, postcondition.line))
        #debug('Completed enforcement wrapper ', fn)
        return ret
    functools.update_wrapper(wrapper, fn)
    return wrapper


class EnforcedConditions:
    def __init__(self,
                 condition_parser: ConditionParser,
                 *envs: Mapping[str, object],
                 interceptor=lambda x: x):
        self.condition_parser = condition_parser
        self.envs = envs
        self.interceptor = interceptor
        self.fns_enforcing: Optional[Set[Callable]] = set()
        self.wrapper_map: Dict[object, Callable] = {}
        self.original_map: Dict[IdentityWrapper[object], object] = {}

    def _wrap_class(self, cls: type) -> None:
        if not self.condition_parser.get_class_conditions(cls).has_any():
            return
        #debug('wrapping class ', cls)
        for superclass in cls.mro():
            super_conditions = self.condition_parser.get_class_conditions(superclass)
            if super_conditions.has_any():
                self._wrap_class_members(superclass, super_conditions)

    def _wrap_class_members(self, cls: type, class_conditions: ClassConditions) -> None:
        method_conditions = dict(class_conditions.methods)
        for method_name in list(cls.__dict__.keys()):
            conditions = method_conditions.get(method_name)
            if conditions is None:
                continue
            ctxfn = FunctionInfo.from_class(cls, method_name)
            raw_fn = ctxfn.descriptor
            wrapper = self.wrapper_map.get(raw_fn)
            if wrapper is None:
                conditions = conditions or self.condition_parser.get_fn_conditions(ctxfn)
                if conditions and conditions.has_any():
                    fn, _ = ctxfn.callable()
                    wrapper = EnforcementWrapper(
                        self.interceptor(fn), conditions, self)
                    functools.update_wrapper(wrapper, fn)
                else:
                    wrapper = fn
                self.wrapper_map[raw_fn] = wrapper
            outer_wrapper = ctxfn.patch_logic(wrapper)
            self.original_map[IdentityWrapper(outer_wrapper)] = raw_fn
            setattr(cls, method_name, outer_wrapper)

    def _transform_singledispatch(self, fn, transformer):
        overloads = list(fn.registry.items())
        wrapped = functools.singledispatch(transformer(overloads[0][1]))
        for overload_typ, overload_fn in overloads[1:]:
            wrapped.register(overload_typ)(transformer(overload_fn))
        return wrapped

    def is_enforcement_wrapper(self, value):
        return IdentityWrapper(value) in self.original_map

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
    def disabled_enforcement(self):
        prev = self.fns_enforcing
        assert prev is not None
        self.fns_enforcing = None
        try:
            yield None
        finally:
            self.fns_enforcing = prev

    @contextlib.contextmanager
    def enabled_enforcement(self):
        prev = self.fns_enforcing
        assert prev is None
        self.fns_enforcing = set()
        try:
            yield None
        finally:
            self.fns_enforcing = prev

    def __enter__(self):
        next_envs = [env.copy() for env in self.envs]
        for env, next_env in zip(self.envs, next_envs):
            for (k, v) in env.items():
                if isinstance(v, (types.FunctionType, types.BuiltinFunctionType)):
                    if is_singledispatcher(v):
                        wrapper = self._transform_singledispatch(
                            v, self._wrap_fn)
                    else:
                        wrapper = self._wrap_fn(v)
                        if wrapper is v:
                            continue
                    next_env[k] = wrapper
                elif isinstance(v, type):
                    self._wrap_class(v)
        for env, next_env in zip(self.envs, next_envs):
            env.update(next_env)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        next_envs = [env.copy() for env in self.envs]
        for env, next_env in zip(self.envs, next_envs):
            for (k, v) in list(env.items()):
                next_env[k] = self._unwrap(v)
        for env, next_env in zip(self.envs, next_envs):
            env.update(next_env)
        return False

    def _unwrap(self, value):
        if self.is_enforcement_wrapper(value):
            return self.original_map[IdentityWrapper(value)]
        elif is_singledispatcher(value):
            return self._transform_singledispatch(value, self._unwrap)
        elif isinstance(value, type):
            self._unwrap_class(value)
        return value

    def _unwrap_class(self, cls: type):
        for method_name, method in list(cls.__dict__.items()):
            if self.is_enforcement_wrapper(method):
                setattr(cls, method_name,
                        self.original_map[IdentityWrapper(method)])

    def _wrap_fn(self, fn: Callable,
                 conditions: Optional[Conditions] = None) -> Callable:
        wrapper = self.wrapper_map.get(fn)
        if wrapper is not None:
            return wrapper
        if conditions is None:
            conditions = self.condition_parser.get_fn_conditions(
                FunctionInfo.from_fn(fn))  # type: ignore
        if conditions and conditions.has_any():
            wrapper = EnforcementWrapper(
                self.interceptor(fn), conditions, self)
            functools.update_wrapper(wrapper, fn)
        else:
            wrapper = fn
        self.wrapper_map[fn] = wrapper
        self.original_map[IdentityWrapper(wrapper)] = fn
        return wrapper
