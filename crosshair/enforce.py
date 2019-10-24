import builtins
import contextlib
import copy
import inspect
import functools
import sys
import traceback
import types
from typing import *

from crosshair.condition_parser import Conditions, get_fn_conditions, ClassConditions, get_class_conditions, fn_globals
from crosshair.util import IdentityWrapper, AttributeHolder


class PreconditionFailed(BaseException):
    pass


class PostconditionFailed(BaseException):
    pass


def is_singledispatcher(fn: Callable) -> bool:
    # type: ignore
    return hasattr(fn, 'registry') and isinstance(fn.registry, Mapping)


def EnforcementWrapper(fn: Callable, conditions: Conditions, enforced: 'EnforcedConditions') -> Callable:
    signature = conditions.sig

    def wrapper(*a, **kw):
        fns_enforcing = enforced.fns_enforcing
        if fns_enforcing is None or fn in fns_enforcing:
            return fn(*a, **kw)
        #print('Calling enforcement wrapper ', fn)
        bound_args = signature.bind(*a, **kw)
        bound_args.apply_defaults()
        old = {}
        mutable_args_remaining = set(conditions.mutable_args)
        for argname, argval in bound_args.arguments.items():
            old[argname] = copy.copy(argval)
            if argname in mutable_args_remaining:
                mutable_args_remaining.remove(argname)
        if mutable_args_remaining:
            raise PostconditionFailed('Unrecognized mutable argument(s) in postcondition: "{}"'.format(
                ','.join(mutable_args_remaining)))
        with enforced.currently_enforcing(fn):
            for precondition in conditions.pre:
                #print(' precondition eval ', precondition.expr_source)
                args = {**fn_globals(fn), **bound_args.arguments}
                if not eval(precondition.expr, args):
                    raise PreconditionFailed(
                        f'Precondition "{precondition.expr_source}" was not satisfied '
                        f'before calling "{fn.__name__}"')
        ret = fn(*a, **kw)
        with enforced.currently_enforcing(fn):
            lcls = {**bound_args.arguments, '__return__': ret,
                    '_': ret, '__old__': AttributeHolder(old)}
            args = {**fn_globals(fn), **lcls}
            for postcondition in conditions.post:
                # print(' postcondition eval ', postcondition.expr_source, fn)#, args.keys())
                if postcondition.expr and not eval(postcondition.expr, args):
                    raise PostconditionFailed('Postcondition failed at {}:{}'.format(
                        postcondition.filename, postcondition.line))
        #print('Completed enforcement wrapper ', fn)
        return ret
    return wrapper


class EnforcedConditions:
    def __init__(self, *envs, interceptor=lambda x: x):
        self.envs = envs
        self.interceptor = interceptor
        self.fns_enforcing: Optional[Set[Callable]] = set()
        self.wrapper_map: Dict[Callable, Callable] = {}
        self.original_map: Dict[IdentityWrapper[Callable], Callable] = {}

    def _wrap_class(self, cls: type, class_conditions: ClassConditions) -> None:
        #print('wrapping class ', cls)
        method_conditions = dict(class_conditions.methods)
        for method_name, method in list(inspect.getmembers(cls, inspect.isfunction)):
            conditions = method_conditions.get(method_name)
            if conditions is None:
                continue
            wrapper = self._wrap_fn(method, conditions)
            setattr(cls, method_name, wrapper)

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
        self.fns_enforcing = None
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
                    conditions = get_class_conditions(v)
                    if conditions.has_any():
                        self._wrap_class(v, conditions)
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
        for method_name, method in list(inspect.getmembers(cls, inspect.isfunction)):
            if self.is_enforcement_wrapper(method):
                setattr(cls, method_name,
                        self.original_map[IdentityWrapper(method)])

    def _wrap_fn(self, fn: Callable, conditions: Optional[Conditions] = None) -> Callable:
        wrapper = self.wrapper_map.get(fn)
        if wrapper is not None:
            return wrapper
        if self.is_enforcement_wrapper(fn):
            return fn

        conditions = conditions or get_fn_conditions(fn)
        if conditions and conditions.has_any():
            wrapper = EnforcementWrapper(
                self.interceptor(fn), conditions, self)
            functools.update_wrapper(wrapper, fn)
        else:
            wrapper = fn
        self.wrapper_map[fn] = wrapper
        self.original_map[IdentityWrapper(wrapper)] = fn
        return wrapper
