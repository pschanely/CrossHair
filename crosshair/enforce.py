import copy
import inspect
import functools
import sys
import traceback
import types
from typing import *

from crosshair.condition_parser import Conditions, get_fn_conditions, ClassConditions, get_class_conditions

class PreconditionFailed(BaseException):
    pass

class PostconditionFailed(BaseException):
    pass

def is_enforcement_wrapper(fn:Callable):
    return getattr(fn, '__is_enforcement_wrapper__', False)

def EnforcementWrapper(fn:Callable, conditions:Conditions):
    signature = inspect.signature(fn)
    def wrapper(*a, **kw):
        bound_args = signature.bind(*a, **kw)
        old = {}
        mutable_args_remaining = set(conditions.mutable_args)
        for argname, argval in bound_args.arguments.items():
            old[argname] = copy.copy(argval)
            if argname in mutable_args_remaining:
                mutable_args_remaining.remove(argname)
        if mutable_args_remaining:
            raise PostconditionFailed('Unrecognized mutable argument(s) in postcondition: "{}"'.format(','.join(mutable_args_remaining)))
        for precondition in conditions.pre:
            if not eval(precondition.expr, fn.__globals__, bound_args.arguments):
                raise PreconditionFailed('Precondition failed at {}:{}'.format(precondition.filename, precondition.line))
        ret = fn(*a, **kw)
        lcls = {**bound_args.arguments, '__return__':ret, '__old__':old}
        for postcondition in conditions.post:
            if not eval(postcondition.expr, fn.__globals__, lcls):
                raise PostconditionFailed('Postcondition failed at {}:{}'.format(postcondition.filename, postcondition.line))
        for argname, argval in bound_args.arguments.items():
            if argname not in conditions.mutable_args:
                if old[argname].__dict__ != argval.__dict__:
                    raise PostconditionFailed('Argument "{}" is not marked as mutable, but has changed'.format(argname))
                
        return ret
    setattr(wrapper, '__is_enforcement_wrapper__', True)
    return wrapper

class EnforcedConditions:
    def __init__(self, *envs, interceptor=lambda x:x):
        self.envs = envs
        self.interceptor = interceptor
        self.wrapper_map = {}
        self.original_map = {}
    def _wrap_class(self, cls:type, class_conditions:ClassConditions):
        method_conditions = dict(class_conditions.methods)
        for method_name, method in list(inspect.getmembers(cls, inspect.isfunction)):
            conditions = method_conditions.get(method)
            if conditions is None:
                continue
            wrapper = self._get_wrapper(method, conditions)
            setattr(cls, method_name, wrapper)
    def _unwrap_class(self, cls:type):
        for method_name, method in list(inspect.getmembers(cls, inspect.isfunction)):
            if is_enforcement_wrapper(method):
                setattr(cls, method_name, self.original_map[method])
    def __enter__(self):
        for env in self.envs:
            for (k, v) in list(env.items()):
                if isinstance(v, types.FunctionType):
                    wrapper = self._get_wrapper(v)
                    if wrapper is v:
                        continue
                    env[k] = wrapper
                elif isinstance(v, type):
                    conditions = get_class_conditions(v)
                    if conditions.has_any():
                        self._wrap_class(v, conditions)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for env in self.envs:
            for (k, v) in list(env.items()):
                if is_enforcement_wrapper(v):
                    env[k] = self.original_map[v]
                elif isinstance(v, type):
                    self._unwrap_class(v)
        return False
    def _get_wrapper(self, fn:Callable, conditions:Optional[Conditions]=None) -> Callable:
        wrapper = self.wrapper_map.get(fn)
        if wrapper is not None:
            return wrapper
        if is_enforcement_wrapper(fn):
            return fn
        conditions = conditions or get_fn_conditions(fn)
        if conditions.has_any():
            wrapper = EnforcementWrapper(self.interceptor(fn), conditions)
            functools.update_wrapper(wrapper, fn)
        else:
            wrapper = fn
        self.wrapper_map[fn] = wrapper
        self.original_map[wrapper] = fn
        return wrapper
