import inspect
import functools
import types
from condition_parser import Conditions, get_fn_conditions
from typing import *

class PreconditionFailed(BaseException):
    pass

class PostconditionFailed(BaseException):
    pass

class EnforcementWrapper:
    def __init__(self, fn:Callable, conditions:Conditions):
        self.fn = fn
        self.signature = inspect.signature(fn)
        self.conditions = conditions
    def __call__(self, *a, **kw):
        bound_args = self.signature.bind(*a, **kw)
        for precondition in self.conditions.pre:
            if not eval(precondition.expr, self.fn.__globals__, bound_args.arguments):
                raise PreconditionFailed('Precondition failed at {}:{}'.format(precondition.filename, precondition.line))
        ret = self.fn(*a, **kw)
        lcls = {**bound_args.arguments, '__return__':ret}
        for postcondition in self.conditions.post:
            if not eval(postcondition.expr, self.fn.__globals__, lcls):
                raise PostconditionFailed('Postcondition failed at {}:{}'.format(postcondition.filename, postcondition.line))
        return ret

class EnforcedConditions:
    def __init__(self, *envs, interceptor=lambda x:x):
        self.envs = envs
        self.interceptor = interceptor
        self.wrapper_map = {}
        self.original_map = {}
    def __enter__(self):
        for env in self.envs:
            for (k, v) in list(env.items()):
                if not isinstance(v, types.FunctionType):
                    continue
                wrapper = self._get_wrapper(v)
                if wrapper is v:
                    continue
                env[k] = wrapper
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for env in self.envs:
            for (k, v) in list(env.items()):
                if not isinstance(v, EnforcementWrapper):
                    continue
                env[k] = self.original_map[v]
        return False
    def _get_wrapper(self, fn:Callable) -> Callable:
        wrapper = self.wrapper_map.get(fn)
        if wrapper is not None:
            return wrapper
        if isinstance(fn, EnforcementWrapper):
            return fn
        conditions = get_fn_conditions(fn)
        if conditions.has_any():
            wrapper = EnforcementWrapper(self.interceptor(fn), conditions)
            functools.update_wrapper(wrapper, fn)
        else:
            wrapper = fn
        self.wrapper_map[fn] = wrapper
        self.original_map[wrapper] = fn
        return wrapper
