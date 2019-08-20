import collections.abc
import typing
from typing import *
import typing_inspect  # type: ignore

def origin_of(typ:Type) -> Type:
    if hasattr(typ, '__origin__'):
        return typ.__origin__
    return typ

def unify(value_type:Type, recv_type:Type, bindings:Optional[typing.ChainMap[object, Type]]=None) -> bool:
    if bindings is None:
        bindings = collections.ChainMap()
    if value_type in (Any, ...) or recv_type in (Any, ...):
        return True
    if isinstance(value_type, list) and isinstance(recv_type, list):
        if len(value_type) == len(recv_type):
            for (varg, targ) in zip(value_type, recv_type):
                if not unify(varg, targ, bindings):
                    return False
            return True
    recv_type = bindings.get(recv_type, recv_type)
    if typing_inspect.is_union_type(recv_type):
        for recv_subtype in typing_inspect.get_args(recv_type):
            writes: Dict[object, Type] = {}
            sub_bindings = bindings.new_child(writes)
            if unify(value_type, recv_subtype, sub_bindings):
                bindings.update(writes)
                return True
        return False
    if typing_inspect.is_typevar(recv_type):
        assert recv_type not in bindings
        bindings[recv_type] = value_type
        return True
    vorigin, rorigin = origin_of(value_type), origin_of(recv_type)
    if issubclass(vorigin, rorigin) or issubclass(rorigin, vorigin):
        def arg_getter(typ):
            origin = origin_of(typ)
            if not getattr(typ, '__args__', True):
                args = []
            else:
                args = list(typing_inspect.get_args(typ, evaluate=True))
            if origin == tuple and len(args) == 2 and args[1] == ...:
                args = [args[0]]
            elif issubclass(origin, collections.abc.Callable):
                if not args:
                    args = [..., Any]
            return args
        vargs = arg_getter(value_type)
        targs = arg_getter(recv_type)
        if len(vargs) == len(targs):
            for (varg, targ) in zip(vargs, targs):
                if not unify(varg, targ, bindings):
                    return False
            return True
    print('Failed to unify ', value_type, vorigin, recv_type, rorigin)
    return False
        
def realize(pytype:Type, bindings:Mapping[object, type]) -> object:
    if typing_inspect.is_typevar(pytype):
        return bindings[pytype]
    if not hasattr(pytype, '__args__'):
        return pytype
    newargs :List = []
    for arg in pytype.__args__:  # type:ignore
        newargs.append(realize(arg, bindings))
    #print('realizing pytype', repr(pytype), 'newargs', repr(newargs))
    pytype_origin = origin_of(pytype)
    if not hasattr(pytype_origin, '_name'):
        pytype_origin = getattr(typing, pytype._name)  # type:ignore
    if pytype_origin is Callable: # Callable args get flattened
        newargs = [newargs[:-1], newargs[-1]]
    return pytype_origin.__getitem__(tuple(newargs))  # type:ignore

