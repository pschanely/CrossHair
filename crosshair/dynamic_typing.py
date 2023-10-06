import collections.abc
import typing
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type

import typing_inspect  # type: ignore

_EMPTYSET: frozenset = frozenset()


def origin_of(typ: Type) -> Type:
    if hasattr(typ, "__origin__"):
        return typ.__origin__
    return typ


"""
def _lowest_common_bases(classes):
    # Idea from https://stackoverflow.com/questions/25786566/greatest-common-superclass

    # pull first class, and start with it's bases
    gen = iter(classes)
    cls = next(gen, None)
    if cls is None:
        return set()
    common = set(cls.__mro__)

    # find set of ALL ancestor classes,
    # by intersecting MROs of all specified classes
    for cls in gen:
        common.intersection_update(cls.__mro__)

    # remove any bases which have at least one subclass also in the set,
    # as they aren't part of "minimal" set of common ancestors.
    result = common.copy()
    for cls in common:
        if cls in result:
            result.difference_update(cls.__mro__[1:])

    # return set
    return result

def infer_generic_type(value: object) -> Type:
    if isinstance(value, tuple):
        ...
"""


def unify_callable_args(
    value_types: Sequence[Type],
    recv_types: Sequence[Type],
    bindings: typing.ChainMap[object, Type],
) -> bool:
    if value_types == ... or recv_types == ...:
        return True
    if len(value_types) != len(recv_types):
        return False
    for (varg, rarg) in zip(value_types, recv_types):
        # note reversal here: Callable is contravariant in argument types
        if not unify(rarg, varg, bindings):
            return False
    return True


def unify_dicts(
    value_types: Optional[Dict[object, Type]],
    recv_types: Optional[Dict[object, Type]],
    bindings: typing.ChainMap[object, Type],
) -> bool:
    if value_types is None or recv_types is None:
        return False
    writes: Dict[object, Type] = {}
    sub_bindings = bindings.new_child(writes)
    for recv_key, recv_item_type in recv_types.items():
        value_item_type = value_types.pop(recv_key, None)
        if value_item_type is None:
            return False
        if not unify(value_item_type, recv_item_type, sub_bindings):
            return False
    if value_types:
        return False
    bindings.maps.insert(0, writes)
    return True


def unify(
    value_type: Type,
    recv_type: Type,
    bindings: Optional[typing.ChainMap[object, Type]] = None,
) -> bool:
    if bindings is None:
        bindings = collections.ChainMap()
    value_type = bindings.get(value_type, value_type)
    recv_type = bindings.get(recv_type, recv_type)
    if value_type == Any or recv_type == Any:
        return True

    # Unions
    if typing_inspect.is_union_type(value_type):
        for value_subtype in typing_inspect.get_args(value_type):
            writes: Dict[object, Type] = {}
            sub_bindings = bindings.new_child(writes)
            if not unify(value_subtype, recv_type, sub_bindings):
                return False
            # Right now, we just discard the bindings here.
            # In theory, we could save bindings that are unifyable across all iterations.
        return True
    if typing_inspect.is_union_type(recv_type):
        for recv_subtype in typing_inspect.get_args(recv_type):
            writes = {}
            sub_bindings = bindings.new_child(writes)
            if unify(value_type, recv_subtype, sub_bindings):
                bindings.update(writes)
                return True
        return False

    # TypeVars
    if typing_inspect.is_typevar(recv_type):
        assert recv_type not in bindings
        bindings[recv_type] = value_type
        return True
    if typing_inspect.is_typevar(value_type):
        value_type = object  # TODO consider typevar bounds etc?
    vorigin, rorigin = origin_of(value_type), origin_of(recv_type)

    # TypedDicts
    recv_required_keys: frozenset = getattr(recv_type, "__required_keys__", _EMPTYSET)
    value_required_keys: frozenset = getattr(value_type, "__required_keys__", _EMPTYSET)
    if recv_required_keys or value_required_keys:
        if not recv_required_keys and value_required_keys:
            return False
        recv_fields = recv_type.__annotations__
        value_fields = value_type.__annotations__

        def filtered_dict(d: dict, fields: frozenset):
            return {k: v for (k, v) in d.items() if k in fields}

        if not unify_dicts(
            filtered_dict(value_fields, value_required_keys),
            filtered_dict(recv_fields, recv_required_keys),
            bindings,
        ):
            return False
        recv_opt_keys: frozenset = getattr(recv_type, "__optional_keys__", _EMPTYSET)
        value_opt_keys: frozenset = getattr(value_type, "__optional_keys__", _EMPTYSET)
        common_keys = recv_opt_keys & value_opt_keys
        return unify_dicts(
            filtered_dict(value_fields, common_keys),
            filtered_dict(recv_fields, common_keys),
            bindings,
        )

    # Tuples
    if vorigin is tuple:
        args = getattr(value_type, "__args__", (object, ...))
        if (len(args) == 2 and args[-1] == ...) or len(set(args)) <= 1:
            arg_type = args[0] if args else object
            writes = {}
            sub_bindings = bindings.new_child(writes)
            if unify(Sequence[arg_type], recv_type, sub_bindings):  # type:ignore
                bindings.update(writes)
                return True
            if args[-1] == ...:
                value_type = tuple
    if rorigin is tuple:
        args = getattr(recv_type, "__args__", (object, ...))
        if len(args) == 2 and args[-1] == ...:
            arg_type = args[0]
            writes = {}
            sub_bindings = bindings.new_child(writes)
            if unify(value_type, Sequence[arg_type], sub_bindings):  # type:ignore
                bindings.update(writes)
                return True
            value_type = tuple

    if issubclass(vorigin, rorigin):

        def arg_getter(typ):
            origin = origin_of(typ)
            if not getattr(typ, "__args__", True):
                args = []
            else:
                args = list(typing_inspect.get_args(typ, evaluate=True))
            # if origin == tuple and len(args) == 2 and args[1] == ...:
            #    args = [args[0]]
            if issubclass(origin, collections.abc.Callable):
                if not args:
                    args = [..., Any]
            return args

        vargs = arg_getter(value_type)
        rargs = arg_getter(recv_type)
        if issubclass(rorigin, collections.abc.Callable):  # type: ignore
            (vcallargs, vcallreturn) = vargs
            (rcallargs, rcallreturn) = rargs
            if not unify(vcallreturn, rcallreturn, bindings):
                return False
            return unify_callable_args(vcallargs, rcallargs, bindings)
        # if one type has type arguments and the other doesn't, we unify whatever types we can:
        if len(vargs) != len(rargs):
            if len(vargs) == 0:
                vargs = [object for _ in rargs]
            else:
                return False
        for (varg, targ) in zip(vargs, rargs):
            if not unify(varg, targ, bindings):
                return False
        return True
    # print('Failed to unify value type ', value_type, '(origin=', vorigin, ') with recv type ', recv_type, '(origin=', rorigin, ')')
    return False


def get_bindings_from_type_arguments(pytype: Type) -> Mapping[object, type]:
    # NOTE: sadly, this won't work for builtin containers, e.g. `List[int]`
    if hasattr(pytype, "__args__"):
        args = pytype.__args__
        params = typing_inspect.get_parameters(typing_inspect.get_origin(pytype))
        if len(params) == len(args):
            return dict(zip(params, args))
    return {}


def realize(pytype: Type, bindings: Mapping[object, type]) -> object:
    if typing_inspect.is_typevar(pytype):
        return bindings[pytype]
    if not hasattr(pytype, "__args__"):
        return pytype
    newargs: List = []
    for arg in pytype.__args__:  # type:ignore
        newargs.append(realize(arg, bindings))
    # print('realizing pytype', repr(pytype), 'newargs', repr(newargs))
    pytype_origin = origin_of(pytype)
    if not hasattr(pytype_origin, "_name"):
        pytype_origin = getattr(typing, pytype._name)  # type:ignore
    if pytype_origin is Callable:  # Callable args get flattened
        newargs = [newargs[:-1], newargs[-1]]
    return pytype_origin.__getitem__(tuple(newargs))
