import enum
import codecs
import collections
import inspect
import keyword
import struct
import sys
from typing import *
import typing

import z3  # type: ignore

from crosshair.util import debug, memo
from crosshair.typed_inspect import signature

z3_type_literals = {}
unpack_type_literals: collections.OrderedDict = collections.OrderedDict()
simplify_literals = {}


def type_matches(typ: Type, spec: Any) -> bool:
    if type(spec) is Union:
        for subspec in spec.__args__:
            if type_matches(typ, subspec):
                return True
        return False
    # if getattr(spec, '__origin__', None) is Optional:
    #    return type_matches(typ, spec.__args__[0])
    if spec is Any:
        return True
    if spec is NamedTuple:
        return issubclass(typ, tuple) and hasattr(typ, '_fields')
    return issubclass(typ, spec)


def register_literal_type(types, z3fn=None, unpack=None, simplify=None):
    if not isinstance(types, tuple):
        types = [types]
    for typ in types:
        if z3fn:
            z3_type_literals[typ] = z3fn
        if unpack:
            unpack_type_literals[typ] = unpack
        if simplify:
            simplify_literals[typ] = simplify


@memo
def z3_converter_for_type(typ: Type) -> Callable:
    converter = z3_type_literals.get(typ)

    if converter is not None:
        return converter
    # Try for a non-exact type match:
    root_type = getattr(typ, '__origin__', typ)
    if root_type is Union:
        raise Exception('not implemented')
    # return (lambda t, r, e: unpack_type(type_param(t, r(1)[0] %
    #                                               len(t.__args__)), r, e))
    for curtype, curconverter in z3_type_literals.items():
        if getattr(curtype, '_is_protocol', False):
            continue
        if curtype is Any:
            continue
        matches = type_matches(root_type, curtype)
        if matches:
            debug('  matches: ', typ, curtype, matches)
            return curconverter
    raise ExpressionNotSmtable(Exception('no converter for ' + str(typ)))


def make_z3_var(typ, name):
    converter = z3_converter_for_type(typ)
    return converter(typ, name)


def type_param(typ, index):
    if not hasattr(typ, '__args__'):
        return Any
    type_args = typ.__args__
    if type_args is None:
        return Any
    try:
        return type_args[index]
    except IndexError:
        return Any


class FuzzFunc:
    def __init__(self, bindings, default):
        self.bindings, self.default = bindings, default

    def __call__(self, *a):
        if a in self.bindings:
            return self.bindings[a]
        else:
            return self.default

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if not self.bindings:
            return '<func that returns {}>'.format(repr(self.default))
        return '<func that maps {} with a default of {}>'.format(
            repr(self.bindings), repr(self.default))


def choose(options, reader):
    num = reader(1)[0]
    return options[num % len(options)]


BASIC_TYPES = [type(None), bool, int, float, tuple]


def unpack_basic_value(t, r, e):
    t = BASIC_TYPES[r(1)[0] % len(BASIC_TYPES)]
    return unpack_type_literals[t](t, r, e)


def unpack_generator(t, r, e):
    return (unpack_type(t, r, e) for _ in range(r(1)[0]))


_ENUM_VAL_CACHE: Dict[Type, Tuple[enum.Enum]] = {}


def unpack_enum(t, r, e):
    values = _ENUM_VAL_CACHE.get(t)
    if values is None:
        values = tuple(t)
        _ENUM_VAL_CACHE[t] = values
    if not values:
        # (this is the abstract enum, "enum.Enum")
        # It isn't really unpackable: we could pick an arbitrary enum instead
        raise InputNotUnpackableError(Exception('abstract enum'))
    return values[r(1)[0] % len(values)]


def unpack_oneof(types, reader, env):
    unpack_type(choose(types, reader), reader, env)


def gen_identifier(r) -> str:
    sz = 1 + r(1)[0] % 16
    candidate = bytes(b ^ ord('a') for b in r(sz)).decode('utf8')
    if keyword.iskeyword(candidate) or not candidate.isidentifier():
        raise InputNotUnpackableError(Exception('invalid identifier'))
    return candidate


def unpack_tuple(t, r, e):
    args = getattr(t, '__args__', None)
    if not args:
        return tuple(unpack_generator(Any, r, e))
    elif len(args) == 2 and args[-1] == ...:
        ret = tuple(unpack_generator(args[0], r, e))
        debug('tup ret', ret)
        return ret
    else:
        return tuple(unpack_type(args[i], r, e) for i in range(len(args)))


def unpack_namedtuple(t, r, e):
    if hasattr(t, '__annotations__'):
        return t(*[unpack_type(subtype, r, e) for subtype in t.__annotations__.values()])
    else:
        num_items = r(1)[0] % 7
        type_name = gen_identifier(r)
        fields = [(gen_identifier(r), unpack_type(Type, r, e))
                  for _ in range(num_items)]
        try:
            nt = NamedTuple(type_name, fields)
        except ValueError as e:  # for "duplicate field name"
            raise InputNotUnpackableError(e)
        return unpack_namedtuple(nt, r, e)


class SymbolicInt:
    def __init__(self, z3var):
        self.z3var = z3var

    def __int__(self):
        return self.z3var

    def __index__(self):
        return self.z3var


class SymbolicSeq:
    def __init__(self, z3var):
        self.z3var = z3var

    def __getitem__(self, arg):
        debug('__getitem__ called ', self.z3var,
              arg, self.z3var.__getitem__(arg))
        return self.z3var.__getitem__(arg)

    def __len__(self):
        debug('__len__ called ', self.z3var)
        return SymbolicInt(z3.Length(self.z3var))

    def __add__(self, other):
        if isinstance(other, SymbolicSeq):
            return SymbolicSeq(self.z3var + other.z3var)
        else:
            return SymbolicSeq(self.z3var + other)

    def __eq__(self, other):
        if isinstance(other, SymbolicSeq):
            return z3.Eq(self.z3var, other.z3var)
        else:
            return NotImplemented


_TYPE_TO_SMT_SORT = {
    int: z3.IntSort(),
    float: z3.Float64(),
    bool: z3.BoolSort(),
    str: z3.StringSort(),
}


def type_to_smt_sort(t: Type):
    if t in _TYPE_TO_SMT_SORT:
        return _TYPE_TO_SMT_SORT[t]
    origin = getattr(t, '__origin__', None)
    if origin in (List, Sequence, Container):
        item_type = t.__args__[0]
        return z3.SeqSort(type_to_smt_sort(item_type))
    elif origin in (Dict, Mapping):
        key_type, val_type = t.__args__
        return z3.ArraySort(type_to_smt_sort(key_type),
                            type_to_smt_sort(val_type))
    return None


def smt_var(typ: Type, name: str):
    z3type = type_to_smt_sort(typ)
    if z3type is None:
        if getattr(typ, '__origin__', None) is Tuple:
            if len(typ.__args__) == 2 and typ.__args__[1] == ...:
                z3type = z3.SeqSort(type_to_smt_sort(typ.__args__[0]))
            else:
                return tuple(smt_var(t, name + str(idx)) for (idx, t) in enumerate(typ.__args__))
    var = z3.Const(name, z3type)
    if isinstance(z3type, z3.SeqSortRef):
        var = SymbolicSeq(var)
    return var


def not_unpackable_because(reason: str):
    def unpack(t, r, e):
        raise InputNotUnpackableError(Exception(reason))
    return unpack


register_literal_type(
    type(None),
    unpack=(lambda t, r, e: None))

register_literal_type(
    bool,
    z3fn=smt_var,
    unpack=(lambda t, r, e: (True if r(1)[0] else False)))

register_literal_type(
    enum.Enum,
    unpack=unpack_enum)

register_literal_type(
    (int, typing.SupportsInt, typing.SupportsRound, typing.SupportsAbs),
    z3fn=smt_var,
    unpack=(lambda t, r, e: struct.unpack('q', r(8))[0]),
    simplify=(lambda v: (int(float(v) / 2),)),
)

register_literal_type(
    (float, typing.SupportsFloat),
    z3fn=(lambda t, n: smt_var(float, n)),
    unpack=(lambda t, r, e: struct.unpack('d', r(8))[0]),
    #simplify=(lambda v: set(round(v,i) for i in range(-30,30,2))),
)

register_literal_type(
    (complex, typing.SupportsComplex),
    unpack=(lambda t, r, e: struct.unpack('d', r(8))[0] + struct.unpack('d', r(8))[0] * 1j))

register_literal_type(
    str,
    z3fn=smt_var,
    unpack=(lambda t, r, e: codecs.decode(r(r(1)[0] % 16), 'utf8')),
    simplify=(lambda v: (v[1:], v[:-1])),
)

register_literal_type(
    (bytes, bytearray),
    unpack=(lambda t, r, e: t(r(r(1)[0] % 16))),
    simplify=(lambda v: (v[1:], v[:-1])),
)

register_literal_type(
    typing.SupportsBytes,
    unpack=(lambda t, r, e: unpack_type(bytes, r, e)))

# register_literal_type(
#    Hashable,
#    unpack=unpack_basic_value)

register_literal_type(
    ByteString,
    unpack=lambda t, r, e: unpack_type(bytearray, r, e))

register_literal_type(
    (collections.deque, Deque),
    unpack=(lambda t, r, e: collections.deque(
        unpack_generator(type_param(t, 0), r, e))))

register_literal_type(
    NamedTuple,
    unpack=unpack_namedtuple)

register_literal_type(
    (set, Set, MutableSet),
    unpack=(lambda t, r, e: set(unpack_generator(type_param(t, 0), r, e))))

register_literal_type(
    (frozenset, FrozenSet),
    unpack=(lambda t, r, e: frozenset(
        unpack_generator(type_param(t, 0), r, e))))

register_literal_type(
    (dict, Mapping, MutableMapping),
    unpack=(lambda t, r, e: {
        unpack_type(type_param(t, 0), r, e):
        unpack_type(type_param(t, 1), r, e)
        for _ in range(r(1)[0])}))

# TODO : misc collections.*

register_literal_type(
    (Tuple, tuple),
    z3fn=smt_var,
    unpack=unpack_tuple)

register_literal_type(
    Callable,
    unpack=lambda t, r, e: FuzzFunc({}, unpack_type(type_param(t, -1), r, e)))

register_literal_type(
    (list, List, MutableSequence, Reversible,
     Container, Sized, Collection, Sequence),
    z3fn=smt_var,
    unpack=(lambda t, r, e: list(unpack_generator(type_param(t, 0), r, e))))

register_literal_type(
    (Iterable, Iterator, Generator),
    unpack=(lambda t, r, e: unpack_generator(type_param(t, 0), r, e)))

# TODO : various *Views, async, context mgr


TYPES = list(x for x in unpack_type_literals.keys() if type(x) == type)

register_literal_type(
    Any,
    z3fn=None,
    unpack=unpack_basic_value)

register_literal_type(
    Type,
    unpack=(lambda t, r, e: TYPES[r(1)[0] % len(TYPES)]))


class InputNotUnpackableError(Exception):
    pass


class ExpressionNotSmtable(Exception):
    pass


class UnpackEnv:
    def __init__(self):
        self.visited: List[Type] = []
        self.type_vars: Dict[str, Type] = {}


def unpack_signature(
        sig: inspect.Signature,
        reader: Callable[[int], bytearray],
        env: UnpackEnv = None) -> Tuple[list, dict]:
    '''
    post: sig.bind(*return[0], **return[1])
    '''
    if env is None:
        env = UnpackEnv()
    args: list = []
    using_kw_args = False
    kwargs: dict = {}
    debug('start (', ', '.join(map(str, sig.parameters.values())), ')')
    for param in sig.parameters.values():
        has_annotation = (param.annotation != inspect.Parameter.empty)
        if has_annotation:
            value = unpack_type(param.annotation, reader, env)
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            if not has_annotation:
                value = unpack_type(List[Any], reader, env)
            args.extend(value)
            using_kw_args = True
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            if not has_annotation:
                value = unpack_type(Mapping[str, Any], reader, env)
            kwargs.update(value)
        else:
            if not has_annotation:
                value = unpack_type(Any, reader, env)  # type: ignore
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                using_kw_args = True
            if using_kw_args:
                kwargs[param.name] = value
            else:
                args.append(value)
    debug('result', list(sig.parameters.keys()), ' a:', args, ' kw:', kwargs)
    return args, kwargs


def make_invocation_unpacker(fn: Callable, sig: inspect.Signature):
    def unpack_sig(t, r, e):
        args, kwargs = unpack_signature(sig, r, e)
        debug('  attempting to create', fn, args, kwargs)
        try:
            return fn(*args, **kwargs)
        except BaseException as e:
            raise InputNotUnpackableError(e)
    return unpack_sig


def reresolve(typ):
    '''
    Intent is to bring pyi definitions back into real code space.
    Causes problems though, not sure if it's worth it
    '''
    if not hasattr(typ, '__qualname__'):  # Union, etc
        return typ
    if hasattr(typ, '__origin__'):  # do not do for parameterized types
        return typ
    while typ.__qualname__ == 'NewType.<locals>.new_type':
        # typing.NewType can be treated at the same as its parent type
        typ = typ.__supertype__
    module = typ.__module__
    if module == 'builtins':
        return typ
    env = sys.modules[module].__dict__
    ret = env[typ.__name__]
    debug('reresolve ', typ, ' to ', ret)
    return ret


def _assert_make_reader_partial(i: int, buf: bytearray):
    '''post: return'''
    return make_reader(buf)(i) == buf[:i] if i <= len(buf) else True


def make_reader(buf: bytearray) -> Callable[[int], bytearray]:
    '''
    post: return(len(buf)) == buf
    '''
    bufholder = [buf]

    def reader(num_bytes):
        if len(bufholder[0]) < num_bytes:
            bufholder[0] = memoryview(bytearray(bufholder[0]) +
                                      bytearray(b'\0' * (num_bytes + 64)))
        b = bufholder[0]
        bufholder[0] = b[num_bytes:]
        return b[:num_bytes]
    return reader


@memo
def unpacker_for_type(typ: Type) -> Callable:
    unpacker = unpack_type_literals.get(typ)

    if unpacker is not None:
        return unpacker

    # Try for a non-exact type match:
    root_type = getattr(typ, '__origin__', typ)
    if root_type is Union:
        return (lambda t, r, e: unpack_type(type_param(t, r(1)[0] %
                                                       len(t.__args__)), r, e))
    for curtype, curhandler in unpack_type_literals.items():
        if getattr(curtype, '_is_protocol', False):
            continue
        if curtype is Any:
            continue
        matches = type_matches(root_type, curtype)
        if matches:
            debug('  matches: ', typ, curtype, matches)
            return curhandler

    # attempt to create one from constructor arguments:
    debug('  init', typ.__init__)
    if typ.__init__ is object.__init__:
        return lambda t, r, e: typ()
    sig = signature(typ.__init__)
    sig = inspect.Signature([p for (k, p) in sig.parameters.items()
                             if k != 'self'])
    return make_invocation_unpacker(typ, sig)


def unpack_type(typ: Type, reader: Callable[[int], bytearray],
                env: UnpackEnv) -> Any:
    if env is None:
        env = UnpackEnv()
    else:
        if typ in env.visited:
            raise InputNotUnpackableError(Exception(
                'refusing to recursively instantiate types: ' +
                str(env.visited + [typ])))

    if type(typ) is TypeVar:
        # TODO: figure out what to do about contravariance; right now only
        # used on the send() type for generators and coroutines
        varname = typ.__name__
        if varname in env.type_vars:
            resolved_type = env.type_vars[varname]
            ret = unpack_type(resolved_type, reader, env)
            if type(ret) is not resolved_type:
                if not typ.__covariant__:
                    raise InputNotUnpackableError(Exception(
                        'invariance requirement not met'))
        else:
            if typ.__constraints__:
                ret = unpack_type(choose(typ.__constraints__, reader),
                                  reader, env)
            else:
                ret = unpack_basic_value(Any, reader, env)
            env.type_vars[varname] = type(ret)
        return ret

    env.visited.append(typ)
    try:
        typ = reresolve(typ)
        unpacker = unpacker_for_type(typ)

        try:
            ret = unpacker(typ, reader, env)
            return ret
        except UnicodeDecodeError as e:
            raise InputNotUnpackableError(e)
    finally:
        env.visited.pop()


def _assert_unpack_type_is_same_type(typ: Type, buf: bytearray) -> bool:
    '''
    post: type_matches(type(return), typ)
    throws: InputNotUnpackableError
    '''
    return unpack_type(typ, make_reader(buf), UnpackEnv())


def _assert_null_buffer_is_unbackable(typ: Type) -> bool:
    '''
    post: True
    '''
    return unpack_type(typ, make_reader(bytearray(b'')), UnpackEnv())


def simplify_value(value: Any) -> Any:
    '''
    post: all(type(i) == type(value) for i in return)
    '''
    simplifier = simplify_literals.get(type(value))
    if simplifier is not None:
        ret = simplifier(value)
        assert all(type(v) == type(value) for v in ret)
        return (v for v in ret if v != value)
    else:
        return ()
