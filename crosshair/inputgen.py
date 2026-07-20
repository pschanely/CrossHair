"""
Shared valid-input generation for CrossHair's operation corpus.

Lifted out of ``crosshair.tools.measure_support`` so that the support-measurement
tool, the patch-equivalence tests, and the symbolic/concrete fuzz tests can all
draw from ONE typeshed-driven surface enumerator + valid-input generator instead
of three hand-rolled ones.

Layers exposed here:
  * surface enumeration -- typeshed signatures for builtin (type, method) pairs
    (MRO-resolved) and module-level free functions.
  * valid-input resolution -- the "prior ladder" above bare type fuzzing:
    Literal sets, runtime param-role registries (codec/error/hash names), and
    roundtrip construction for decoders (feed enc(<fuzzed>) to a decoder).
  * ``valid_inputs(native_callable)`` -- the public bridge: concrete, valid
    argument tuples for an arbitrary builtin/stdlib callable.
"""

import ast as _ast
import contextlib
import copy
import functools
import importlib
import io
import multiprocessing
import sys
import typing
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

from hypothesis import HealthCheck, given
from hypothesis import seed as hyp_seed
from hypothesis import settings
from hypothesis import strategies as st

from crosshair.auditwall import SideEffectDetected, enabled_auditwall

# ---------------------------------------------------------------------------
# the operation surface: builtin types + their methods
# ---------------------------------------------------------------------------
OP_DUNDERS = [
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
    "__divmod__",
    "__matmul__",
    "__neg__",
    "__abs__",
    "__round__",
    "__and__",
    "__or__",
    "__xor__",
    "__lshift__",
    "__rshift__",
    "__invert__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__hash__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__contains__",
    "__len__",
    "__iter__",
    "__next__",
    "__reversed__",
    "__int__",
    "__float__",
    "__bool__",
    "__str__",
    "__repr__",
    "__bytes__",
    "__index__",
    "__format__",
]


def _is_namedtuple(typ: type) -> bool:
    """A typing.NamedTuple / collections.namedtuple class: a tuple subclass carrying
    the namedtuple protocol (``_fields`` tuple + the ``_make`` builder)."""
    return (
        isinstance(typ, type)
        and issubclass(typ, tuple)
        and isinstance(getattr(typ, "_fields", None), tuple)
        and callable(getattr(typ, "_make", None))
    )


def surface(typ: type) -> List[str]:
    # Public methods only: drop every underscore-prefixed name -- both dunders (the
    # operators we want are re-added from OP_DUNDERS below) and single-underscore
    # privates (`_check_int_address`, `_ip_int_from_string`, ...), which are internal
    # implementation details typeshed doesn't annotate and aren't part of the surface.
    methods = [
        n for n in dir(typ) if not n.startswith("_") and callable(getattr(typ, n, None))
    ]
    # namedtuple's PUBLIC api is underscore-prefixed by design (_make/_replace/_asdict),
    # so the private-name filter above drops it; add it back for namedtuple classes
    # (e.g. urllib.parse.SplitResult._replace / ._asdict).
    if _is_namedtuple(typ):
        methods += [
            n
            for n in ("_make", "_replace", "_asdict")
            if callable(getattr(typ, n, None))
        ]
    return sorted(methods) + [n for n in OP_DUNDERS if hasattr(typ, n)]


# How to *invoke* a method/operator on a receiver named ``a``.  Operators must be
# applied with real operator syntax (``a >= b``), NOT the type's dunder descriptor
# (``list.__ge__(a, b)`` rejects a symbolic receiver), so both the support
# measurement and the differential test build a source expression and eval it.
_BINOP = {
    "__add__": "+",
    "__sub__": "-",
    "__mul__": "*",
    "__truediv__": "/",
    "__floordiv__": "//",
    "__mod__": "%",
    "__pow__": "**",
    "__matmul__": "@",
    "__and__": "&",
    "__or__": "|",
    "__xor__": "^",
    "__lshift__": "<<",
    "__rshift__": ">>",
    "__eq__": "==",
    "__ne__": "!=",
    "__lt__": "<",
    "__le__": "<=",
    "__gt__": ">",
    "__ge__": ">=",
}
_UNARY = {"__neg__": "-{a}", "__pos__": "+{a}", "__invert__": "~{a}"}
_CALLOP = {
    "__abs__": "abs({a})",
    "__len__": "len({a})",
    "__str__": "str({a})",
    "__repr__": "repr({a})",
    "__bool__": "bool({a})",
    "__int__": "int({a})",
    "__float__": "float({a})",
    "__bytes__": "bytes({a})",
    "__hash__": "hash({a})",
    "__round__": "round({a})",
    "__format__": "format({a})",
}
# Dunders with no value-comparable forward form (iterators / identity-ish junk).
# ``__setitem__``/``__delitem__`` are intentionally absent: they return None but
# mutate, so callers that track post-state can still drive them.
SKIP_DUNDERS = {
    "__init__",
    "__new__",
    "__init_subclass__",
    "__subclasshook__",
    "__class__",
    "__iter__",
    "__next__",
    "__reversed__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__getnewargs__",
    "__reduce__",
    "__reduce_ex__",
    "__sizeof__",
    "__dir__",
}


def receiver_name(argnames: Sequence[str]) -> str:
    """A receiver identifier that won't collide with any argument name (some
    typeshed signatures name a parameter ``a``, which would otherwise duplicate
    the synthesized receiver)."""
    recv = "a"
    while recv in argnames:
        recv = "_" + recv
    return recv


def call_expr(method: str, argnames: Sequence[str], recv: str = "a") -> Optional[str]:
    """The source expression invoking ``method`` on receiver ``recv`` with the
    given argument names, or None when an operator form needs an argument the
    signature doesn't supply."""
    if method in _BINOP:
        return f"{recv} {_BINOP[method]} {argnames[0]}" if argnames else None
    if method == "__divmod__":
        return f"divmod({recv}, {argnames[0]})" if argnames else None
    if method in _UNARY and not argnames:
        return _UNARY[method].format(a=recv)
    if method in _CALLOP and not argnames:
        return _CALLOP[method].format(a=recv)
    if method == "__contains__":
        return f"{argnames[0]} in {recv}" if argnames else None
    if method == "__getitem__":
        return f"{recv}[{argnames[0]}]" if argnames else None
    return f"{recv}.{method}({', '.join(argnames)})"


ANN_NS = vars(typing) | {
    "int": int,
    "str": str,
    "bytes": bytes,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "bytearray": bytearray,
    "datetime": __import__("datetime"),
    "collections": __import__("collections"),
}


# ---------------------------------------------------------------------------
# Explicit construction strategies for stdlib types Hypothesis's from_type can't
# build.  from_type falls back to builds(cls), which needs an introspectable
# signature or annotated fields.  C-extension types hide their signature
# (inspect.signature(array.array) raises "no signature found for builtin type")
# and the stdlib namedtuples expose fields but carry no annotations (ParseResult,
# DecimalTuple, struct_time), so builds() calls the constructor with zero
# arguments and raises.  Without a strategy for the receiver, every method of such
# a type measures as an empty "no valid input found" grey cell -- and no number of
# fuzz attempts ever helps, because the strategy can't be built in the first place.
# sized() consults this registry before its from_type fallback, so both the
# support map and valid_inputs() get real receivers.  Each factory takes the size
# n so the constructed value grows with the sweep.
# ---------------------------------------------------------------------------
def _small_ints(n: int) -> "st.SearchStrategy[int]":
    return st.integers(min_value=-(10**n), max_value=10**n)


def _array_strategy(n: int) -> "st.SearchStrategy[Any]":
    import array

    signed = st.integers(min_value=-100, max_value=100)  # fits the narrowest ('b')
    unsigned = st.integers(min_value=0, max_value=200)  # fits the narrowest ('B')
    codes = {
        "b": signed,
        "h": signed,
        "i": signed,
        "l": signed,
        "q": signed,
        "B": unsigned,
        "H": unsigned,
        "I": unsigned,
        "L": unsigned,
        "Q": unsigned,
        "f": st.floats(allow_nan=False, allow_infinity=False, width=32),
        "d": st.floats(allow_nan=False, allow_infinity=False),
    }

    def make(code: str) -> "st.SearchStrategy[Any]":
        return st.lists(codes[code], min_size=n, max_size=n).map(
            lambda xs: array.array(code, xs)
        )

    return st.sampled_from(list(codes)).flatmap(make)


def _struct_time_strategy(n: int) -> "st.SearchStrategy[Any]":
    import time

    return st.tuples(
        st.integers(1970, 2038),
        st.integers(1, 12),
        st.integers(1, 28),
        st.integers(0, 23),
        st.integers(0, 59),
        st.integers(0, 61),
        st.integers(0, 6),
        st.integers(1, 366),
        st.integers(-1, 1),
    ).map(time.struct_time)


def _decimaltuple_strategy(n: int) -> "st.SearchStrategy[Any]":
    from decimal import DecimalTuple  # (sign, digits, exponent)

    return st.builds(
        DecimalTuple,
        st.sampled_from((0, 1)),
        st.lists(st.integers(0, 9), min_size=1, max_size=max(n, 1)).map(tuple),
        st.integers(min_value=-(n + 1), max_value=n + 1),
    )


def _namedtuple_strategy(cls: type, elem: Any) -> Any:
    """A factory building ``cls`` from one ``elem(n)`` strategy per field -- for the
    stdlib namedtuples (urllib.parse results) whose unannotated fields defeat builds().
    """

    def factory(n: int) -> "st.SearchStrategy[Any]":
        return st.builds(cls, *[elem(n) for _ in cls._fields])  # type: ignore[attr-defined]

    return factory


def _deque_strategy(n: int) -> "st.SearchStrategy[Any]":
    from collections import deque

    return st.lists(_small_ints(1), min_size=n, max_size=n).map(deque)


def _userlist_strategy(n: int) -> "st.SearchStrategy[Any]":
    from collections import UserList

    return st.lists(_small_ints(1), min_size=n, max_size=n).map(UserList)


def _userstring_strategy(n: int) -> "st.SearchStrategy[Any]":
    from collections import UserString

    return st.text(min_size=n, max_size=n).map(UserString)


def _dict_map_strategy(ctor: Any) -> Any:
    """A factory building ``ctor(dict)`` -- for OrderedDict/UserDict/Counter/ChainMap,
    all of which accept a single mapping positional argument."""

    def factory(n: int) -> "st.SearchStrategy[Any]":
        return st.dictionaries(
            _small_ints(1), _small_ints(1), min_size=n, max_size=n
        ).map(ctor)

    return factory


def _defaultdict_strategy(n: int) -> "st.SearchStrategy[Any]":
    from collections import defaultdict  # first arg must be a callable default_factory

    return st.dictionaries(_small_ints(1), _small_ints(1), min_size=n, max_size=n).map(
        lambda d: defaultdict(int, d)
    )


@functools.lru_cache(maxsize=1)
def _type_strategies() -> Dict[type, Any]:
    import array
    from collections import (
        ChainMap,
        Counter,
        OrderedDict,
        UserDict,
        UserList,
        UserString,
        defaultdict,
        deque,
    )
    from decimal import DecimalTuple
    from time import struct_time
    from urllib.parse import (
        DefragResult,
        DefragResultBytes,
        ParseResult,
        ParseResultBytes,
        SplitResult,
        SplitResultBytes,
    )

    def text(n: int) -> "st.SearchStrategy[Any]":
        return st.text(max_size=max(n, 1))

    def binary(n: int) -> "st.SearchStrategy[Any]":
        return st.binary(max_size=max(n, 1))

    return {
        array.array: _array_strategy,
        deque: _deque_strategy,
        UserList: _userlist_strategy,
        UserString: _userstring_strategy,
        OrderedDict: _dict_map_strategy(OrderedDict),
        UserDict: _dict_map_strategy(UserDict),
        Counter: _dict_map_strategy(Counter),
        ChainMap: _dict_map_strategy(ChainMap),
        defaultdict: _defaultdict_strategy,
        struct_time: _struct_time_strategy,
        DecimalTuple: _decimaltuple_strategy,
        ParseResult: _namedtuple_strategy(ParseResult, text),
        SplitResult: _namedtuple_strategy(SplitResult, text),
        DefragResult: _namedtuple_strategy(DefragResult, text),
        ParseResultBytes: _namedtuple_strategy(ParseResultBytes, binary),
        SplitResultBytes: _namedtuple_strategy(SplitResultBytes, binary),
        DefragResultBytes: _namedtuple_strategy(DefragResultBytes, binary),
    }


def sized(ann: Any, n: int) -> "st.SearchStrategy[Any]":
    origin = typing.get_origin(ann)
    if ann is int:
        return st.integers(min_value=-(10**n), max_value=10**n)
    if ann is str:
        return st.text(min_size=n, max_size=n)
    if ann is bytes:
        return st.binary(min_size=n, max_size=n)
    if ann is float:
        return st.floats(allow_nan=False, allow_infinity=False)
    if ann is complex:
        return st.complex_numbers(allow_nan=False, allow_infinity=False)
    if ann is bool:
        return st.booleans()
    if ann is bytearray:
        return st.binary(min_size=n, max_size=n).map(bytearray)
    if origin in (list, typing.List):
        a = typing.get_args(ann) or (int,)
        return st.lists(sized(a[0], 1), min_size=n, max_size=n)
    if origin in (tuple, typing.Tuple):
        a = typing.get_args(ann)
        if len(a) == 2 and a[1] is Ellipsis:  # Tuple[X, ...]
            return st.lists(sized(a[0], 1), min_size=n, max_size=n).map(tuple)
        return st.tuples(*[sized(x, 1) for x in (a or (int,))])
    if origin in (set, typing.Set, frozenset, typing.FrozenSet):
        a = typing.get_args(ann) or (int,)
        s = st.sets(sized(a[0], 1), min_size=n, max_size=n)
        return s.map(frozenset) if origin in (frozenset, typing.FrozenSet) else s
    if origin in (dict, typing.Dict):
        a = typing.get_args(ann) or (int, int)
        return st.dictionaries(sized(a[0], 1), sized(a[1], 1), min_size=n, max_size=n)
    if isinstance(ann, type):  # stdlib types from_type can't construct (see registry)
        factory = _type_strategies().get(ann)
        if factory is not None:
            return factory(n)
    return st.from_type(ann)


def _jsonable(n: int) -> "st.SearchStrategy[Any]":
    leaf = st.none() | st.booleans() | st.integers() | st.text(max_size=n)
    return st.recursive(
        leaf,
        lambda c: st.lists(c, max_size=n)
        | st.dictionaries(st.text(max_size=3), c, max_size=n),
        max_leaves=n + 1,
    )


def _arg_strategy(spec: Any, n: int) -> "st.SearchStrategy[Any]":
    """Build a Hypothesis strategy for one argument.  ``spec`` is one of:
    ("sampled", values)  -- pick from a known set (Literal values, codec/error/hash names);
    ("smallint",) -- a small index/key, biased to land in range for size-<=5 inputs;
    ("roundtrip", enc_module, enc_func, base_kind) -- generate a valid encoded
        input by running the inverse encoder on fuzzed base data;
    a plain type -- the default size-based fuzz."""
    tag = spec[0] if isinstance(spec, tuple) and spec else None
    if tag == "sampled":
        return st.sampled_from(list(spec[1]))
    if tag == "smallint":
        return st.integers(min_value=-8, max_value=8)
    if tag == "roundtrip":
        _, em, ef, base = spec
        base_strat = (
            _jsonable(n)
            if base == "jsonable"
            else (
                st.text(min_size=n, max_size=n)
                if base == "str"
                else st.binary(min_size=n, max_size=n)
            )
        )
        try:
            enc = getattr(importlib.import_module(em), ef)
        except Exception:  # encoder unavailable -> fall back to raw base data
            return base_strat
        return base_strat.map(enc)
    return sized(spec, n)


# --- typeshed access, pinned to the RUNNING interpreter --------------------
# get_stub_names evaluates `sys.version_info`/`sys.platform` guards for the given
# search context, so the surface matches what THIS interpreter actually has: no
# version skew (no math.fma on 3.12), no manual `if` descent, and re-exports
# (bisect_left <- _bisect) and class members come pre-resolved.
_SEARCH_CTX: Any = None
_STUB_NAMES: Dict[str, Dict[str, Any]] = {}  # module -> {name: NameInfo}
# builtins holds the concrete types + object; typing holds the ABC bases
# (MutableSequence/Mapping/...) where typeshed declares inherited methods.
_STUB_CLASS_MODULES = ("builtins", "typing")


def _search_ctx() -> Any:
    global _SEARCH_CTX
    if _SEARCH_CTX is None:
        import typeshed_client as _tc

        _SEARCH_CTX = _tc.get_search_context(
            version=sys.version_info[:2], platform=sys.platform
        )
    return _SEARCH_CTX


def _stub_names(module: str) -> Dict[str, Any]:
    """Lazily fetch the version/platform-resolved {name: NameInfo} for a module."""
    if module not in _STUB_NAMES:
        import typeshed_client as _tc

        try:
            _STUB_NAMES[module] = (
                _tc.get_stub_names(module, search_context=_search_ctx()) or {}
            )
        except Exception:
            _STUB_NAMES[module] = {}
    return _STUB_NAMES[module]


def _funcdefs(ni: Any, _depth: int = 0) -> List[Any]:
    """FunctionDefs behind a NameInfo: a plain def, an @overload group, or a
    re-export (followed across modules), else []."""
    if ni is None or _depth > 4:
        return []
    import typeshed_client as _tc

    node = ni.ast
    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
        return [node]
    if isinstance(node, _tc.OverloadedName):
        return [
            d
            for d in node.definitions
            if isinstance(d, (_ast.FunctionDef, _ast.AsyncFunctionDef))
        ]
    if isinstance(
        node, _tc.ImportedName
    ):  # re-export, e.g. bisect.bisect_left <- _bisect
        return _funcdefs(
            _stub_names(".".join(node.module_name)).get(cast(str, node.name)),
            _depth + 1,
        )
    return []


def _stub_class(name: str, module: str = "builtins", _depth: int = 0) -> Optional[Any]:
    """The version-resolved (class NameInfo, defining module) for ``name``, searched
    in ``(module, "builtins", "typing")`` and following cross-module re-exports
    (e.g. ``fractions.Fraction``'s base ``Rational`` -> ``numbers.Rational``).
    Returns the module the ClassDef actually lives in so base-following can search
    there next."""
    if _depth > 4:
        return None
    import typeshed_client as _tc

    for mod in (module, *_STUB_CLASS_MODULES):
        ni = _stub_names(mod).get(name)
        if ni is None:
            continue
        if isinstance(ni.ast, _ast.ClassDef):
            return (ni, mod)
        if isinstance(ni.ast, _tc.ImportedName):  # re-export -> follow to its module
            return _stub_class(
                cast(str, ni.ast.name or name),
                ".".join(ni.ast.module_name),
                _depth + 1,
            )
    return None


def _base_ref(base: Any, default_mod: str) -> Optional[Tuple[str, str]]:
    """(base_class_name, module_to_resolve_it_in) for a base-class AST node.

    A bare ``Name`` (``class Fraction(Rational)``) resolves in the current class's
    module (and _stub_class follows any re-export from there).  A dotted
    ``Attribute`` (``class RegexFlag(enum.IntFlag)``) resolves its final attr in the
    *qualifier* module -- ``enum`` here, ``os.path`` for ``os.path.X`` -- so bases
    imported as ``import enum`` (not ``from enum import IntFlag``) still follow to
    where the class is actually defined instead of dead-ending in the child module."""
    if isinstance(base, _ast.Name):
        return (base.id, default_mod)
    if isinstance(base, _ast.Attribute):
        parts = []
        node: Any = base.value
        while isinstance(node, _ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, _ast.Name):
            parts.append(node.id)
            return (base.attr, ".".join(reversed(parts)))
    return None


def _class_chain(cls_name: str, module: str = "builtins") -> List[Any]:
    """Typeshed MRO for cls_name: derived-first class NameInfos, following declared
    bases (subscripts stripped), with ``object`` always last.  ``module`` is the
    class's owning module; each base resolves in the module named by its qualifier
    (dotted bases) or the module where the current class was found (bare names)."""
    chain: List[Any] = []
    seen: Set[str] = set()

    def visit(name: str, mod: str) -> None:
        if name in seen:
            return
        res = _stub_class(name, mod)
        if res is None:
            return
        ni, found_mod = res
        seen.add(name)
        chain.append(ni)
        for b in ni.ast.bases:
            base = b.value if isinstance(b, _ast.Subscript) else b
            ref = _base_ref(base, found_mod)
            if ref and ref[0] not in ("Generic", "Protocol"):
                visit(*ref)

    visit(cls_name, module)
    if "object" not in seen:
        obj = _stub_class("object")
        if obj is not None:
            chain.append(obj[0])
    return chain


# The generatable type for every unconstrained "object-like" slot: a bare
# object/Any parameter, an unbound element TypeVar, or a generic container's element
# type.
GENERIC = "int"  # TODO: Something like "Union[int, float, str]" (requires speeding up pinning)


# receiver annotation + TypeVar bindings.  Bind all of typeshed's element TypeVar
# names (_T/_T_co/_KT/_VT) to the container's element type.
def _elem(t: str = GENERIC) -> Dict[str, str]:
    return {"_T": t, "_S": t, "_T_co": t, "_KT": t, "_VT": t}


RECV = {
    int: ("int", {}),
    float: ("float", {}),
    bool: ("bool", {}),
    str: ("str", _elem("str")),
    bytes: ("bytes", _elem("int")),
    bytearray: ("bytearray", _elem("int")),
    list: (f"List[{GENERIC}]", _elem()),
    tuple: (f"Tuple[{GENERIC}, ...]", _elem()),
    dict: (f"Dict[{GENERIC}, {GENERIC}]", _elem()),
    set: (f"Set[{GENERIC}]", _elem()),
    frozenset: (f"FrozenSet[{GENERIC}]", _elem()),
}


def _extend_recv() -> None:
    """Register the non-builtin-value receiver types that have both a symbolic
    proxy and a concrete construction strategy, so their methods are drivable.
    The generic containers bind their element TypeVars like the builtins above."""
    import array
    import collections
    import datetime
    import decimal
    import fractions
    import random
    import re

    RECV.update(
        {
            complex: ("complex", {}),
            memoryview: ("memoryview", {}),
            range: ("range", {}),
            datetime.date: ("datetime.date", {}),
            datetime.datetime: ("datetime.datetime", {}),
            datetime.time: ("datetime.time", {}),
            datetime.timedelta: ("datetime.timedelta", {}),
            datetime.timezone: ("datetime.timezone", {}),
            decimal.Decimal: ("decimal.Decimal", {}),
            fractions.Fraction: ("fractions.Fraction", {}),
            random.Random: ("random.Random", {}),
            re.Match: ("re.Match", {}),
            re.Pattern: ("re.Pattern", {}),
            array.array: ("array.array", {}),
            collections.deque: ("collections.deque", _elem()),
            collections.OrderedDict: ("collections.OrderedDict", _elem()),
            collections.Counter: ("collections.Counter", _elem()),
            collections.ChainMap: ("collections.ChainMap", _elem()),
            collections.defaultdict: ("collections.defaultdict", _elem()),
        }
    )


_extend_recv()


# typeshed leaf names -> a fuzzable annotation.  Beyond the concrete builtins,
# this maps the common stdlib type-vocabulary the probe surfaced: numeric
# protocols (math/statistics), generic element TypeVars (free functions have no
# receiver to bind them, so they default to GENERIC -- for builtin methods RECV
# binds win since _map_ann checks `binds` first), and str/buffer families.  Over-broad
# mappings are safe: the fuzz-validation step drops any candidate whose call
# doesn't actually run.
_NAME_MAP = {
    "int": "int",
    "str": "str",
    "bytes": "bytes",
    "float": "float",
    "bool": "bool",
    "bytearray": "bytearray",
    "complex": "complex",
    "SupportsComplex": "complex",
    "_SupportsComplex": "complex",
    "SupportsIndex": "int",
    "_PositiveInteger": "int",
    "_NegativeInteger": "int",
    "ConvertibleToInt": "int",
    "ConvertibleToFloat": "float",
    "LiteralString": "str",
    "ReadableBuffer": "bytes",
    "memoryview": "bytes",
    "object": GENERIC,
    "Any": GENERIC,
    # numeric protocols / numeric TypeVars
    "_SupportsFloatOrIndex": "float",
    "SupportsFloat": "float",
    "_SupportsCeil": "float",
    "_SupportsFloor": "float",
    "_SupportsTrunc": "float",
    "_Number": "int",
    "_NumberT": "int",
    "_Numeric": "int",
    "SupportsAbs": "int",
    "SupportsRound": "int",
    "SupportsRichComparison": "int",
    "SupportsRichComparisonT": "int",
    "_SupportsComparison": "int",
    "_SupportsInversion": "int",
    "SupportsAdd": "int",
    "SupportsRAdd": "int",
    "_HashableT": "int",
    "Hashable": "int",
    "_MultiplicableT1": "int",
    "_MultiplicableT2": "int",
    "_SupportsProdNoDefaultT": "int",
    "_SupportsSumNoDefaultT": "int",
    # generic element TypeVars (the unbound-free-function default)
    "_T": GENERIC,
    "_S": GENERIC,
    "_U": GENERIC,
    "_T1": GENERIC,
    "_T2": GENERIC,
    "_KT": GENERIC,
    "_VT": GENERIC,
    "_K": GENERIC,
    "_V": GENERIC,
    "_T_co": GENERIC,
    "_T_contra": GENERIC,
    # string / buffer / path families
    "AnyStr": "str",
    "StrOrLiteralStr": "str",
    "AnyOrLiteralStr": "str",
    "StrPath": "str",
    "StrOrBytesPath": "str",
    "FileDescriptorOrPath": "str",
    "PathLike": "str",
    "_AsciiBuffer": "bytes",
    "WriteableBuffer": "bytes",
}


class _Unsupported(Exception):
    pass


def normalize(name: str, binds: Dict[str, str]) -> Optional[str]:
    """Map a typeshed leaf type NAME to a generatable annotation string, or None.

    The returned string serves as both the parameter annotation and (re-eval'd via
    :func:`_ann`) the fuzz strategy.  ``binds`` (Self / receiver / element TypeVars)
    take precedence over the default table.
    """
    if name in binds:
        return binds[name]
    return _NAME_MAP.get(name)


def _map_ann(node: Any, binds: Dict[str, str]) -> str:
    """typeshed annotation AST -> a fuzzable annotation string (or raise)."""
    if isinstance(node, _ast.Name):
        got = normalize(node.id, binds)
        if got is not None:
            return got
        raise _Unsupported(node.id)
    if isinstance(node, _ast.Attribute):  # typing.SupportsIndex -> SupportsIndex
        return _map_ann(_ast.Name(id=node.attr), binds)
    if isinstance(node, _ast.Constant):
        if node.value is None:
            raise _Unsupported("None")
        return _map_ann(_ast.Name(id=type(node.value).__name__), binds)
    if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.BitOr):  # unions
        for member in (node.left, node.right):
            if isinstance(member, _ast.Constant) and member.value is None:
                continue
            try:
                return _map_ann(member, binds)
            except _Unsupported:
                continue
        raise _Unsupported("union")
    if isinstance(node, _ast.Subscript):
        base = (
            node.value.id
            if isinstance(node.value, _ast.Name)
            else getattr(node.value, "attr", "")
        )
        sl = node.slice
        elts = sl.elts if isinstance(sl, _ast.Tuple) else [sl]
        if base in (
            "Iterable",
            "Sequence",
            "Collection",
            "MutableSequence",
            "Container",
            "list",
            "List",
        ):
            return f"List[{_map_ann(elts[0], binds)}]"
        if base in (
            "set",
            "Set",
            "AbstractSet",
            "MutableSet",
        ):  # operators need a set operand
            return f"Set[{_map_ann(elts[0], binds)}]"
        if base in ("frozenset", "FrozenSet"):
            return f"FrozenSet[{_map_ann(elts[0], binds)}]"
        if base in ("tuple", "Tuple"):
            inner = [
                e
                for e in elts
                if not (isinstance(e, _ast.Constant) and e.value is Ellipsis)
            ]
            if any(isinstance(e, _ast.Constant) and e.value is Ellipsis for e in elts):
                return f"Tuple[{_map_ann(inner[0], binds)}, ...]"
            return "Tuple[" + ", ".join(_map_ann(e, binds) for e in inner) + "]"
        if base in ("dict", "Dict", "SupportsKeysAndGetItem"):
            return f"Dict[{_map_ann(elts[0], binds)}, {_map_ann(elts[1], binds)}]"
        if base == "Literal":
            # Emit the Literal verbatim for constant kinds we can render
            # (int/str/bytes/bool); enum / qualified members fall back to the
            # underlying scalar type.
            rendered: Optional[List[str]] = []
            for e in elts:
                neg = isinstance(e, _ast.UnaryOp) and isinstance(e.op, _ast.USub)
                c = e.operand if isinstance(e, _ast.UnaryOp) else e  # Literal[-1]
                if isinstance(c, _ast.Constant) and type(c.value) in (
                    int,
                    str,
                    bytes,
                    bool,
                ):
                    r = repr(c.value)
                    rendered.append(f"-{r}" if neg else r)  # type: ignore[union-attr]
                else:
                    rendered = None
                    break
            if rendered:
                return f"Literal[{', '.join(rendered)}]"
            for e in elts:  # fall back: map to the underlying scalar type
                c = e.operand if isinstance(e, _ast.UnaryOp) else e
                if isinstance(c, _ast.Constant) and c.value is not None:
                    return _map_ann(_ast.Name(id=type(c.value).__name__), binds)
            raise _Unsupported("Literal")
        # a subscripted scalar protocol/TypeVar (SupportsAbs[_T], PathLike[AnyStr],
        # _SupportsInversion[_T_co], ...): the element type doesn't change how we
        # fuzz it, so fall back to the base name's scalar mapping.
        got = normalize(base, binds)
        if got is not None:
            return got
        raise _Unsupported(base)
    raise _Unsupported(type(node).__name__)


def _method_overloads(typ: type, method: str, module: str = "builtins") -> List[Any]:
    """typeshed FunctionDefs for typ.method, resolved up the MRO: the first class
    in the chain that defines it wins (so a derived override beats an inherited
    base), returning all (version-resolved) overloads from that class."""
    for ni in _class_chain(typ.__name__, module):
        fns = _funcdefs((ni.child_nodes or {}).get(method))
        if fns:
            return fns
    return []


def _overload_sigs(
    overloads: List[Any],
    binds: Dict[str, str],
    module: str,
    drop: Tuple[str, ...],
) -> List[List[Tuple[str, str, Tuple[Any, ...]]]]:
    """Map each overload's required positional args (receiver/names in ``drop``
    excluded) to [(argname, annotation_str, literal_values), ...], de-duplicated.
    An empty inner list means a zero-arg call; [] means no overload could be mapped.

    A *args op (set.update(*s), math.gcd(*ints), ...) with no required positionals
    also gets a candidate that passes one expanded vararg, so consumers that only
    take *args become drivable."""
    out, seen = [], set()
    for fn in overloads:
        pos = fn.args.posonlyargs + fn.args.args
        ndef = len(fn.args.defaults)
        required = pos[: len(pos) - ndef] if ndef else pos
        required = [a for a in required if a.arg not in drop]
        try:
            sig = tuple(
                (a.arg,) + _map_arg(a.annotation, binds, module)
                for a in required
                if a.annotation
            )
        except _Unsupported:
            continue
        if len(sig) != len(required):  # an arg lacked an annotation
            continue
        variants = [sig]
        if fn.args.vararg is not None and fn.args.vararg.annotation is not None:
            try:
                va = fn.args.vararg
                variants.append(
                    sig + ((va.arg,) + _map_arg(va.annotation, binds, module),)
                )
            except _Unsupported:
                pass
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(list(v))
    return out


# The reflexive comparison dunders: a user reads these as "compare two of the
# SAME type".  typeshed types the comparand of ==/!= as bare ``object`` (and some
# ordering dunders as ``object``/``Any`` too), which _NAME_MAP resolves to ``int``
# -- so e.g. str.__eq__ would be measured as str-vs-int: a vacuous always-False
# compare that CrossHair "solves" for any input, yielding a meaningless green cell
# with an unreadable astral-plane demo.  For these methods we bind the generic
# comparand annotation to the receiver's own type instead (a no-op where typeshed
# already types the arg concretely, e.g. str.__lt__(value: str)).
_REFLEXIVE_CMP = frozenset({"__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"})


@functools.lru_cache(maxsize=None)
def _candidate_sigs(
    typ: type, method: str, module: str = "builtins"
) -> List[List[Tuple[str, str, Tuple[Any, ...]]]]:
    """Candidate signatures per typeshed overload of typ.method (see _overload_sigs).
    ``module`` is the type's owning module; for a non-builtin class the receiver
    carries no element-TypeVar binds and arg annotations resolve against ``module``.
    ``Self`` binds to the receiver's own annotation, so methods taking another
    instance (ipaddress ``subnet_of(other: Self)``, ...) become drivable."""
    recv_ann, recv_binds = RECV.get(typ, (f"{module}.{typ.__name__}", {}))
    binds = {"Self": recv_ann, **recv_binds}
    if method in _REFLEXIVE_CMP:  # measure ==/!=/ordering against the SAME type
        binds = {**binds, "object": recv_ann, "Any": recv_ann}
    return _overload_sigs(
        _method_overloads(typ, method, module), binds, module, ("self",)
    )


@functools.lru_cache(maxsize=None)
def _ann(s: str) -> Any:
    ns = dict(ANN_NS)
    # dotted names (e.g. "decimal.Decimal", "List[fractions.Fraction]") need their
    # leading module imported into the eval namespace; import lazily on demand.
    for _ in range(6):
        try:
            return eval(s, ns)
        except NameError as e:
            name = getattr(e, "name", None)
            if not name or name in ns:
                raise
            ns[name] = importlib.import_module(name)
        except AttributeError:
            # The leading name resolved to the WRONG object -- ANN_NS derives from
            # vars(typing), which carries typing's deprecated submodule aliases
            # (typing.re, typing.io), so "re.RegexFlag" hits typing.re (no RegexFlag)
            # instead of the stdlib re.  Rebind the leading name to the real module
            # and retry; bail if it's already real or not an importable module.
            head = s.split("[", 1)[0].split(".", 1)[0]
            try:
                real = importlib.import_module(head)
            except ImportError:
                raise
            if ns.get(head) is real:
                raise
            ns[head] = real
    return eval(s, ns)


# ---------------------------------------------------------------------------
# valid-input resolver -- climb the prior ladder above bare type-based fuzzing
#   - Literal value-sets declared in the type (incl. named aliases)
#   - parameter-role known-value sets sourced from the live runtime
# A wrong/over-broad prior is safe: fuzz-validation still drops candidates whose
# call doesn't run.  (We DO aim for diverse values, not one degenerate easy one.)
# ---------------------------------------------------------------------------
_LIT_ALIASES: Dict[str, Dict[str, Tuple[Any, ...]]] = {}


def _module_literal_aliases(module: str) -> Dict[str, Tuple[Any, ...]]:
    """Lazily map module-level ``X = Literal[...]`` / ``X: TypeAlias = Literal[...]``
    aliases (e.g. unicodedata._NormalizationForm) to their value tuples."""
    if module not in _LIT_ALIASES:
        amap: Dict[str, Tuple[Any, ...]] = {}
        _LIT_ALIASES[module] = amap  # store first (guards alias->alias re-entry)
        for name, ni in _stub_names(module).items():
            node = ni.ast
            val = (
                node.value if isinstance(node, (_ast.Assign, _ast.AnnAssign)) else None
            )
            if val is not None:
                lv = _literal_values(val, module)
                if lv:
                    amap[name] = lv
    return _LIT_ALIASES[module]


_ALIAS_ANNSTR: Dict[str, Dict[str, str]] = {}
# AST shapes that denote a type expression (vs a value constant / a TypeVar call)
_ALIAS_VALUE_TYPES = (_ast.Name, _ast.Subscript, _ast.BinOp, _ast.Attribute)


def _module_alias_annstr(module: str) -> Dict[str, str]:
    """Lazily map module-level type aliases (e.g. cmath ``_C = SupportsFloat |
    SupportsComplex | ... | complex``) to a fuzzable annotation string, so a bare
    Name referring to an alias resolves.  These get folded into ``binds`` (which
    _map_ann checks first), which also handles alias-in-subscript (list[_C])."""
    if module not in _ALIAS_ANNSTR:
        resolved: Dict[str, str] = {}
        _ALIAS_ANNSTR[module] = resolved
        # Seed with the module's own public classes, so a bare-Name arg referring to
        # a sibling type (datetime.date taking `timedelta`/`datetime`, decimal /
        # ipaddress / fractions methods taking their peers, ...) resolves to a
        # qualified, fuzzable annotation.  Module-scoped (folded into binds only when
        # resolving THIS module's args), so it doesn't pollute the global _NAME_MAP;
        # over-broad is safe, since fuzz-validation drops any candidate that can't run.
        # Also lets the alias pass below chain private aliases (_Date = date, ...).
        try:
            mod = importlib.import_module(module)
        except Exception:
            mod = None
        if mod is not None:
            for cname, cni in _stub_names(module).items():
                # Skip names _NAME_MAP already handles: it deliberately simplifies
                # some builtins (object/memoryview -> int/bytes) to keep fuzzing
                # cheap, and seeding "builtins.object" here would override that and
                # feed arbitrary heavy objects into every Iterable[object] arg.
                if cname in _NAME_MAP or cname.startswith("_"):
                    continue
                if isinstance(cni.ast, _ast.ClassDef) and isinstance(
                    getattr(mod, cname, None), type
                ):
                    resolved[cname] = f"{module}.{cname}"
        raw: Dict[str, Any] = {}
        for name, ni in _stub_names(module).items():
            node = ni.ast
            val = (
                node.value if isinstance(node, (_ast.Assign, _ast.AnnAssign)) else None
            )
            if isinstance(val, _ALIAS_VALUE_TYPES):
                raw[name] = val
        for _ in range(3):  # a few passes to resolve alias->alias chains
            progress = False
            for name, val in raw.items():
                if name in resolved:
                    continue
                try:
                    resolved[name] = _map_ann(
                        val, resolved
                    )  # resolved doubles as binds
                    progress = True
                except _Unsupported:
                    pass
            if not progress:
                break
    return _ALIAS_ANNSTR[module]


def _literal_values(node: Any, module: str) -> Tuple[Any, ...]:
    """Tuple of concrete values if ``node`` is a Literal[...] (or an alias / a
    union containing one); else ()."""
    if isinstance(node, _ast.Subscript):
        base = (
            node.value.id
            if isinstance(node.value, _ast.Name)
            else getattr(node.value, "attr", "")
        )
        if base == "Literal":
            elts = (
                node.slice.elts if isinstance(node.slice, _ast.Tuple) else [node.slice]
            )
            return tuple(
                e.value
                for e in elts
                if isinstance(e, _ast.Constant) and e.value is not None
            )
    if isinstance(node, _ast.Name):
        return _module_literal_aliases(module).get(node.id, ())
    if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.BitOr):  # unions
        return _literal_values(node.left, module) + _literal_values(node.right, module)
    return ()


def _map_arg(
    annotation: Any, binds: Dict[str, str], module: str
) -> Tuple[str, Tuple[Any, ...]]:
    """(annotation_str, literal_values) for one arg.  Module type aliases (cmath
    _C, ...) are folded into binds so bare-Name aliases resolve; falls back to the
    Literal value's type when only a Literal alias is available."""
    lits = _literal_values(annotation, module)
    ext = dict(_module_alias_annstr(module))
    ext.update(binds)  # receiver TypeVar binds win over module aliases
    try:
        annstr = _map_ann(annotation, ext)
    except _Unsupported:
        if not lits:
            raise
        annstr = type(lits[0]).__name__
    return (annstr, lits)


def _codec_names() -> Tuple[str, ...]:
    # a diverse, realistic spread (NOT just utf-8) so the cliff still shows
    return (
        "utf-8",
        "ascii",
        "latin-1",
        "utf-16",
        "utf-32",
        "cp1252",
        "utf-8-sig",
        "iso-8859-2",
        "mac-roman",
        "big5",
        "shift_jis",
    )


def _hash_names() -> Tuple[str, ...]:
    import hashlib

    return tuple(sorted(hashlib.algorithms_available))


_ERRORS = (
    "strict",
    "ignore",
    "replace",
    "backslashreplace",
    "xmlcharrefreplace",
    "namereplace",
    "surrogateescape",
)


# parameter-role registry: param-name -> () -> values  (string-typed args only)
_PARAM_STRATS = {
    "encoding": _codec_names,
    "errors": lambda: _ERRORS,
    "byteorder": lambda: ("little", "big"),
}


# precise (module, func, param) overrides where the param name alone is ambiguous
_FUNC_ARG_STRATS = {
    ("hashlib", "new", "name"): _hash_names,
}


# decoders/parsers whose valid input is best produced by running the
# inverse encoder: (module, decoder) -> (enc_module, enc_func, base_kind).  The
# decoder's primary (first) arg is fed enc(<fuzzed base data>).
# --- custom per-op input strategies ---------------------------------------
# A registry of WHOLE-TUPLE input strategies, keyed by seedkey, for ops where
# independent per-arg fuzzing can't produce a useful input because the arguments
# must be CORRELATED or ALIASED:
#   * a decoder wants a valid ENCODED input (run the inverse encoder on fuzzed
#     base data) -- the former _ROUNDTRIP, now :func:`_roundtrip` entries;
#   * ``getattr(o, name)`` wants ``name`` to be an attribute OF ``o`` -- a random
#     string is almost always an AttributeError;
#   * ``x is x`` / ``x is not x`` needs the SAME object in both positions, which
#     independent draws never produce.
# Each value is ``(specs, size) -> strategy-over-the-full-arg-tuple``; it may build
# on the op's own per-arg ``specs`` (so it adapts to whatever arity a caller drives
# the op with) or ignore them.


def _roundtrip(
    enc_module: str, enc_func: str, base_kind: str
) -> Callable[[List[Any], int], "st.SearchStrategy[tuple]"]:
    """A strategy builder that feeds a decoder a VALID encoded input: fuzz base
    data, run ``enc_module.enc_func`` on it, pass that as the first argument, and
    fuzz any remaining args normally."""

    def build(specs: List[Any], n: int) -> "st.SearchStrategy[tuple]":
        first = _arg_strategy(("roundtrip", enc_module, enc_func, base_kind), n)
        rest = [_arg_strategy(s, n) for s in specs[1:]]
        return st.tuples(first, *rest)

    return build


def _getattr_inputs(specs: List[Any], n: int) -> "st.SearchStrategy[tuple]":
    """``getattr(o, name, ...)`` where ``name`` is a real (data) attribute of ``o``
    -- so it exercises actual attribute access instead of raising AttributeError on
    a fuzzed string.  ``o`` is drawn from its own annotation-derived strategy
    (``specs[0]``).  Preserves the op's arity (any trailing ``default`` fuzzes
    normally)."""
    objs = _arg_strategy(specs[0], n) if specs else st.integers()

    def with_name(o: Any) -> "st.SearchStrategy[tuple]":
        names = [
            a
            for a in dir(o)
            if not a.startswith("_") and not callable(getattr(o, a, None))
        ]
        name_strat = st.sampled_from(names) if names else st.just("__class__")
        rest = [_arg_strategy(s, n) for s in specs[2:]]
        return st.tuples(st.just(o), name_strat, *rest)

    return objs.flatmap(with_name)


def _aliased_pair(specs: List[Any], n: int) -> "st.SearchStrategy[tuple]":
    """``(x, x)`` -- the SAME object in both positions, to witness identity ops
    (``x is x``).  Uses non-interned values (large ints / str / list) so identity
    is meaningful (small ints and short strings may be cached)."""
    vals = st.one_of(
        st.integers(min_value=1000),
        st.text(min_size=1, max_size=n),
        st.lists(st.integers(), min_size=1, max_size=n),
    )
    return vals.map(lambda x: (x, x))


# How much a correlated strategy outweighs plain fuzzing in the blend below: ~80%
# transforming-regime inputs, ~20% independent draws.  The plain draws keep the
# no-op regime (a replace that matches nothing, a width that pads nothing) reachable
# for the differential/fuzz clients, so a bug that only bites there stays findable.
_CORRELATION_WEIGHT = 4


def _blend_with_plain(
    correlated: "st.SearchStrategy[tuple]", specs: List[Any], n: int
) -> "st.SearchStrategy[tuple]":
    """Draw mostly from ``correlated`` but occasionally from plain independent
    fuzzing, so a custom strategy sharpens coverage without dropping the cases the
    default fuzz already reached."""
    plain = st.tuples(*[_arg_strategy(s, n) for s in specs])
    return st.one_of(*([correlated] * _CORRELATION_WEIGHT + [plain]))


def _substring_of(a: str) -> "st.SearchStrategy[str]":
    """A (usually non-empty) substring of ``a`` -- the needle a search/replace op
    needs for its match to actually fire."""
    if not a:
        return st.just("")
    return st.integers(0, len(a) - 1).flatmap(
        lambda start: st.integers(start + 1, len(a)).map(lambda end: a[start:end])
    )


def _needle_inputs(specs: List[Any], n: int) -> "st.SearchStrategy[tuple]":
    """A search/replace op whose needle actually occurs: draw the receiver, then a
    substring of it for the first argument (``str.replace``/``count``/``find``/
    ``index``/...).  A fuzzed needle almost never occurs, so the op no-ops (returns
    the receiver, ``-1``, or ``0``) and never exercises real matching."""
    recv = _arg_strategy(specs[0], n)

    def build(a: str) -> "st.SearchStrategy[tuple]":
        rest = [_arg_strategy(s, n) for s in specs[2:]]
        return st.tuples(st.just(a), _substring_of(a), *rest)

    return _blend_with_plain(recv.flatmap(build), specs, n)


def _pad_to_width_inputs(specs: List[Any], n: int) -> "st.SearchStrategy[tuple]":
    """A justify/fill op whose width exceeds the receiver length, so it actually
    pads (``str.ljust``/``rjust``/``center``/``zfill``).  A fuzzed width is usually
    ``<= len`` (no-op) or enormous (unreadable), so neither the support map nor a
    generated demo shows real padding."""
    recv = _arg_strategy(specs[0], n)

    def build(a: str) -> "st.SearchStrategy[tuple]":
        width = st.integers(len(a) + 1, len(a) + max(n, 1) + 2)
        rest = [_arg_strategy(s, n) for s in specs[2:]]
        return st.tuples(st.just(a), width, *rest)

    return _blend_with_plain(recv.flatmap(build), specs, n)


_STRIPPABLE_WS = " \t\n\r\v\f"


def _surrounded_inputs(specs: List[Any], n: int) -> "st.SearchStrategy[tuple]":
    """A trim op whose receiver carries real surrounding whitespace, so it actually
    strips (``str.strip``/``lstrip``/``rstrip``).  A fuzzed string rarely has
    leading/trailing whitespace, so the op no-ops and returns the receiver
    unchanged; any explicit ``chars`` argument is drawn from the same whitespace so
    it strips too."""
    core = st.text(min_size=1, max_size=max(n, 1))
    pad = st.text(alphabet=_STRIPPABLE_WS, min_size=1, max_size=3)
    recv = st.builds(lambda left, mid, right: left + mid + right, pad, core, pad)

    def build(a: str) -> "st.SearchStrategy[tuple]":
        rest = [
            st.text(alphabet=_STRIPPABLE_WS, min_size=1, max_size=3) for _ in specs[1:]
        ]
        return st.tuples(st.just(a), *rest)

    return _blend_with_plain(recv.flatmap(build), specs, n)


CUSTOM_INPUTS: Dict[str, Callable[[List[Any], int], "st.SearchStrategy[tuple]"]] = {
    "builtins.getattr": _getattr_inputs,
    "operator.is_": _aliased_pair,
    "operator.is_not": _aliased_pair,
    # str ops fuzzed almost entirely in their no-op regime without correlation:
    "str.replace": _needle_inputs,
    "str.count": _needle_inputs,
    "str.find": _needle_inputs,
    "str.rfind": _needle_inputs,
    "str.index": _needle_inputs,
    "str.rindex": _needle_inputs,
    "str.ljust": _pad_to_width_inputs,
    "str.rjust": _pad_to_width_inputs,
    "str.center": _pad_to_width_inputs,
    "str.zfill": _pad_to_width_inputs,
    "str.strip": _surrounded_inputs,
    "str.lstrip": _surrounded_inputs,
    "str.rstrip": _surrounded_inputs,
    "base64.b64decode": _roundtrip("base64", "b64encode", "bytes"),
    "base64.standard_b64decode": _roundtrip("base64", "standard_b64encode", "bytes"),
    "base64.urlsafe_b64decode": _roundtrip("base64", "urlsafe_b64encode", "bytes"),
    "base64.b16decode": _roundtrip("base64", "b16encode", "bytes"),
    "base64.b32decode": _roundtrip("base64", "b32encode", "bytes"),
    "base64.b32hexdecode": _roundtrip("base64", "b32hexencode", "bytes"),
    "base64.a85decode": _roundtrip("base64", "a85encode", "bytes"),
    "base64.b85decode": _roundtrip("base64", "b85encode", "bytes"),
    "base64.z85decode": _roundtrip("base64", "z85encode", "bytes"),
    "base64.decodebytes": _roundtrip("base64", "encodebytes", "bytes"),
    "binascii.a2b_hex": _roundtrip("binascii", "b2a_hex", "bytes"),
    "binascii.unhexlify": _roundtrip("binascii", "hexlify", "bytes"),
    "binascii.a2b_base64": _roundtrip("binascii", "b2a_base64", "bytes"),
    "json.loads": _roundtrip("json", "dumps", "jsonable"),
    "ast.literal_eval": _roundtrip("builtins", "repr", "jsonable"),
    "urllib.parse.unquote": _roundtrip("urllib.parse", "quote", "str"),
    "urllib.parse.unquote_plus": _roundtrip("urllib.parse", "quote_plus", "str"),
    "zlib.decompress": _roundtrip("zlib", "compress", "bytes"),
    "gzip.decompress": _roundtrip("gzip", "compress", "bytes"),
    "bz2.decompress": _roundtrip("bz2", "compress", "bytes"),
}


# subscript / index / pop-style ops whose int argument is an INDEX (or, for
# dict/set, a key) -- bias it small so it actually lands in a size-<=5 receiver
# instead of almost always raising IndexError/KeyError on a random huge int.
_INDEX_OPS = {"__getitem__", "__setitem__", "__delitem__", "insert", "pop", "index"}


def _resolve_arg(
    name: str, annstr: str, literals: Sequence[Any], module: str, func: str
) -> Any:
    """A fuzz spec for one arg: ("sampled", values) from a registry/Literal, else
    the resolved type for the default size-based path."""
    override = _FUNC_ARG_STRATS.get((module, func, name))
    if override is not None:
        return ("sampled", tuple(override()))
    if literals:
        return ("sampled", tuple(literals))
    if annstr == "str" and name in _PARAM_STRATS:
        return ("sampled", tuple(_PARAM_STRATS[name]()))
    resolved = _ann(annstr)
    if func in _INDEX_OPS and resolved is int:
        return ("smallint",)
    return resolved


# ---------------------------------------------------------------------------
# module-level (free) functions -- math.sqrt, json.dumps, ...
#
# Same machinery, minus the receiver: synthesize ``module.func(<fuzzed args>)``.
# Arg types come from the module's own typeshed stub (no TypeVar receiver to
# bind, so binds={} -- bare type params fall back via _NAME_MAP).
# ---------------------------------------------------------------------------
_MODULE_FUNCS: Dict[str, Dict[str, List[Any]]] = {}


def _module_funcs(module: str) -> Dict[str, List[Any]]:
    """Lazily map name -> [FunctionDef overloads] for a module's free functions,
    version/platform-resolved, with @overloads grouped and re-exports followed."""
    if module not in _MODULE_FUNCS:
        funcs: Dict[str, List[Any]] = {}
        for name, ni in _stub_names(module).items():
            defs = _funcdefs(ni)
            if defs:
                funcs[name] = defs
        _MODULE_FUNCS[module] = funcs
    return _MODULE_FUNCS[module]


@functools.lru_cache(maxsize=None)
def _func_candidate_sigs(
    module: str, func: str
) -> List[List[Tuple[str, str, Tuple[Any, ...]]]]:
    """Candidate signatures per overload of module.func (see _overload_sigs)."""
    return _overload_sigs(
        _module_funcs(module).get(func, []), {}, module, ("self", "cls")
    )


def _module_classes(module: str) -> List[type]:
    """Public classes DEFINED in ``module`` (typeshed ClassDef present + a runtime
    ``type`` of the same name, instantiable-ish), excluding private names.  Mirrors
    ``_module_funcs``.  For ``builtins`` this yields the core value types (int, str,
    list, ...) alongside the rest (range, slice, the exception classes, ...) --
    catalog() drives them all through the one loop, no separate curated type list."""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return []
    out: List[type] = []
    for name, ni in _stub_names(module).items():
        if name.startswith("_") or not isinstance(ni.ast, _ast.ClassDef):
            continue
        obj = getattr(mod, name, None)
        if isinstance(obj, type) and obj.__name__ == name:
            out.append(obj)
    return sorted(out, key=lambda t: t.__name__)


# ---------------------------------------------------------------------------
# public bridge: a native callable -> concrete, valid argument tuples
# ---------------------------------------------------------------------------
def _sig_for(fn: Any) -> Optional[Tuple[str, Any, str, Any]]:
    """('method', typ, name, sigs) | ('func', module, name, sigs) | None."""
    objcls = getattr(fn, "__objclass__", None)
    if objcls in RECV and getattr(fn, "__name__", None):
        module = getattr(objcls, "__module__", "builtins")
        return (
            "method",
            objcls,
            fn.__name__,
            _candidate_sigs(objcls, fn.__name__, module),
        )
    mod, name = getattr(fn, "__module__", None), getattr(fn, "__name__", None)
    if mod and name:
        return ("func", mod, name, _func_candidate_sigs(mod, name))
    return None


def primary_sig(sigs: Sequence[Any]) -> Any:
    """The candidate overload to drive an op with.  Prefer the first NON-empty
    one: a variadic like ``set.difference_update(*s)`` yields both a zero-arg and
    a one-arg candidate, and the zero-arg form gives no coverage (and trips
    spurious arity differences), so we drive the form that actually passes args."""
    return next((s for s in sigs if s), sigs[0])


def _specs_for(fn: Any) -> Optional[List[Any]]:
    """Per-arg fuzz specs (and, for methods, a leading receiver spec), or None."""
    info = _sig_for(fn)
    if not info or not info[3]:
        return None
    kind = info[0]
    sig = primary_sig(info[3])
    if kind == "method":
        typ, name = info[1], info[2]
        module = getattr(typ, "__module__", "builtins")
        return [_ann(RECV[typ][0])] + [
            _resolve_arg(n, ann, lits, module, name) for n, ann, lits in sig
        ]
    mod, name = info[1], info[2]
    return [_resolve_arg(n, ann, lits, mod, name) for n, ann, lits in sig]


def tuple_strategy(
    seedkey: Optional[str], specs: List[Any], size: int
) -> "st.SearchStrategy[tuple]":
    """The argument-tuple strategy for an op: a :data:`CUSTOM_INPUTS` override when
    one is registered (correlated / aliased / roundtrip inputs), else independent
    per-arg fuzzing.  Shared by both input paths -- valid_inputs and the support
    sweep -- so a custom input reaches both consumers."""
    custom = CUSTOM_INPUTS.get(seedkey) if seedkey else None
    if custom is not None:
        return custom(specs, size)
    return st.tuples(*[_arg_strategy(s, size) for s in specs])


def valid_inputs(
    fn: Any,
    k: int = 5,
    seed: int = 0,
    develop: bool = True,
    size: int = 3,
    seedkey: Optional[str] = None,
) -> List[Tuple[Any, ...]]:
    """Up to ``k`` concrete, valid argument tuples for ``fn`` (a builtin function
    or method descriptor).  Deterministic given ``seed``.  ``develop`` drops the
    first (often degenerate) example for more diverse values.  ``size`` scales the
    generated arguments (string/collection length, etc.) -- sweep it to see where
    an op cliffs.  ``seedkey`` is the op's catalog identity -- pass it to enable a
    :data:`CUSTOM_INPUTS` override (correlated / aliased / roundtrip inputs); a
    caller with only ``fn`` (which can't recover the public seedkey -- ``operator``
    funcs report ``__module__ == "_operator"``) simply omits it.  Returns [] when no
    signature can be resolved."""
    specs = _specs_for(fn)
    if not specs:
        return []
    strat = tuple_strategy(seedkey, specs, size)
    out = []

    @hyp_seed(seed)
    @settings(
        max_examples=k * 5,
        database=None,
        deadline=None,
        suppress_health_check=list(HealthCheck),
    )
    @given(strat)
    def run(t):
        out.append(t)

    try:
        run()
    except Exception:
        pass
    return (out[1 : k + 1] or out[:k]) if develop else out[:k]


# How to *call* one operation -- shared by the differential test and the support
# measurement so both drive an op the same way.  Returns (fn, expr, arg_names,
# eval_globals): ``fn`` for input generation, and ``expr`` an eval-able source
# over ``arg_names`` (receiver named ``a`` for methods) -- operators MUST use
# operator syntax, so we eval rather than call the dunder descriptor.
def op_call(
    typ: type, method: str, module: str = "builtins"
) -> Optional[Tuple[Any, str, List[str], Dict[str, Any]]]:
    """Call spec for a (type, method), or None if not drivable.  ``module`` is the
    type's owning module (``"builtins"`` for the builtin types)."""
    if method in SKIP_DUNDERS:
        return None
    sigs = _candidate_sigs(typ, method, module)
    if not sigs:
        return None
    argnames = [n for n, _, _ in primary_sig(sigs)]
    recv = receiver_name(argnames)
    expr = call_expr(method, argnames, recv)
    if expr is None:  # operator form needs an arg the signature doesn't supply
        return None
    return (getattr(typ, method), expr, [recv] + argnames, {})


def func_call(
    module: str, name: str
) -> Optional[Tuple[Any, str, List[str], Dict[str, Any]]]:
    """Call spec for a module-level free function, or None if not drivable."""
    fn = getattr(importlib.import_module(module), name, None)
    if fn is None:
        return None
    fsigs = _func_candidate_sigs(module, name)
    if not fsigs:
        return None
    argnames = [n for n, _, _ in primary_sig(fsigs)]
    if not argnames:  # nothing to vary
        return None
    return (fn, f"_fn({', '.join(argnames)})", argnames, {"_fn": fn})


def func_surface(module: str) -> List[str]:
    """Public free-function names of ``module`` that the catalog targets:
    typeshed-known, public, present + callable at runtime, and not a type (type
    constructors belong to the type surface, not here)."""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return []
    out = []
    for name in sorted(_module_funcs(module)):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        if obj is None or not callable(obj) or isinstance(obj, type):
            continue
        out.append(name)
    return out


@dataclass
class Operation:
    """One catalogued operation -- a builtin method, a stdlib class method, or a
    module-level free function -- with the classification both consumers read.

    ``call`` is the drive spec ``(fn, expr, arg_names, eval_globals)``, or None
    when the op isn't drivable.  ``seedkey`` is the dotted identity
    (``"<owner>.<name>"``); ``key`` is the rendered form.

    At most one classification field is set: they are checked in this precedence
    order and the first that applies wins.

    * ``skip_reason`` -- statically not drivable: a dunder, no typeshed signature,
      or an operator arity gap.
    * ``out_of_scope`` -- takes an OS-layer handle (a file descriptor, a process
      id, ...) that CrossHair can never model.
    * ``no_inputs`` -- has a signature, but no concrete input tuple can be
      synthesized for it (typically an unconstructable receiver -- a class not in
      :data:`RECV` -- so ``valid_inputs`` is always empty).  Neither consumer can
      exercise it; this marks an input-generation gap rather than a property of the
      op.  Detected statically via :func:`_specs_for` (no live run needed).
    * ``not_value_function`` -- drivable, but its output isn't a deterministic,
      value-comparable function of the inputs (unordered-container ordering, an
      arbitrary popped element, an identity-eq result, reflection), so the forward
      differential can't judge it.  Static cases are in :data:`NOT_VALUE_FUNCTION`;
      runtime nondeterminism is caught at measure time by :func:`is_deterministic`.
    * ``side_effect`` -- the op isn't safe to run forward: it reaches for I/O (an
      auditwall event -- open/subprocess/socket/... -- fires before the effect) OR
      it mutates hidden global interpreter/process state (gc/recursion-limit/cwd/
      env/locale), which the probe can't see but which would corrupt an in-process
      run.  I/O ops are in :data:`SIDE_EFFECT_OVERRIDES` (the rest a live probe
      finds); the state mutators are hand-curated in :data:`GLOBAL_STATE_OVERRIDES`.
    * ``probe_hazard`` -- the concrete probe can't run it: the op blocks (HANG) or
      crashes the interpreter (CRASH).
    """

    key: str
    seedkey: str
    kind: str  # "method" | "func"
    module: str  # owning module ("builtins" for the builtin types)
    owner: str  # type name (methods) or module name (funcs)
    name: str
    call: Optional[Tuple[Any, str, List[str], Dict[str, Any]]]
    skip_reason: Optional[str] = None  # statically not drivable
    out_of_scope: Optional[str] = None  # OS-layer handle; never modelable
    no_inputs: Optional[str] = None  # signature present but no inputs synthesizable
    not_value_function: Optional[str] = None  # output not a comparable value fn
    side_effect: Optional[str] = None  # audit event the concrete op reaches for
    probe_hazard: Optional[str] = None  # op blocks/crashes the concrete probe

    @property
    def drivable(self) -> bool:
        return (
            self.call is not None
            and self.skip_reason is None
            and self.out_of_scope is None
            and self.no_inputs is None
        )

    def inputs(self, k: int = 5, seed: int = 0, size: int = 3) -> List[Tuple[Any, ...]]:
        """Concrete valid argument tuples for this op at the given ``size``."""
        if self.call is None:
            return []
        return valid_inputs(
            self.call[0], k=k, seed=seed, size=size, seedkey=self.seedkey
        )


# Parameter NAMES that denote an OS-layer handle.  typeshed annotates all of these
# as bare ``int`` (``os.read(fd: int, ...)``), so the TYPE carries no signal -- the
# name is the only reliable one, and typeshed is consistent about it.  An op taking
# one is out of scope: a symbolic int can stand in at the Python layer, but the
# syscall behind it needs a REAL handle CrossHair can't conjure.  (``maxfds`` is
# deliberately absent -- it's a COUNT of fds, not a descriptor.  Signal NUMBERS are
# absent too -- a small modelable int, not a handle.)
_OS_HANDLE_PARAMS: Dict[str, str] = {
    "fd": "takes an OS file descriptor",
    "fd_low": "takes an OS file descriptor",
    "fd_high": "takes an OS file descriptor",
    "fd2": "takes an OS file descriptor",
    "pidfd": "takes an OS file descriptor",
    "pid": "takes an OS process id",
    "pgid": "takes an OS process-group id",
    "thread_id": "takes an OS thread id",
}


def _out_of_scope_reason(
    call: Optional[Tuple[Any, str, List[str], Dict[str, Any]]],
) -> Optional[str]:
    """Why this op is fundamentally out of CrossHair's scope, or None.  Currently:
    it takes an OS-layer handle (a file descriptor) we can never model."""
    if call is None:
        return None
    _fn, _expr, argnames, _eg = call
    for name in argnames:
        reason = _OS_HANDLE_PARAMS.get(name)
        if reason is not None:
            return reason
    return None


# Ops whose output the forward differential can't meaningfully compare -- NOT
# because they're undrivable, but because the output isn't a deterministic,
# value-comparable function of the inputs: unordered-container ordering, an
# arbitrary popped element, an identity-eq result, or a reflective/introspective
# operation (name-based attribute access, code eval, namespace peeking).  seedkey
# -> reason.  Both consumers read this off the Operation: the fuzz test SKIPS these
# (no sound symbolic-vs-concrete assertion exists), and the support sweep still
# grades their inversion but suppresses the black (forward-soundness) check.  This
# is the static half of "not a pure value function"; the dynamic half is caught at
# measure time by :func:`is_deterministic` (which flags time/random/hash-ordering).
NOT_VALUE_FUNCTION: Dict[str, str] = {
    # repr/str of an unordered container -- element order in the string is arbitrary
    "set.__repr__": "set repr element order is unspecified",
    "frozenset.__repr__": "frozenset repr element order is unspecified",
    "dict.__repr__": "dict repr order need not match between symbolic and concrete",
    "set.__str__": "set str element order is unspecified",
    "frozenset.__str__": "frozenset str element order is unspecified",
    "dict.__str__": "dict str order need not match between symbolic and concrete",
    # pop an arbitrary element -- which one is unspecified
    "set.pop": "set.pop removes an arbitrary, unspecified element",
    "dict.popitem": "dict.popitem removes an arbitrary, unspecified item",
    # not value-comparable
    "dict.values": "dict_values has identity equality; not value-comparable",
    # builtins that aren't pure value transforms (identity / code / names)
    "builtins.id": "identity, not a value",
    "builtins.__import__": "imports a module; not a value transform",
    "builtins.__lazy_import__": "[3.15+] lazily imports a module; not a value transform",
    "builtins.compile": "compiles source; output isn't value-comparable",
    "builtins.exec": "executes code for side effects",
    "builtins.eval": "evaluates a symbolic code string -- not meaningful",
    # (builtins.getattr is deliberately ABSENT: given a real attribute name it IS
    # a deterministic value function, and CUSTOM_INPUTS["builtins.getattr"] draws
    # exactly that -- so we fuzz-test it rather than skip it.)
    "builtins.setattr": "mutates an attribute by name",
    "builtins.delattr": "deletes an attribute by name",
    "builtins.hasattr": "attribute probe by name",
    "builtins.globals": "introspects the caller's namespace",
    "builtins.locals": "introspects the caller's namespace",
    "builtins.vars": "introspects an object's namespace",
    "builtins.dir": "introspection; order/content not a pure value",
    # stdlib ops that aren't value functions (surfaced by the differential fuzz):
    # dict-subclass popitem removes an arbitrary item (like dict.popitem);
    "collections.Counter.popitem": "Counter.popitem removes an arbitrary item",
    "collections.defaultdict.popitem": "defaultdict.popitem removes an arbitrary item",
    # runtime/identity/implementation-dependent introspection:
    "gc.is_tracked": "GC-tracking state is implementation-dependent",
    "gc.get_referents": "GC referents are runtime/identity-dependent",
    "gc.get_referrers": "GC referrers are runtime/identity-dependent",
    "sys.getrefcount": "refcount is runtime-dependent",
    "sys.getsizeof": "object size is implementation-dependent",
    "inspect.getmodule": "introspection; returns a runtime module object",
    "inspect.getdoc": "introspection; runtime docstring",
    "weakref.proxy": "returns a proxy with identity semantics",
    # environment-dependent reads (system config / MIME db):
    "os.confstr": "reads system configuration (environment-dependent)",
    "os.sysconf": "reads system configuration (environment-dependent)",
    "posix.confstr": "reads system configuration (environment-dependent)",
    "posix.sysconf": "reads system configuration (environment-dependent)",
    "mimetypes.guess_extension": "reads the system MIME database (environment-dependent)",
    "mimetypes.guess_all_extensions": "reads the system MIME database (environment-dependent)",
    # return a lazy iterator / generator -- not value-comparable:
    "csv.reader": "returns a reader object; identity, not value-comparable",
    "re.finditer": "returns an iterator; not value-comparable",
    "heapq.merge": "returns a lazy iterator; not value-comparable",
    "difflib.context_diff": "returns a lazy generator; not value-comparable",
    "difflib.unified_diff": "returns a lazy generator; not value-comparable",
    "codecs.iterencode": "returns a lazy generator; not value-comparable",
    # nondeterministic: draws from os.urandom, so concrete != symbolic by design:
    "secrets.randbelow": "returns a cryptographically-random int; nondeterministic",
}

# Ops statically known to reach for I/O, hardcoded so ``probe=False`` (the fast
# default both consumers use) classifies them without a live probe and neither runs
# them concretely.  (The concrete sweep otherwise fuzzes an op FORWARD; ``chmod`` /
# ``rename`` / ``subprocess.run`` on fuzzed args would touch the real machine.)  On
# the exclusion-model surface (every documented module -- see catalog()) this must
# name EVERY drivable I/O op the auditwall would flag; the set was enumerated by an
# ``probe="isolated"`` sweep and is guarded by test_uncategorized_ops_probe_cleanly.
# seedkey -> reason.  Re-run the sweep after a Python/typeshed bump to catch new ops.
SIDE_EFFECT_OVERRIDES: Dict[str, str] = {
    "builtins.open": "opens a file (I/O)",
    "builtins.print": "writes output (I/O)",
    "ftplib.ftpcp": "network I/O",
    "glob.glob": "reads the filesystem (I/O)",
    "glob.iglob": "reads the filesystem (I/O)",
    "os.chmod": "mutates the filesystem (I/O)",
    "os.chown": "mutates the filesystem (I/O)",
    "os.execvp": "spawns a process (I/O)",
    "os.lchown": "mutates the filesystem (I/O)",
    "os.link": "mutates the filesystem (I/O)",
    "os.makedirs": "mutates the filesystem (I/O)",
    "os.mkdir": "mutates the filesystem (I/O)",
    "os.popen": "spawns a process (I/O)",
    "os.remove": "mutates the filesystem (I/O)",
    "os.removedirs": "mutates the filesystem (I/O)",
    "os.removexattr": "mutates the filesystem (I/O)",
    "os.rename": "mutates the filesystem (I/O)",
    "os.renames": "mutates the filesystem (I/O)",
    "os.replace": "mutates the filesystem (I/O)",
    "os.rmdir": "mutates the filesystem (I/O)",
    "os.setxattr": "mutates the filesystem (I/O)",
    "os.spawnl": "spawns a process (I/O)",
    "os.spawnlp": "spawns a process (I/O)",
    "os.spawnv": "spawns a process (I/O)",
    "os.spawnvp": "spawns a process (I/O)",
    "os.symlink": "mutates the filesystem (I/O)",
    "os.system": "spawns a process (I/O)",
    "os.truncate": "mutates the filesystem (I/O)",
    "os.unlink": "mutates the filesystem (I/O)",
    "os.utime": "mutates the filesystem (I/O)",
    "posix.chmod": "mutates the filesystem (I/O)",
    "posix.chown": "mutates the filesystem (I/O)",
    "posix.lchown": "mutates the filesystem (I/O)",
    "posix.link": "mutates the filesystem (I/O)",
    "posix.mkdir": "mutates the filesystem (I/O)",
    "posix.remove": "mutates the filesystem (I/O)",
    "posix.removexattr": "mutates the filesystem (I/O)",
    "posix.rename": "mutates the filesystem (I/O)",
    "posix.replace": "mutates the filesystem (I/O)",
    "posix.rmdir": "mutates the filesystem (I/O)",
    "posix.setxattr": "mutates the filesystem (I/O)",
    "posix.symlink": "mutates the filesystem (I/O)",
    "posix.system": "spawns a process (I/O)",
    "posix.truncate": "mutates the filesystem (I/O)",
    "posix.unlink": "mutates the filesystem (I/O)",
    "posix.utime": "mutates the filesystem (I/O)",
    "pty.slave_open": "spawns a process / opens a pty (I/O)",
    "pty.spawn": "spawns a process / opens a pty (I/O)",
    "shutil.chown": "filesystem I/O",
    "shutil.copymode": "filesystem I/O",
    "shutil.copystat": "filesystem I/O",
    "shutil.make_archive": "filesystem I/O",
    "shutil.unpack_archive": "filesystem I/O",
    "socket.create_connection": "network I/O",
    "socket.getaddrinfo": "network I/O",
    "socket.gethostbyaddr": "network I/O",
    "socket.getnameinfo": "network I/O",
    "socket.getservbyname": "network I/O",
    "socket.getservbyport": "network I/O",
    "socket.send_fds": "network I/O",
    "socket.sethostname": "network I/O",
    "sqlite3.connect": "opens a database file (I/O)",
    "ssl.get_server_certificate": "network I/O",
    "subprocess.call": "spawns a process (I/O)",
    "subprocess.check_call": "spawns a process (I/O)",
    "subprocess.check_output": "spawns a process (I/O)",
    "subprocess.getoutput": "spawns a process (I/O)",
    "subprocess.getstatusoutput": "spawns a process (I/O)",
    "subprocess.run": "spawns a process (I/O)",
    "venv.create": "writes a virtual environment (I/O)",
    "webbrowser.open": "launches a web browser (I/O)",
    "webbrowser.open_new": "launches a web browser (I/O)",
    "webbrowser.open_new_tab": "launches a web browser (I/O)",
    # --- platform-specific entrypoints, hand-classified (the Linux isolated sweep
    # can't see them; they're dormant here -- the module/func simply isn't present
    # -- and activate on the platform that ships them).  Conservative: anything that
    # touches the registry / console / audio / filesystem is named, so a per-platform
    # CI pass classifies them without ever running them for real. ---
    # Windows registry (winreg) -- read OR write both hit the registry:
    "winreg.CloseKey": "Windows registry access (I/O)",
    "winreg.ConnectRegistry": "Windows registry access (I/O)",
    "winreg.CreateKey": "Windows registry access (I/O)",
    "winreg.CreateKeyEx": "Windows registry access (I/O)",
    "winreg.DeleteKey": "Windows registry access (I/O)",
    "winreg.DeleteKeyEx": "Windows registry access (I/O)",
    "winreg.DeleteValue": "Windows registry access (I/O)",
    "winreg.DisableReflectionKey": "Windows registry access (I/O)",
    "winreg.EnableReflectionKey": "Windows registry access (I/O)",
    "winreg.EnumKey": "Windows registry access (I/O)",
    "winreg.EnumValue": "Windows registry access (I/O)",
    "winreg.FlushKey": "Windows registry access (I/O)",
    "winreg.LoadKey": "Windows registry access (I/O)",
    "winreg.OpenKey": "Windows registry access (I/O)",
    "winreg.OpenKeyEx": "Windows registry access (I/O)",
    "winreg.QueryInfoKey": "Windows registry access (I/O)",
    "winreg.QueryReflectionKey": "Windows registry access (I/O)",
    "winreg.QueryValue": "Windows registry access (I/O)",
    "winreg.QueryValueEx": "Windows registry access (I/O)",
    "winreg.SaveKey": "Windows registry access (I/O)",
    "winreg.SetValue": "Windows registry access (I/O)",
    "winreg.SetValueEx": "Windows registry access (I/O)",
    # (winreg.ExpandEnvironmentStrings is pure env expansion -- deliberately absent.)
    # Windows sound (winsound) -- drives the audio device:
    "winsound.Beep": "plays a sound (I/O)",
    "winsound.MessageBeep": "plays a sound (I/O)",
    "winsound.PlaySound": "plays a sound (I/O)",
    # Windows console writes (msvcrt); the blocking console READS are hazards below:
    "msvcrt.putch": "writes the console (I/O)",
    "msvcrt.putwch": "writes the console (I/O)",
    "msvcrt.ungetch": "writes the console (I/O)",
    "msvcrt.ungetwch": "writes the console (I/O)",
    "socket.fromshare": "reconstructs a socket (I/O)",
    # os entrypoints present only off Linux (Windows / macOS / BSD):
    "os.startfile": "launches a file (I/O)",  # win32
    "os.add_dll_directory": "modifies the DLL search path (I/O)",  # win32
    "os.listdrives": "queries OS storage (I/O)",  # win32
    "os.listmounts": "queries OS storage (I/O)",  # win32
    "os.listvolumes": "queries OS storage (I/O)",  # win32
    "os.lchmod": "mutates the filesystem (I/O)",  # win32 / darwin / BSD
    "os.chflags": "mutates the filesystem (I/O)",  # darwin / BSD
    "os.lchflags": "mutates the filesystem (I/O)",  # darwin / BSD
    # --- process exec / credential / node-creation ops the I/O sweep missed: a
    # fuzzed arg errors BEFORE the audit event fires, so the probe reads clean, but
    # with valid args they replace the process / drop privileges / touch the fs. ---
    "os.execl": "replaces the process image (exec)",
    "os.execle": "replaces the process image (exec)",
    "os.execlp": "replaces the process image (exec)",
    "os.execlpe": "replaces the process image (exec)",
    "os.spawnle": "spawns a process (I/O)",
    "os.spawnlpe": "spawns a process (I/O)",
    "os.setuid": "changes process credentials",
    "os.setgid": "changes process credentials",
    "os.seteuid": "changes process credentials",
    "os.setegid": "changes process credentials",
    "os.setreuid": "changes process credentials",
    "os.setregid": "changes process credentials",
    "os.setresuid": "changes process credentials",
    "os.setresgid": "changes process credentials",
    "os.setgroups": "changes process credentials",
    "os.initgroups": "changes process credentials",
    "os.unshare": "unshares OS namespaces",
    "os.mkfifo": "creates a filesystem node (I/O)",
    "os.mknod": "creates a filesystem node (I/O)",
    # filesystem reads surfaced by the STRICT probe wall (the default wall allows
    # these for the importer; they read the fs, so they're not value functions):
    "os.listdir": "reads the filesystem (I/O)",
    "os.scandir": "reads the filesystem (I/O)",
    "os.getxattr": "reads the filesystem (I/O)",
    "posix.listdir": "reads the filesystem (I/O)",
    "posix.scandir": "reads the filesystem (I/O)",
    "posix.getxattr": "reads the filesystem (I/O)",
    "glob.glob1": "reads the filesystem (I/O)",
    "socket.gethostbyname": "network I/O",
    "socket.gethostbyname_ex": "network I/O",
    "syslog.syslog": "writes to the system log (I/O)",
    # import/execute-a-module ops (surfaced by the differential fuzz):
    "pydoc.doc": "imports/executes a module to document it (I/O)",
    "pydoc.render_doc": "imports/executes a module to document it (I/O)",
    "pydoc.locate": "imports a module by name (I/O)",
    "pydoc.resolve": "imports a module by name (I/O)",
    "pydoc.safeimport": "imports a module by name (I/O)",
    "pydoc.writedoc": "imports a module and writes an HTML file (I/O)",
    "pkgutil.find_loader": "imports/queries a module (I/O)",
    "pkgutil.get_loader": "imports/queries a module (I/O)",
    "pkgutil.get_data": "reads package data (I/O)",
    "doctest.testsource": "imports a module to extract its doctests (I/O)",
    "posix.setgroups": "changes process credentials",
    "time.clock_settime_ns": "sets the system clock",
    "curses.mousemask": "curses terminal I/O",
    "curses.ungetmouse": "curses terminal I/O",
    # logging emitters write a record to the handlers (stderr by default):
    "logging.critical": "writes a log record (I/O)",
    "logging.error": "writes a log record (I/O)",
    "logging.warning": "writes a log record (I/O)",
    "logging.warn": "writes a log record (I/O)",
    "logging.info": "writes a log record (I/O)",
    "logging.debug": "writes a log record (I/O)",
    "logging.log": "writes a log record (I/O)",
    "logging.exception": "writes a log record (I/O)",
    "warnings.warn": "emits a warning (I/O; leaks the message into the warning system)",
    # filesystem scanners + code-executors surfaced by the per-version CI gate (the
    # Linux dev sweep missed these -- inputs/imports differ by environment):
    "compileall.compile_dir": "walks a directory tree and writes .pyc files (I/O)",
    "pyclbr.readmodule": "scans the filesystem to locate/read a module (I/O)",
    "pyclbr.readmodule_ex": "scans the filesystem to locate/read a module (I/O)",
    "site.addsitedir": "scans a site directory and mutates sys.path (I/O + state)",
    "site.addsitepackages": "scans site dirs and mutates sys.path (I/O + state)",
    "doctest.debug_script": "executes example code under sys.settrace (I/O)",
    "doctest.debug_src": "executes example code under sys.settrace (I/O)",
    "doctest.run_docstring_examples": "executes example code under sys.settrace (I/O)",
    "imp.load_dynamic": "dlopen()s and executes a shared library (I/O; <3.12 only)",
    "ossaudiodev.open": "opens the audio device for writing (I/O)",
    "pydoc.pipepager": "pipes text to a pager subprocess (I/O)",
    "pydoc.tempfilepager": "writes text to a temp file and launches a pager (I/O)",
    # --- method forms and posix twins the single-input probe misses.  The live
    # sweep (test_uncategorized_ops_probe_cleanly) probes each op with ONE fuzzed
    # input, which errors before the I/O call fires -- but measure_support's real
    # fuzzer eventually drives a VALID input that reaches it, so these must be
    # named by hand (same reason as the exec/credential block above).  They only
    # differ from already-listed entries by owner: a bound METHOD seedkey
    # (module.Class.method) rather than the module function, or the ``posix``
    # alias of an ``os`` entry. ---
    "venv.EnvBuilder.create": "writes a virtual environment (I/O)",  # method of venv.create
    "posix.mknod": "creates a filesystem node (I/O)",
    "posix.mkfifo": "creates a filesystem node (I/O)",
    "posix.setuid": "changes process credentials",
    "posix.setgid": "changes process credentials",
    "posix.seteuid": "changes process credentials",
    "posix.setegid": "changes process credentials",
    "posix.setreuid": "changes process credentials",
    "posix.setregid": "changes process credentials",
    "posix.setresuid": "changes process credentials",
    "posix.setresgid": "changes process credentials",
    "posix.initgroups": "changes process credentials",
    "posix.setpriority": "changes process scheduling priority",
    "posix.chroot": "changes the process root directory (I/O)",
    "posix.unshare": "unshares OS namespaces",
    "zipapp.create_archive": "writes an application archive (I/O)",
    "pydoc.writedocs": "writes HTML documentation files (I/O)",
    # webbrowser.open* dispatch to a concrete browser class per-platform; each
    # bound method launches a browser, so name every class's open/open_new/open_new_tab.
    "webbrowser.BackgroundBrowser.open": "launches a web browser (I/O)",
    "webbrowser.BackgroundBrowser.open_new": "launches a web browser (I/O)",
    "webbrowser.BackgroundBrowser.open_new_tab": "launches a web browser (I/O)",
    "webbrowser.BaseBrowser.open": "launches a web browser (I/O)",
    "webbrowser.BaseBrowser.open_new": "launches a web browser (I/O)",
    "webbrowser.BaseBrowser.open_new_tab": "launches a web browser (I/O)",
    "webbrowser.Chrome.open": "launches a web browser (I/O)",
    "webbrowser.Chrome.open_new": "launches a web browser (I/O)",
    "webbrowser.Chrome.open_new_tab": "launches a web browser (I/O)",
    "webbrowser.Elinks.open": "launches a web browser (I/O)",
    "webbrowser.Elinks.open_new": "launches a web browser (I/O)",
    "webbrowser.Elinks.open_new_tab": "launches a web browser (I/O)",
    "webbrowser.GenericBrowser.open": "launches a web browser (I/O)",
    "webbrowser.GenericBrowser.open_new": "launches a web browser (I/O)",
    "webbrowser.GenericBrowser.open_new_tab": "launches a web browser (I/O)",
    "webbrowser.Konqueror.open": "launches a web browser (I/O)",
    "webbrowser.Konqueror.open_new": "launches a web browser (I/O)",
    "webbrowser.Konqueror.open_new_tab": "launches a web browser (I/O)",
    "webbrowser.Mozilla.open": "launches a web browser (I/O)",
    "webbrowser.Mozilla.open_new": "launches a web browser (I/O)",
    "webbrowser.Mozilla.open_new_tab": "launches a web browser (I/O)",
    "webbrowser.Opera.open": "launches a web browser (I/O)",
    "webbrowser.Opera.open_new": "launches a web browser (I/O)",
    "webbrowser.Opera.open_new_tab": "launches a web browser (I/O)",
    "webbrowser.UnixBrowser.open": "launches a web browser (I/O)",
    "webbrowser.UnixBrowser.open_new": "launches a web browser (I/O)",
    "webbrowser.UnixBrowser.open_new_tab": "launches a web browser (I/O)",
    # logging emitters have module-function forms (logging.info, ...) already
    # listed; the bound methods on Logger/LoggerAdapter/RootLogger write too.
    "logging.Logger.critical": "writes a log record (I/O)",
    "logging.Logger.debug": "writes a log record (I/O)",
    "logging.Logger.error": "writes a log record (I/O)",
    "logging.Logger.exception": "writes a log record (I/O)",
    "logging.Logger.info": "writes a log record (I/O)",
    "logging.Logger.log": "writes a log record (I/O)",
    "logging.Logger.warn": "writes a log record (I/O)",
    "logging.Logger.warning": "writes a log record (I/O)",
    "logging.LoggerAdapter.critical": "writes a log record (I/O)",
    "logging.LoggerAdapter.debug": "writes a log record (I/O)",
    "logging.LoggerAdapter.error": "writes a log record (I/O)",
    "logging.LoggerAdapter.exception": "writes a log record (I/O)",
    "logging.LoggerAdapter.info": "writes a log record (I/O)",
    "logging.LoggerAdapter.log": "writes a log record (I/O)",
    "logging.LoggerAdapter.warn": "writes a log record (I/O)",
    "logging.LoggerAdapter.warning": "writes a log record (I/O)",
    "logging.RootLogger.critical": "writes a log record (I/O)",
    "logging.RootLogger.debug": "writes a log record (I/O)",
    "logging.RootLogger.error": "writes a log record (I/O)",
    "logging.RootLogger.exception": "writes a log record (I/O)",
    "logging.RootLogger.info": "writes a log record (I/O)",
    "logging.RootLogger.log": "writes a log record (I/O)",
    "logging.RootLogger.warn": "writes a log record (I/O)",
    "logging.RootLogger.warning": "writes a log record (I/O)",
    # exec-arbitrary-code + remaining I/O ops the single-input probe misses (a valid
    # input reaches them in the full fuzz; hand-named like the block above):
    "doctest.DocFileCase.debug": "executes example code under sys.settrace (I/O)",
    "doctest.DocTestCase.debug": "executes example code under sys.settrace (I/O)",
    "doctest.SkipDocTestCase.debug": "executes example code under sys.settrace (I/O)",
    "cProfile.Profile.run": "executes a code string under the profiler (exec)",
    "profile.Profile.run": "executes a code string under the profiler (exec)",
    "gettext.bindtextdomain": "binds a message-catalog directory (I/O + global state)",
    "os.copy_file_range": "copies between file descriptors (I/O)",
    "posix.copy_file_range": "copies between file descriptors (I/O)",
}

# Ops that MUTATE GLOBAL INTERPRETER / PROCESS STATE -- what the side-effect probe
# is blind to (it catches I/O audit events + hang/crash, but a hidden state change
# raises no event and returns cleanly).  Run forward in-process (as the differential
# fuzz test does), they CORRUPT every later test: gc.set_debug spews debug output,
# sys.setrecursionlimit->1 breaks everything, os.chdir/putenv move the cwd/env,
# locale.setlocale reshapes formatting.  Hand-curated (the probe can't discover
# them); classified into the ``side_effect`` field so both consumers skip them.
# seedkey -> reason.  NOTE: the audit-EVENTFUL mutators here (os.chdir / putenv /
# unsetenv) ARE caught by the strict probe (see :data:`PROBE_REJECT_EVENTS`), so the
# guard test would flag them if missing -- but the static ``probe=False`` path used
# by both consumers still needs the hardcoded entry, so they are named here too.
GLOBAL_STATE_OVERRIDES: Dict[str, str] = {
    "gc.set_debug": "mutates global GC debug state",
    "gc.set_threshold": "mutates global GC thresholds",
    "sys.setrecursionlimit": "mutates the interpreter recursion limit",
    "sys.setswitchinterval": "mutates the interpreter switch interval",
    "sys.set_int_max_str_digits": "mutates the int<->str conversion limit",
    "sys.setdlopenflags": "mutates global dlopen flags",
    "sys.set_coroutine_origin_tracking_depth": "mutates coroutine origin tracking",
    "sys.activate_stack_trampoline": "mutates the interpreter stack trampoline",
    "sys.set_lazy_imports": "[3.15+] mutates process-wide lazy-import mode",
    "warnings.filterwarnings": "mutates the global warnings filters",
    "warnings.simplefilter": "mutates the global warnings filters",
    "locale.setlocale": "mutates the global locale",
    "locale.textdomain": "mutates the global gettext domain",
    "locale.bindtextdomain": "mutates the global gettext domain",
    "locale.bind_textdomain_codeset": "mutates the global gettext domain",
    "signal.signal": "installs a global signal handler",
    "signal.pthread_sigmask": "mutates the thread signal mask",
    "faulthandler.dump_traceback_later": "arms a global faulthandler timer",
    "faulthandler.register": "installs a global faulthandler",
    "faulthandler.unregister": "mutates global faulthandler state",
    "readline.add_history": "mutates global readline state",
    "readline.append_history_file": "mutates global readline state",
    "readline.insert_text": "mutates global readline state",
    "readline.parse_and_bind": "mutates global readline state",
    "readline.remove_history_item": "mutates global readline state",
    "readline.replace_history_item": "mutates global readline state",
    "readline.set_auto_history": "mutates global readline state",
    "readline.set_completer_delims": "mutates global readline state",
    "readline.set_history_length": "mutates global readline state",
    "os.chdir": "changes the working directory (global process state)",
    "posix.chdir": "changes the working directory (global process state)",
    "os.chroot": "changes the process root directory",
    "os.putenv": "mutates the process environment",
    "posix.putenv": "mutates the process environment",
    "os.unsetenv": "mutates the process environment",
    "posix.unsetenv": "mutates the process environment",
    "os.umask": "mutates the process umask",
    "syslog.setlogmask": "mutates the syslog priority mask",
    "os.nice": "changes the process priority",
    "posix.nice": "changes the process priority",
    "posix.umask": "mutates the process umask",
    "os.setpriority": "changes the process priority",
    "ctypes.set_errno": "mutates the thread errno",
    "logging.addLevelName": "mutates the global logging level registry",
    "logging.captureWarnings": "mutates global warnings->logging redirection",
    # stdlib registries the differential fuzz caught being mutated:
    "csv.register_dialect": "mutates the global CSV dialect registry",
    "csv.unregister_dialect": "mutates the global CSV dialect registry",
    "doctest.register_optionflag": "mutates doctest's global option-flag registry",
    "mimetypes.add_type": "mutates the global MIME-types registry",
    "modulefinder.AddPackagePath": "mutates modulefinder's global package-path table",
    "modulefinder.ReplacePackage": "mutates modulefinder's global replace-package map",
}


def is_deterministic(
    forward: Any, args: Sequence[Any], result: Any, mut: bool = False
) -> bool:
    """Is this op's output a function of its inputs -- same input, same output?

    Re-runs ``forward`` on a deepcopy of ``args`` and compares.  If it differs
    (hash with a per-process seed, set/dict-ordering-derived values, time / random
    / id) the output isn't a function of the inputs, so any inversion verdict is
    meaningless.  ``mut``: the op mutates its receiver in place, so compare the
    post-call receiver (``args[0]``) rather than the return value.  A flake on
    recompute returns True -- don't penalize an op we couldn't re-measure.  This is
    the DYNAMIC counterpart to :data:`NOT_VALUE_FUNCTION` (which hardcodes the
    statically-known cases)."""
    try:
        args_copy = copy.deepcopy(tuple(args))
        r = forward(*args_copy)
        recomputed = args_copy[0] if mut else r
        return bool(result == recomputed)
    except Exception:
        return True


# Ops the concrete probe can't RUN -- they block or crash without the auditwall
# catching them first, so their nature is hardcoded rather than observed.  seedkey
# -> the probe_hazard reason.  Everything NOT listed (and not out_of_scope) is safe
# to probe in-process; a ``probe="isolated"`` pass surfaces a new hazard as
# HANG/CRASH -- re-run one after a Python/typeshed bump to catch new ones.
PROBE_HAZARD_OVERRIDES: Dict[str, str] = {
    # file-opening ops: block on a fuzzed path (the wall's ``open`` check only
    # blocks WRITE flags, so a read-mode open of a real path proceeds and hangs).
    "os.open": "opens a file (blocks the probe)",
    "gzip.open": "opens a file (blocks the probe)",
    "bz2.open": "opens a file (blocks the probe)",
    "lzma.open": "opens a file (blocks the probe)",
    "codecs.open": "opens a file (blocks the probe)",
    "tokenize.open": "opens a file (blocks the probe)",
    "mimetypes.read_mime_types": "opens a file (blocks the probe)",
    "signal.sigwait": "blocks waiting for a signal",
    "builtins.breakpoint": "enters the debugger (blocks on debugger input)",
    # more file/exec blockers surfaced by the exclusion-model isolated sweep
    "dbm.open": "opens/reads a file (blocks the probe)",
    "dbm.whichdb": "opens/reads a file (blocks the probe)",
    "io.open_code": "opens/reads a file (blocks the probe)",
    "linecache.getline": "opens/reads a file (blocks the probe)",
    "linecache.getlines": "opens/reads a file (blocks the probe)",
    "linecache.updatecache": "opens/reads a file (blocks the probe)",
    "pdb.find_function": "opens/reads a file (blocks the probe)",
    "posix.open": "opens/reads a file (blocks the probe)",
    "py_compile.compile": "opens/reads a file (blocks the probe)",
    "pydoc.apropos": "imports/executes a module (blocks the probe)",
    "pydoc.importfile": "imports/executes a module (blocks the probe)",
    "pydoc.synopsis": "imports/executes a module (blocks the probe)",
    "runpy.run_path": "imports/executes a module (blocks the probe)",
    "shelve.open": "opens/reads a file (blocks the probe)",
    "tabnanny.check": "opens/reads a file (blocks the probe)",
    "tarfile.is_tarfile": "opens/reads a file (blocks the probe)",
    "wave.open": "opens/reads a file (blocks the probe)",
    "sndhdr.what": "opens/reads a file (blocks the probe)",
    "sndhdr.whathdr": "opens/reads a file (blocks the probe)",
    "sunau.open": "opens/reads a file (blocks the probe)",
    "uu.decode": "opens/reads a file (blocks the probe)",
    "uu.encode": "opens/reads a file (blocks the probe)",
    "zipapp.get_interpreter": "opens/reads a file (blocks the probe)",
    "zipfile.is_zipfile": "opens/reads a file (blocks the probe)",
    # Removed-module blockers surfaced only on the <=3.12 surface (aifc/imghdr are
    # PEP 594 dead batteries gone in 3.13; imp was removed in 3.12).  The isolated
    # classification sweep ran on 3.13+, where these no longer exist to be found, so
    # they slipped the table; the keys are inert on newer versions (no matching op).
    "aifc.open": "opens/reads a file (blocks the probe)",
    "imghdr.what": "opens/reads a file (blocks the probe)",
    "imp.load_compiled": "imports/executes a module (blocks the probe)",
    "imp.load_source": "imports/executes a module (blocks the probe)",
    # ctypes pointer-deref ops: a fuzzed int is read as an address -> segfault
    "ctypes.string_at": "dereferences an arbitrary address (crashes the probe)",
    "ctypes.wstring_at": "dereferences an arbitrary address (crashes the probe)",
    "ctypes.memoryview_at": "dereferences an arbitrary address (crashes the probe)",
    # Windows console reads (msvcrt) block on real console input (they read CONIN$,
    # not the redirected sys.stdin) -- hand-classified; dormant off Windows.
    "msvcrt.getch": "blocks on console input",
    "msvcrt.getche": "blocks on console input",
    "msvcrt.getwch": "blocks on console input",
    "msvcrt.getwche": "blocks on console input",
    # debugger / arbitrary-code-execution entrypoints: run fuzzed code, which can
    # enter an interactive debugger (blocks on input) or execute an arbitrary module.
    "sys.breakpointhook": "enters the debugger (blocks on debugger input)",
    "pdb.run": "runs code under the debugger (may block on input)",
    "pdb.runeval": "runs code under the debugger (may block on input)",
    "cProfile.run": "executes arbitrary code (may block)",
    "profile.run": "executes arbitrary code (may block)",
    "runpy.run_module": "imports and executes a module (may block / side effects)",
    "doctest.debug": "runs code under the debugger (may block)",
    "time.sleep": "blocks for the argument's duration",
    # combine two checksums over a fuzzed length; work is linear in the length's
    # bit-count, so a huge fuzzed int spins for a very long time (new in 3.15).
    "zlib.adler32_combine": "hangs on a huge fuzzed length argument (blocks the probe)",
    "zlib.crc32_combine": "hangs on a huge fuzzed length argument (blocks the probe)",
}

# Audit events the DEFAULT auditwall allows as an analysis convenience (the importer
# walks the fs; test clients open sockets) but that the classification PROBE wants to
# catch -- an isolated stdlib op reaching for any of these IS reaching for an effect.
# Passed to ``enabled_auditwall(reject_prefixes=...)`` so the strict wall lives here,
# with the rest of the classification policy, rather than in auditwall.py.  Each
# raises its audit event unconditionally (before touching the fs), so detection is
# deterministic; the benign ctypes buffer/errno events are deliberately NOT listed.
PROBE_REJECT_EVENTS: Tuple[str, ...] = (
    "os.chdir",
    "os.putenv",
    "os.unsetenv",
    # NB: os.listdir / os.scandir are deliberately NOT rejected here -- the import
    # machinery raises them (scanning ``sys.path``) the first time an op lazily
    # imports a submodule (e.g. time.strptime pulling in _strptime), which would
    # be a false positive.  The stdlib ops that genuinely scan the filesystem
    # (compileall.compile_dir, pyclbr.readmodule*, site.addsitedir*) are instead
    # classified by hand in SIDE_EFFECT_OVERRIDES.
    "os.walk",
    "os.fwalk",
    "os.getxattr",
    "os.listxattr",
    "glob.glob",
    "pathlib.Path.glob",
    "socket.gethostbyname",
    "socket.__new__",
    "socket.bind",
    "socket.connect",
    "sys.settrace",
    "sys.setprofile",
    "ctypes.dlopen",  # native FFI: loads/calls arbitrary code
    "ctypes.dlsym",
    "ctypes.call_function",
    "ctypes.cdata",
    "syslog.syslog",  # system logger: writes / reconfigures
    "syslog.openlog",
    "syslog.closelog",
    "syslog.setlogmask",
)

# probe_hazard values for an op whose concrete probe misbehaved -- each a signal to
# add the op to PROBE_HAZARD_OVERRIDES with a hand-written classification.  HANG:
# still running at the timeout (a blocking op).  CRASH: the child died without a
# result (a segfault / abort -- e.g. a C call handed a bogus pointer-shaped int).
HANG = "unprobeable: concrete probe did not terminate"
CRASH = "unprobeable: concrete probe crashed the interpreter"


def probe_side_effect(
    call: Optional[Tuple[Any, str, List[str], Dict[str, Any]]],
    seedkey: Optional[str] = None,
    k: int = 3,
    seed: int = 0,
    size: int = 1,
) -> Optional[str]:
    """Run the op CONCRETELY on a few valid inputs with the auditwall engaged; if
    it reaches for a blocked side effect (I/O, subprocess, socket, ...) return the
    audit event message, else None.  A ``seedkey`` in :data:`PROBE_HAZARD_OVERRIDES`
    short-circuits to its hardcoded reason WITHOUT running the op.

    The wall raises BEFORE the effect happens, so this detects I/O without
    performing it (no junk files).  std streams are redirected so an op that reads
    stdin (``input``) or writes stdout (``print``) can't block or spew.  This is an
    in-process probe: safe on a vetted surface, but an op that BLOCKS (and isn't
    overridden) will wedge the caller -- use :func:`probe_side_effect_isolated` for
    an unvetted surface."""
    if call is None:
        return None
    if seedkey is not None and seedkey in PROBE_HAZARD_OVERRIDES:
        return PROBE_HAZARD_OVERRIDES[seedkey]
    fn, expr, argnames, eval_globals = call
    inputs = valid_inputs(fn, k=k, seed=seed, size=size)
    if not inputs:
        return None
    for vals in inputs:
        if len(vals) != len(argnames):
            continue
        try:
            with enabled_auditwall(
                reject_prefixes=PROBE_REJECT_EVENTS
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                stdin, sys.stdin = sys.stdin, io.StringIO()
                try:
                    eval(expr, dict(eval_globals), dict(zip(argnames, vals)))
                finally:
                    sys.stdin = stdin
        except SideEffectDetected as exc:
            return str(exc)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            continue  # the op's OWN error isn't a side effect; try another input
    return None


def _probe_child(call: Any, seedkey: Optional[str], q: Any) -> None:
    try:
        q.put(("ok", probe_side_effect(call, seedkey)))
    except BaseException as exc:  # the probe itself blew up (not the op's own error)
        q.put(("crash", f"{type(exc).__name__}: {exc}"))


def probe_side_effect_isolated(
    call: Optional[Tuple[Any, str, List[str], Dict[str, Any]]],
    seedkey: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[str]:
    """Like :func:`probe_side_effect`, but runs in a KILLABLE ``fork`` subprocess so
    an op that blocks (``time.sleep``, a ``select``/``signal`` wait) the auditwall
    can't catch returns :data:`HANG` after ``timeout`` (or :data:`CRASH` if the
    child dies without a result) instead of wedging the caller.  Use for bulk
    probing of an unvetted surface, or to DISCOVER which ops to add to
    :data:`PROBE_HAZARD_OVERRIDES`.

    Forks, so call it from a clean process -- no active z3/tracing whose inherited
    locks would deadlock the child (the lesson the measurement pool learned)."""
    if call is None:
        return None
    if seedkey is not None and seedkey in PROBE_HAZARD_OVERRIDES:
        return PROBE_HAZARD_OVERRIDES[seedkey]
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_probe_child, args=(call, seedkey, q), daemon=True)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()  # SIGKILL reaches even a native-wedged child
        proc.join()
        return HANG  # still running: a blocking op (sleep / wait / huge loop)
    try:
        status, val = q.get(timeout=1.0)
    except Exception:
        return CRASH  # finished but no result: segfault / abort / os._exit
    return val if status == "ok" else None  # a probe-side crash isn't a side effect


def _probe(
    call: Optional[Tuple[Any, str, List[str], Dict[str, Any]]],
    seedkey: str,
    mode: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """LIVE probe (a no-op unless ``mode`` is truthy).  Returns ``(side_effect,
    probe_hazard)``: a HANG/CRASH from the isolated probe is a hazard, a discovered
    audit event a side_effect.  ``mode``: False (skip), True (in-process), or
    "isolated" (killable subprocess).  Hardcoded PROBE_HAZARD_OVERRIDES are applied
    by :func:`_classify` BEFORE this runs, so they don't need a live probe."""
    if not mode or call is None:
        return (None, None)
    reason = (
        probe_side_effect_isolated(call, seedkey)
        if mode == "isolated"
        else probe_side_effect(call, seedkey)
    )
    if reason in (HANG, CRASH):
        return (None, reason)
    return (reason, None)  # a discovered audit event, or (None, None)


def _classify(
    call: Optional[Tuple[Any, str, List[str], Dict[str, Any]]],
    seedkey: str,
    probe: Any,
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    """The six classification fields for one op, in precedence order: static skip,
    out_of_scope, no_inputs, not_value_function, hardcoded side-effect (I/O override
    or global-state mutator) / hazard, then (only if none apply) a LIVE probe.
    Everything but the live probe is STATIC --
    so ``probe=False`` still yields a complete classification except for
    live-discovered side effects (empty on a curated surface, where the known I/O
    ops are already in SIDE_EFFECT_OVERRIDES); consumers enumerate cheaply without
    probing."""
    if call is None:  # not drivable: a dunder, no typeshed signature, or arity gap
        return ("no drivable signature", None, None, None, None, None)
    oos = _out_of_scope_reason(call)
    if oos is not None:  # never modelable -> don't waste a probe on it
        return (None, oos, None, None, None, None)
    if _specs_for(call[0]) is None:  # signature exists but no inputs synthesizable
        return (
            None,
            None,
            "no synthesizable inputs (unconstructable arguments)",
            None,
            None,
            None,
        )
    nvf = NOT_VALUE_FUNCTION.get(seedkey)
    if nvf is not None:  # drivable, but the forward differential can't judge it
        return (None, None, None, nvf, None, None)
    if seedkey in SIDE_EFFECT_OVERRIDES:  # known I/O -> don't run it concretely
        return (None, None, None, None, SIDE_EFFECT_OVERRIDES[seedkey], None)
    if seedkey in GLOBAL_STATE_OVERRIDES:  # mutates global state -> also don't run
        return (None, None, None, None, GLOBAL_STATE_OVERRIDES[seedkey], None)
    if seedkey in PROBE_HAZARD_OVERRIDES:  # hardcoded; applies at any probe level
        return (None, None, None, None, None, PROBE_HAZARD_OVERRIDES[seedkey])
    side_effect, hazard = _probe(call, seedkey, probe)
    return (None, None, None, None, side_effect, hazard)


def _method_op(module: str, typ: type, meth: str, probe: Any) -> Operation:
    call = op_call(typ, meth, module)
    if module == "builtins":
        key = f"builtins.{typ.__name__}_{meth}_method"
        seedkey = f"{typ.__name__}.{meth}"
    else:
        key = f"{module}.{typ.__name__}_{meth}_method"
        seedkey = f"{module}.{typ.__name__}.{meth}"
    skip, oos, no_inputs, nvf, side_effect, hazard = _classify(call, seedkey, probe)
    return Operation(
        key=key,
        seedkey=seedkey,
        kind="method",
        module=module,
        owner=typ.__name__,
        name=meth,
        call=call,
        skip_reason=skip,
        out_of_scope=oos,
        no_inputs=no_inputs,
        not_value_function=nvf,
        side_effect=side_effect,
        probe_hazard=hazard,
    )


def _func_op(module: str, name: str, probe: Any) -> Operation:
    call = func_call(module, name)
    seedkey = f"{module}.{name}"
    skip, oos, no_inputs, nvf, side_effect, hazard = _classify(call, seedkey, probe)
    return Operation(
        key=seedkey,
        seedkey=seedkey,
        kind="func",
        module=module,
        owner=module,
        name=name,
        call=call,
        skip_reason=skip,
        out_of_scope=oos,
        no_inputs=no_inputs,
        not_value_function=nvf,
        side_effect=side_effect,
        probe_hazard=hazard,
    )


# Top-level stdlib modules DOCUMENTED on the newest supported CPython (3.14), taken
# from the Sphinx module index (``docs.python.org/3.14/objects.inv`` -- the dot-free
# ``py:module`` names).  This is authoritative for "is a module public/documented",
# which is the bar for the catalog surface: it EXCLUDES undocumented internals
# (sre_parse, opcode, nturl2path, genericpath, ...).  It retains a few "tombstoned"
# modules the docs still index after runtime removal (aifc, cgi, distutils, ...);
# those are harmless -- they aren't in ``sys.stdlib_module_names`` on a version that
# dropped them, so any consumer intersecting with the live stdlib skips them.
#
# Regenerate when bumping the newest supported version: fetch the new objects.inv,
# keep the dot-free py:module names, replace the set, and add a delta row for the
# old top (``doc(old) = (doc(new) - removed) | restored``).
_DOCUMENTED_MODULES_LATEST: FrozenSet[str] = frozenset(
    {
        "__future__",
        "__main__",
        "_thread",
        "_tkinter",
        "abc",
        "aifc",
        "annotationlib",
        "argparse",
        "array",
        "ast",
        "asynchat",
        "asyncio",
        "asyncore",
        "atexit",
        "audioop",
        "base64",
        "bdb",
        "binascii",
        "bisect",
        "builtins",
        "bz2",
        "cProfile",
        "calendar",
        "cgi",
        "cgitb",
        "chunk",
        "cmath",
        "cmd",
        "code",
        "codecs",
        "codeop",
        "collections",
        "colorsys",
        "compileall",
        "compression",
        "configparser",
        "contextlib",
        "contextvars",
        "copy",
        "copyreg",
        "crypt",
        "csv",
        "ctypes",
        "curses",
        "dataclasses",
        "datetime",
        "dbm",
        "decimal",
        "difflib",
        "dis",
        "distutils",
        "doctest",
        "email",
        "encodings",
        "ensurepip",
        "enum",
        "errno",
        "faulthandler",
        "fcntl",
        "filecmp",
        "fileinput",
        "fnmatch",
        "fractions",
        "ftplib",
        "functools",
        "gc",
        "getopt",
        "getpass",
        "gettext",
        "glob",
        "graphlib",
        "grp",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "idlelib",
        "imaplib",
        "imghdr",
        "imp",
        "importlib",
        "inspect",
        "io",
        "ipaddress",
        "itertools",
        "json",
        "keyword",
        "linecache",
        "locale",
        "logging",
        "lzma",
        "mailbox",
        "mailcap",
        "marshal",
        "math",
        "mimetypes",
        "mmap",
        "modulefinder",
        "msilib",
        "msvcrt",
        "multiprocessing",
        "netrc",
        "nis",
        "nntplib",
        "numbers",
        "operator",
        "optparse",
        "os",
        "ossaudiodev",
        "pathlib",
        "pdb",
        "pickle",
        "pickletools",
        "pipes",
        "pkgutil",
        "platform",
        "plistlib",
        "poplib",
        "posix",
        "pprint",
        "profile",
        "pstats",
        "pty",
        "pwd",
        "py_compile",
        "pyclbr",
        "pydoc",
        "queue",
        "quopri",
        "random",
        "re",
        "readline",
        "reprlib",
        "resource",
        "rlcompleter",
        "runpy",
        "sched",
        "secrets",
        "select",
        "selectors",
        "shelve",
        "shlex",
        "shutil",
        "signal",
        "site",
        "sitecustomize",
        "smtpd",
        "smtplib",
        "sndhdr",
        "socket",
        "socketserver",
        "spwd",
        "sqlite3",
        "ssl",
        "stat",
        "statistics",
        "string",
        "stringprep",
        "struct",
        "subprocess",
        "sunau",
        "symtable",
        "sys",
        "sysconfig",
        "syslog",
        "tabnanny",
        "tarfile",
        "telnetlib",
        "tempfile",
        "termios",
        "test",
        "textwrap",
        "threading",
        "time",
        "timeit",
        "tkinter",
        "token",
        "tokenize",
        "tomllib",
        "trace",
        "traceback",
        "tracemalloc",
        "tty",
        "turtle",
        "turtledemo",
        "types",
        "typing",
        "unicodedata",
        "unittest",
        "urllib",
        "usercustomize",
        "uu",
        "uuid",
        "venv",
        "warnings",
        "wave",
        "weakref",
        "webbrowser",
        "winreg",
        "winsound",
        "wsgiref",
        "xdrlib",
        "xml",
        "xmlrpc",
        "zipapp",
        "zipfile",
        "zipimport",
        "zlib",
        "zoneinfo",
    }
)

# Backward deltas from the newest baked set, one row per minor-version BOUNDARY,
# newest first: ``(older_minor, restored, removed)`` where doc(older_minor) is
# reconstructed from doc(older_minor + 1) by dropping ``removed`` (modules first
# documented in the newer version) and adding back ``restored`` (documented in the
# older version, gone in the newer).  Applied top-down for every running minor
# <= older_minor, so the chain rebuilds any supported version (down to 3.8).
_DOCUMENTED_MODULE_DELTAS: Tuple[Tuple[int, Tuple[str, ...], Tuple[str, ...]], ...] = (
    (13, (), ("annotationlib", "compression")),
    (12, ("lib2to3",), ("encodings",)),
    (11, (), ("xmlrpc",)),
    (10, ("binhex",), ("_tkinter", "sitecustomize", "tomllib", "usercustomize")),
    (9, ("formatter", "parser", "symbol"), ("idlelib",)),
    (8, ("_dummy_thread", "dummy_threading"), ("graphlib", "zoneinfo")),
)


def documented_stdlib_modules() -> FrozenSet[str]:
    """Top-level stdlib modules documented on the RUNNING CPython, reconstructed
    from :data:`_DOCUMENTED_MODULES_LATEST` by applying the backward deltas for
    every boundary at or above the running minor version.  On a version newer than
    the newest baked one, returns the baked set unchanged (best effort until it is
    regenerated).  This is the "is it public/documented" gate for the catalog
    surface -- undocumented internals are absent by construction.

    Where the runtime exposes it (``sys.stdlib_module_names``, 3.10+), the result
    is intersected with the modules actually PRESENT in this interpreter -- so a
    "tombstoned" module the docs still index after removal (aifc, cgi, distutils,
    ...) drops out on a version that no longer ships it.  Below 3.10 no such list
    exists, so the documented set is returned unfiltered."""
    minor = sys.version_info[1]
    mods = set(_DOCUMENTED_MODULES_LATEST)
    for boundary, restored, removed in _DOCUMENTED_MODULE_DELTAS:
        if minor <= boundary:
            mods.difference_update(removed)
            mods.update(restored)
    present = getattr(sys, "stdlib_module_names", None)
    if present is not None:
        mods &= set(present)
    return frozenset(mods)


# The operation surface is an EXCLUSION model: rather than hand-listing the modules
# we DO cover, we enumerate every documented, present stdlib module
# (:func:`documented_stdlib_modules`) and let PER-OP classification decide what is
# actually usable.  A newly-documented stdlib module joins the surface automatically;
# undocumented internals (sre_parse, opcode, nturl2path, genericpath, ...) never
# appear, because they are absent from the documented set.
#
# CATALOG_MODULE_DENYLIST: documented modules deliberately kept OFF the surface.
# EMPTY today -- the per-op classification (out_of_scope / no_inputs / side_effect /
# probe_hazard / not_value_function), plus import-failure-to-[] for platform modules
# absent at runtime, already neutralizes everything, so no whole-module exclusion is
# justified.  Add a name here only for a reason the per-op classification can't
# express, with that reason in a comment.
CATALOG_MODULE_DENYLIST: FrozenSet[str] = frozenset()

# Documented submodules that carry drivable ops but are not top-level importable
# names (so they are absent from documented_stdlib_modules, which is top-level only
# and gets intersected with sys.stdlib_module_names).  Folded into catalog_modules()
# so they contribute BOTH free functions AND class methods, exactly like a top-level
# module -- no func-only special case (that split is what left urllib.parse's
# SplitResult/ParseResult methods uncatalogued).  Assumed present and importable on
# every runtime, so unlike the documented set they get no live-presence intersection.
# ``os.path`` is the platform path module (it re-exports the genericpath functions);
# its undocumented impl modules posixpath / ntpath / genericpath are not named.
_CATALOG_SUBMODULES: Tuple[str, ...] = ("os.path", "urllib.parse")


def catalog_modules() -> Tuple[str, ...]:
    """Sorted stdlib modules the catalog enumerates: :func:`documented_stdlib_modules`
    plus the curated dotted :data:`_CATALOG_SUBMODULES`, minus
    :data:`CATALOG_MODULE_DENYLIST`.  The single source for both the free-function
    and the class-method surface."""
    return tuple(
        sorted(
            (documented_stdlib_modules() | set(_CATALOG_SUBMODULES))
            - CATALOG_MODULE_DENYLIST
        )
    )


def catalog(*, probe: Any = False) -> Iterator[Operation]:
    """Yield one :class:`Operation` per catalogued op -- the single surface both the
    support map and the differential fuzz test draw from.  ONE uniform loop over
    :func:`catalog_modules` (``builtins`` included, no special case): each module
    contributes its public classes' methods (:func:`_module_classes`, which yields
    the builtin value types int/str/list/... alongside range/slice/exceptions/...)
    and its module-level free functions (:func:`func_surface`).

    ``probe`` selects the LIVE side-effect probe per drivable op (static
    classification -- skip / out_of_scope / hardcoded hazard -- always applies):
      * ``False``      -- no live probe (the safe default; fast, and complete on a
                          pure surface where nothing new is discovered).
      * ``True``       -- in-process (fast; safe on a vetted surface, but a
                          non-overridden blocking op will wedge the caller).
      * ``"isolated"`` -- each op in a killable subprocess (safe on ANY surface;
                          a blocking op is tagged :data:`HANG`).  Slower; forks,
                          so run it from a clean process."""
    for module in catalog_modules():
        for typ in _module_classes(module):
            for meth in surface(typ):
                yield _method_op(module, typ, meth, probe)
        for name in func_surface(module):
            yield _func_op(module, name, probe)
