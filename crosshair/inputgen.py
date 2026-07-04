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
import functools
import importlib
import sys
import typing
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

from hypothesis import HealthCheck, given
from hypothesis import seed as hyp_seed
from hypothesis import settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# the operation surface: builtin types + their methods
# ---------------------------------------------------------------------------
TYPES = [int, float, bool, str, bytes, bytearray, list, tuple, dict, set, frozenset]


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


def surface(typ: type) -> List[str]:
    methods = [
        n
        for n in dir(typ)
        if not n.startswith("__") and callable(getattr(typ, n, None))
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


def call_expr(method: str, argnames: Sequence[str]) -> Optional[str]:
    """The source expression invoking ``method`` on receiver ``a`` with the given
    argument names, or None when an operator form needs an argument the signature
    doesn't supply."""
    if method in _BINOP:
        return f"a {_BINOP[method]} {argnames[0]}" if argnames else None
    if method == "__divmod__":
        return f"divmod(a, {argnames[0]})" if argnames else None
    if method in _UNARY and not argnames:
        return _UNARY[method].format(a="a")
    if method in _CALLOP and not argnames:
        return _CALLOP[method].format(a="a")
    if method == "__contains__":
        return f"{argnames[0]} in a" if argnames else None
    if method == "__getitem__":
        return f"a[{argnames[0]}]" if argnames else None
    return f"a.{method}({', '.join(argnames)})"


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


# receiver annotation + TypeVar bindings.  Every container is mono-element, so we
# bind all of typeshed's element TypeVar names (_T/_T_co/_KT/_VT) -- as used by
# the ABC bases methods are inherited from -- to the receiver's element type.
def _elem(t: str) -> Dict[str, str]:
    return {"_T": t, "_S": t, "_T_co": t, "_KT": t, "_VT": t}


RECV = {
    int: ("int", {}),
    float: ("float", {}),
    bool: ("bool", {}),
    str: ("str", _elem("str")),
    bytes: ("bytes", _elem("int")),
    bytearray: ("bytearray", _elem("int")),
    list: ("List[int]", _elem("int")),
    tuple: ("Tuple[int, ...]", _elem("int")),
    dict: ("Dict[int, int]", _elem("int")),
    set: ("Set[int]", _elem("int")),
    frozenset: ("FrozenSet[int]", _elem("int")),
}


# typeshed leaf names -> a fuzzable annotation.  Beyond the concrete builtins,
# this maps the common stdlib type-vocabulary the probe surfaced: numeric
# protocols (math/statistics), generic element TypeVars (free functions have no
# receiver to bind them, so they default to int -- for builtin methods RECV binds
# win since _map_ann checks `binds` first), and str/buffer families.  Over-broad
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
    "object": "int",
    "Any": "int",
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
    "_HashableT": "int",
    "Hashable": "int",
    "_MultiplicableT1": "int",
    "_MultiplicableT2": "int",
    "_SupportsProdNoDefaultT": "int",
    "_SupportsSumNoDefaultT": "int",
    # generic element TypeVars (default for unbound free-function type params)
    "_T": "int",
    "_S": "int",
    "_U": "int",
    "_T1": "int",
    "_T2": "int",
    "_KT": "int",
    "_VT": "int",
    "_T_co": "int",
    "_T_contra": "int",
    # string / buffer families
    "AnyStr": "str",
    "StrOrLiteralStr": "str",
    "StrPath": "str",
    "_AsciiBuffer": "bytes",
    "WriteableBuffer": "bytes",
}


class _Unsupported(Exception):
    pass


def _map_ann(node: Any, binds: Dict[str, str]) -> str:
    """typeshed annotation AST -> a fuzzable annotation string (or raise)."""
    if isinstance(node, _ast.Name):
        if node.id in binds:
            return binds[node.id]
        if node.id in _NAME_MAP:
            return _NAME_MAP[node.id]
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
        if base in ("Iterable", "Sequence", "Collection", "list", "List"):
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
        if (
            base == "Literal"
        ):  # map to the underlying value's type (e.g. Literal[-1,0,1] -> int)
            for e in elts:
                c = (
                    e.operand if isinstance(e, _ast.UnaryOp) else e
                )  # Literal[-1] -> UnaryOp
                if isinstance(c, _ast.Constant) and c.value is not None:
                    return _map_ann(_ast.Name(id=type(c.value).__name__), binds)
            raise _Unsupported("Literal")
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
_ROUNDTRIP = {
    ("base64", "b64decode"): ("base64", "b64encode", "bytes"),
    ("base64", "standard_b64decode"): ("base64", "standard_b64encode", "bytes"),
    ("base64", "urlsafe_b64decode"): ("base64", "urlsafe_b64encode", "bytes"),
    ("base64", "b16decode"): ("base64", "b16encode", "bytes"),
    ("base64", "b32decode"): ("base64", "b32encode", "bytes"),
    ("base64", "b32hexdecode"): ("base64", "b32hexencode", "bytes"),
    ("base64", "a85decode"): ("base64", "a85encode", "bytes"),
    ("base64", "b85decode"): ("base64", "b85encode", "bytes"),
    ("base64", "z85decode"): ("base64", "z85encode", "bytes"),
    ("base64", "decodebytes"): ("base64", "encodebytes", "bytes"),
    ("binascii", "a2b_hex"): ("binascii", "b2a_hex", "bytes"),
    ("binascii", "unhexlify"): ("binascii", "hexlify", "bytes"),
    ("binascii", "a2b_base64"): ("binascii", "b2a_base64", "bytes"),
    ("json", "loads"): ("json", "dumps", "jsonable"),
    ("ast", "literal_eval"): ("builtins", "repr", "jsonable"),
    ("urllib.parse", "unquote"): ("urllib.parse", "quote", "str"),
    ("urllib.parse", "unquote_plus"): ("urllib.parse", "quote_plus", "str"),
    ("zlib", "decompress"): ("zlib", "compress", "bytes"),
    ("gzip", "decompress"): ("gzip", "compress", "bytes"),
    ("bz2", "decompress"): ("bz2", "compress", "bytes"),
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
    ``type`` of the same name, instantiable-ish), excluding the builtin ``TYPES``
    (covered by the type surface) and private names.  Mirrors ``_module_funcs``."""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return []
    out: List[type] = []
    for name, ni in _stub_names(module).items():
        if name.startswith("_") or not isinstance(ni.ast, _ast.ClassDef):
            continue
        obj = getattr(mod, name, None)
        if isinstance(obj, type) and obj not in TYPES and obj.__name__ == name:
            out.append(obj)
    return sorted(out, key=lambda t: t.__name__)


# ---------------------------------------------------------------------------
# public bridge: a native callable -> concrete, valid argument tuples
# ---------------------------------------------------------------------------
def _sig_for(fn: Any) -> Optional[Tuple[str, Any, str, Any]]:
    """('method', typ, name, sigs) | ('func', module, name, sigs) | None."""
    objcls = getattr(fn, "__objclass__", None)
    if objcls in RECV and getattr(fn, "__name__", None):
        return ("method", objcls, fn.__name__, _candidate_sigs(objcls, fn.__name__))
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
        return [_ann(RECV[typ][0])] + [
            _resolve_arg(n, ann, lits, "builtins", name) for n, ann, lits in sig
        ]
    mod, name = info[1], info[2]
    specs = [_resolve_arg(n, ann, lits, mod, name) for n, ann, lits in sig]
    rt = _ROUNDTRIP.get((mod, name))
    if rt and specs:
        specs[0] = ("roundtrip",) + rt
    return specs


def valid_inputs(
    fn: Any, k: int = 5, seed: int = 0, develop: bool = True
) -> List[Tuple[Any, ...]]:
    """Up to ``k`` concrete, valid argument tuples for ``fn`` (a builtin function
    or method descriptor).  Deterministic given ``seed``.  ``develop`` drops the
    first (often degenerate) example for more diverse values.  Returns [] when no
    signature can be resolved."""
    specs = _specs_for(fn)
    if not specs:
        return []
    strat = st.tuples(*[_arg_strategy(s, 3) for s in specs])
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
    expr = call_expr(method, argnames)
    if expr is None:  # operator form needs an arg the signature doesn't supply
        return None
    return (getattr(typ, method), expr, ["a"] + argnames, {})


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
