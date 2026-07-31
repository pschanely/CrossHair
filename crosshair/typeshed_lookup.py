"""
Typeshed stub resolution: module name -> the function definitions it declares.

Everything here stops at the AST. Callers decide what a typeshed annotation
*means*: :mod:`crosshair.stubs_parser` turns it into a faithful
:class:`inspect.Signature`, while :mod:`crosshair.inputgen` maps it to a
generatable approximation.

Lookups run against a search context pinned to the running interpreter, so
``sys.version_info`` / ``sys.platform`` guards in the stubs are already resolved
and the surface matches what this interpreter actually has. ``@overload`` groups
arrive grouped, and re-exports (``bisect.bisect_left`` <- ``_bisect``) are
followed to where the definition really lives.
"""

import ast
import functools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import typeshed_client as tc  # type: ignore

FunctionDef = Union[ast.FunctionDef, ast.AsyncFunctionDef]
_FUNCTION_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)

# Following re-exports and base classes walks a graph that stubs can make cyclic;
# every recursive resolver is bounded by this.
_MAX_DEPTH = 4

# Fallback modules for a class name that the requested module doesn't declare.
# builtins holds the concrete types and object; typing holds the ABC bases
# (MutableSequence, Mapping, ...) that typeshed names as bases and on which it
# declares inherited methods.
_CLASS_FALLBACK_MODULES = ("builtins", "typing")


@functools.lru_cache(maxsize=1)
def search_context() -> tc.SearchContext:
    """The typeshed search context for the running interpreter: this process's
    ``sys.path``, at its version and platform.

    Passing ``search_path`` matters -- without it ``typeshed_client`` recovers the
    path by spawning ``sys.executable``, which the auditwall rejects.
    """
    return tc.get_search_context(search_path=[Path(p) for p in sys.path if p])


@functools.lru_cache(maxsize=None)
def stub_names(module: str) -> Mapping[str, tc.NameInfo]:
    """The version/platform-resolved ``{name: NameInfo}`` a module's stub declares,
    or ``{}`` when it has no stub."""
    try:
        return tc.get_stub_names(module, search_context=search_context()) or {}
    except Exception:
        return {}


@functools.lru_cache(maxsize=None)
def stub_source(module: str) -> Optional[str]:
    """The text of a module's stub file, or None."""
    try:
        path = tc.get_stub_file(module, search_context=search_context())
    except Exception:
        return None
    if path is None:
        return None
    try:
        return path.read_text()
    except OSError:
        return None


@functools.lru_cache(maxsize=None)
def stub_module_ast(module: str) -> Optional[ast.Module]:
    """The parsed stub module, or None. Useful for the statements ``stub_names``
    discards, e.g. the imports that a name's annotations resolve against."""
    try:
        node = tc.get_stub_ast(module, search_context=search_context())
    except Exception:
        return None
    return node if isinstance(node, ast.Module) else None


def _resolve(
    name_info: Optional[tc.NameInfo], module: str, depth: int
) -> Optional[Tuple[tc.NameInfo, str]]:
    """Follow ``name_info`` across re-exports to the (NameInfo, defining module)
    that actually holds a definition."""
    if name_info is None or depth > _MAX_DEPTH:
        return None
    node = name_info.ast
    if not isinstance(node, tc.ImportedName):
        return (name_info, module)
    target_module = ".".join(node.module_name)
    target_name = cast(str, node.name or name_info.name)
    return _resolve(
        stub_names(target_module).get(target_name), target_module, depth + 1
    )


def resolve_name(name: str, module: str) -> Optional[Tuple[tc.NameInfo, str]]:
    """The (NameInfo, defining module) for a module-level name, re-exports followed."""
    return _resolve(stub_names(module).get(name), module, 0)


def funcdefs(name_info: Optional[tc.NameInfo], module: str) -> List[FunctionDef]:
    """The function definitions behind a NameInfo: a plain ``def``, every member of
    an ``@overload`` group, or whatever a re-export points at. ``[]`` for anything
    that isn't a function."""
    resolved = _resolve(name_info, module, 0)
    if resolved is None:
        return []
    node = resolved[0].ast
    if isinstance(node, _FUNCTION_DEFS):
        return [node]
    if isinstance(node, tc.OverloadedName):
        return [d for d in node.definitions if isinstance(d, _FUNCTION_DEFS)]
    return []


def module_funcdefs(module: str) -> Dict[str, List[FunctionDef]]:
    """Every free function a module declares, name -> overloads."""
    out: Dict[str, List[FunctionDef]] = {}
    for name, name_info in stub_names(module).items():
        defs = funcdefs(name_info, module)
        if defs:
            out[name] = defs
    return out


def stub_class(
    name: str, module: str = "builtins"
) -> Optional[Tuple[tc.NameInfo, str]]:
    """The (class NameInfo, defining module) for ``name``, searched in ``module``
    then builtins and typing, following re-exports."""
    for candidate_module in (module, *_CLASS_FALLBACK_MODULES):
        resolved = _resolve(stub_names(candidate_module).get(name), candidate_module, 0)
        if resolved is not None and isinstance(resolved[0].ast, ast.ClassDef):
            return resolved
    return None


def _base_ref(base: ast.expr, default_module: str) -> Optional[Tuple[str, str]]:
    """(base class name, module to resolve it in) for a base-class expression.

    A bare ``Name`` (``class Fraction(Rational)``) resolves in the module holding
    the deriving class. A dotted ``Attribute`` (``class RegexFlag(enum.IntFlag)``)
    resolves its final attribute in the *qualifier* module, so a base imported as
    ``import enum`` still follows to where the class is defined.
    """
    if isinstance(base, ast.Name):
        return (base.id, default_module)
    if isinstance(base, ast.Attribute):
        qualifier: List[str] = []
        node: ast.expr = base.value
        while isinstance(node, ast.Attribute):
            qualifier.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            qualifier.append(node.id)
            return (base.attr, ".".join(reversed(qualifier)))
    return None


@functools.lru_cache(maxsize=None)
def class_chain(
    name: str, module: str = "builtins"
) -> Tuple[Tuple[tc.NameInfo, str], ...]:
    """Typeshed's MRO for a class: derived-first (NameInfo, defining module) pairs
    following declared bases, with ``object`` last."""
    chain: List[Tuple[tc.NameInfo, str]] = []
    seen = set()

    def visit(cls_name: str, cls_module: str) -> None:
        if cls_name in seen:
            return
        resolved = stub_class(cls_name, cls_module)
        if resolved is None:
            return
        name_info, found_module = resolved
        seen.add(cls_name)
        chain.append(resolved)
        for base in cast(ast.ClassDef, name_info.ast).bases:
            node = base.value if isinstance(base, ast.Subscript) else base
            ref = _base_ref(node, found_module)
            if ref is not None and ref[0] not in ("Generic", "Protocol"):
                visit(*ref)

    visit(name, module)
    if "object" not in seen:
        obj = stub_class("object")
        if obj is not None:
            chain.append(obj)
    return tuple(chain)


def method_funcdefs(
    class_name: str, method: str, module: str = "builtins"
) -> Tuple[List[FunctionDef], str]:
    """(overloads, defining module) for a method, resolved up typeshed's MRO: the
    first class declaring it wins, so an override beats what it inherits.
    ``([], module)`` when no class in the chain declares it."""
    for name_info, found_module in class_chain(class_name, module):
        defs = funcdefs((name_info.child_nodes or {}).get(method), found_module)
        if defs:
            return (defs, found_module)
    return ([], module)


def qualname_funcdefs(
    module: str, path: Sequence[str]
) -> Tuple[List[FunctionDef], str]:
    """(overloads, defining module) for a ``__qualname__`` path inside a module's
    stub. A ``Class.method`` path resolves up the MRO; deeper paths descend nested
    classes literally."""
    if not path:
        return ([], module)
    if len(path) == 1:
        return (funcdefs(stub_names(module).get(path[0]), module), module)
    if len(path) == 2:
        return method_funcdefs(path[0], path[1], module)
    outer = stub_class(path[0], module)
    if outer is None:
        return ([], module)
    name_info, found_module = outer
    for step in path[1:-1]:
        nested = _resolve((name_info.child_nodes or {}).get(step), found_module, 0)
        if nested is None:
            return ([], found_module)
        name_info, found_module = nested
    return (
        funcdefs((name_info.child_nodes or {}).get(path[-1]), found_module),
        found_module,
    )


def decorator_names(funcdef: FunctionDef) -> FrozenSet[str]:
    """The bare/dotted decorator names on a definition, e.g. ``{"overload"}``."""
    names = set()
    for node in funcdef.decorator_list:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return frozenset(names)


@dataclass(frozen=True)
class StubParams:
    """A stub definition's parameters, with the receiver split out.

    ``receiver`` is identified by POSITION, not by name -- typeshed does not always
    call it ``self`` (``fractions.pyi`` declares ``def __neg__(a) -> Fraction``).
    """

    receiver: Optional[ast.arg]
    positional: Tuple[ast.arg, ...]
    defaults: Tuple[ast.expr, ...]
    vararg: Optional[ast.arg]
    kwonly: Tuple[ast.arg, ...]
    kwonly_defaults: Tuple[Optional[ast.expr], ...]
    kwarg: Optional[ast.arg]

    @property
    def required_positional(self) -> Tuple[ast.arg, ...]:
        """Positional parameters with no default."""
        if not self.defaults:
            return self.positional
        return self.positional[: max(0, len(self.positional) - len(self.defaults))]

    @property
    def optional_positional(self) -> Tuple[ast.arg, ...]:
        """Positional parameters with a default, in declaration order."""
        if not self.defaults:
            return ()
        return self.positional[max(0, len(self.positional) - len(self.defaults)) :]

    @property
    def required_kwonly(self) -> Tuple[ast.arg, ...]:
        """Keyword-only parameters with no default."""
        return tuple(
            arg
            for arg, default in zip(self.kwonly, self.kwonly_defaults)
            if default is None
        )

    @property
    def optional_kwonly(self) -> Tuple[ast.arg, ...]:
        """Keyword-only parameters with a default."""
        return tuple(
            arg
            for arg, default in zip(self.kwonly, self.kwonly_defaults)
            if default is not None
        )


def params(funcdef: FunctionDef, is_method: bool = False) -> StubParams:
    """Split a stub definition's parameters into a :class:`StubParams`.

    ``is_method`` claims the leading positional parameter as the receiver, except
    on a ``@staticmethod`` (whose first parameter is a real argument, as in
    ``str.maketrans``).
    """
    args = funcdef.args
    positional = list(args.posonlyargs) + list(args.args)
    receiver = None
    if is_method and positional and "staticmethod" not in decorator_names(funcdef):
        receiver = positional.pop(0)
    return StubParams(
        receiver=receiver,
        positional=tuple(positional),
        defaults=tuple(args.defaults),
        vararg=args.vararg,
        kwonly=tuple(args.kwonlyargs),
        kwonly_defaults=tuple(args.kw_defaults),
        kwarg=args.kwarg,
    )
