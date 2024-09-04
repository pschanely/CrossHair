import collections
import inspect
import sys
from typing import Dict, List, Optional, Type

import z3  # type: ignore

from crosshair.tracers import NoTracing
from crosshair.util import CrossHairInternal, CrossHairValue
from crosshair.z3util import z3Eq, z3Not

_MAP: Optional[Dict[type, List[type]]] = None

_IGNORED_MODULE_ROOTS = {
    # CrossHair will get confused if we try to proxy our own types:
    "crosshair",
    "z3",
    # These are disabled for performance or type search effectiveness:
    "hypothesis",
    "pkg_resources",
    "pytest",
    "py",  # (part of pytest)
}


def _add_class(cls: type) -> None:
    """Add a class just for testing purposes."""
    global _MAP
    get_subclass_map()  # ensure we're initialized
    assert _MAP is not None
    for base in cls.__bases__:
        subs = _MAP[base]
        if cls not in subs:
            subs.append(cls)


_UNSAFE_MEMBERS = frozenset(
    ["__copy___", "__deepcopy__", "__reduce_ex__", "__reduce__"]
)


def _class_known_to_be_copyable(cls: type) -> bool:
    if _UNSAFE_MEMBERS & cls.__dict__.keys():
        return False
    return True


def get_subclass_map() -> Dict[type, List[type]]:
    """
    Crawl all types presently in memory and makes a map from parent to child classes.

    Only direct children are included.
    TODO: Does not yet handle "protocol" subclassing (eg "Iterator", "Mapping", etc).
    """
    global _MAP
    if _MAP is None:
        classes = set()
        for module_name, module in list(sys.modules.items()):
            if module_name.split(".", 1)[0] in _IGNORED_MODULE_ROOTS:
                continue
            if module is None:
                # We set the internal _datetime module to None, ensuring that
                # we don't load the C implementation.
                continue
            try:
                module_classes = inspect.getmembers(module, inspect.isclass)
            except ImportError:
                continue
            for _, cls in module_classes:
                if _class_known_to_be_copyable(cls):
                    classes.add(cls)
        subclass = collections.defaultdict(list)
        for cls in classes:
            for base in cls.__bases__:
                if base in classes:
                    subclass[base].append(cls)
        _MAP = subclass
    return _MAP


def rebuild_subclass_map():
    global _MAP
    _MAP = None


def _make_maybe_sort():
    datatype = z3.Datatype("optional_bool")
    datatype.declare("yes")
    datatype.declare("no")
    datatype.declare("exception")
    return datatype.create()


MAYBE_SORT = _make_maybe_sort()


def _pyissubclass(pytype1: Type, pytype2: Type) -> z3.ExprRef:
    try:
        return MAYBE_SORT.yes if issubclass(pytype1, pytype2) else MAYBE_SORT.no
    except Exception:
        return MAYBE_SORT.exception


PYTYPE_SORT = z3.DeclareSort("pytype_sort")

SMT_SUBTYPE_FN = z3.Function(
    "pytype_sort_subtype", PYTYPE_SORT, PYTYPE_SORT, MAYBE_SORT
)


class SymbolicTypeRepository:
    pytype_to_smt: Dict[Type, z3.ExprRef]

    def __init__(self, solver: z3.Solver):
        self.pytype_to_smt = {}
        self.solver = solver

    def smt_can_subclass(self, typ1: z3.ExprRef, typ2: z3.ExprRef) -> z3.ExprRef:
        return SMT_SUBTYPE_FN(typ1, typ2) != MAYBE_SORT.exception

    def smt_issubclass(self, typ1: z3.ExprRef, typ2: z3.ExprRef) -> z3.ExprRef:
        return SMT_SUBTYPE_FN(typ1, typ2) == MAYBE_SORT.yes

    def get_type(self, typ: Type) -> z3.ExprRef:
        with NoTracing():
            if issubclass(typ, CrossHairValue):
                raise CrossHairInternal(
                    "Type repo should not be queried with a symbolic"
                )
        pytype_to_smt = self.pytype_to_smt
        if typ not in pytype_to_smt:
            stmts = []
            expr = z3.Const(f"typrepo_{typ.__qualname__}_{id(typ):x}", PYTYPE_SORT)
            for other_pytype, other_expr in pytype_to_smt.items():
                stmts.append(z3Not(z3Eq(other_expr, expr)))
                stmts.append(
                    z3Eq(
                        SMT_SUBTYPE_FN(expr, other_expr),
                        _pyissubclass(typ, other_pytype),
                    )
                )
                stmts.append(
                    z3Eq(
                        SMT_SUBTYPE_FN(other_expr, expr),
                        _pyissubclass(other_pytype, typ),
                    )
                )
            stmts.append(z3Eq(SMT_SUBTYPE_FN(expr, expr), _pyissubclass(typ, typ)))
            self.solver.add(stmts)
            pytype_to_smt[typ] = expr
        return pytype_to_smt[typ]
