import collections
import inspect
import sys
from typing import *

from crosshair.util import debug
import z3  # type: ignore

_MAP = None

def get_subclass_map():
    '''
    Crawls all types presently in memory and makes a map from parent to child classes.
    Only direct children are included.
    Does not yet handle "protocol" subclassing (eg "Iterator", "Mapping", etc).

    >>> SmtTypeRepository in get_subclass_map()[object]
    True
    '''
    global _MAP
    if _MAP is None:
        classes = set()
        modules = list(sys.modules.values())
        for module in modules:
            try:
                members = inspect.getmembers(module, inspect.isclass)
            except ModuleNotFoundError:
                continue
            for _, member in members:
                classes.add(member)
        subclass = collections.defaultdict(list)
        for cls in classes:
            for base in cls.__bases__:
                subclass[base].append(cls)
        _MAP = subclass
    return _MAP


def rebuild_subclass_map():
    global _MAP
    _MAP = None


PYTYPE_SORT = z3.DeclareSort('pytype_sort')
SMT_SUBTYPE_FN = z3.Function(
    'pytype_sort_subtype', PYTYPE_SORT, PYTYPE_SORT, z3.BoolSort())

class SmtTypeRepository:
    pytype_to_smt: Dict[Type, z3.ExprRef]
    def __init__(self, solver: z3.Solver):
        self.pytype_to_smt = {}
        self.solver = solver
        # preload a few:
        for typ in (object, int, str):
            self.get_type(typ)

    def smt_issubclass(self, typ1: z3.ExprRef, typ2: z3.ExprRef) -> z3.ExprRef:
        return SMT_SUBTYPE_FN(typ1, typ2)

    def issubclass(self, typ1: Type, typ2: Type) -> z3.ExprRef:
        return SMT_SUBTYPE_FN(self.get_type(typ1),
                              self.get_type(typ2))

    def get_type(self, typ: Type) -> z3.ExprRef:
        pytype_to_smt = self.pytype_to_smt
        if typ not in pytype_to_smt:
            stmts = []
            expr = z3.Const('typrepo_'+typ.__qualname__, PYTYPE_SORT)
            for other_pytype, other_expr in pytype_to_smt.items():
                stmts.append(other_expr != expr)
                stmts.append(SMT_SUBTYPE_FN(expr, other_expr) ==
                             issubclass(typ, other_pytype))
                stmts.append(SMT_SUBTYPE_FN(other_expr, expr) ==
                             issubclass(other_pytype, typ))
            stmts.append(SMT_SUBTYPE_FN(expr, expr) == True)
            self.solver.add(stmts)
            pytype_to_smt[typ] = expr
        return pytype_to_smt[typ]
