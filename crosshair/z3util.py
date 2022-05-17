import z3  # type: ignore
from z3 import IntNumRef, IntSort, Z3_mk_numeral

from crosshair.tracers import NoTracing

_ctx = z3.main_ctx()
_ctx_ref = _ctx.ref()
_int_sort_ast = IntSort(_ctx).ast


def z3IntVal(x: int) -> z3.IntNumRef:
    with NoTracing():  # TODO: Ideally, tracing would never be on when we get here.
        # Use __index__ to get a regular integer for int subtypes (e.g. enums)
        return IntNumRef(
            Z3_mk_numeral(_ctx_ref, x.__index__().__str__(), _int_sort_ast), _ctx
        )
