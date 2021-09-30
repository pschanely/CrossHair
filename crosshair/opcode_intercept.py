import dis

from crosshair.core import register_opcode_patch
from crosshair.libimpl.builtinslib import SymbolicInt
from crosshair.libimpl.builtinslib import AnySymbolicStr
from crosshair.libimpl.builtinslib import LazyIntSymbolicStr
from crosshair.simplestructs import ShellMutableSequence, SimpleDict, SliceView
from crosshair.tracers import TracingModule
from crosshair.tracers import frame_stack_read
from crosshair.tracers import frame_stack_write
from crosshair.util import debug

BINARY_SUBSCR = dis.opmap["BINARY_SUBSCR"]
COMPARE_OP = dis.opmap["COMPARE_OP"]
CONTAINS_OP = dis.opmap.get("CONTAINS_OP", 118)


def frame_op_arg(frame):
    return frame.f_code.co_code[frame.f_lasti + 1]


class SymbolicSubscriptInterceptor(TracingModule):
    opcodes_wanted = frozenset([BINARY_SUBSCR])

    def trace_op(self, frame, codeobj, codenum):
        # Note that because this is called from inside a Python trace handler, tracing
        # is automatically disabled, so there's no need for a `with NoTracing():` guard.
        key = frame_stack_read(frame, -1)
        if isinstance(key, (int, float, str)):
            return
        # If we got this far, the index is likely symbolic (or perhaps a slice object)
        container = frame_stack_read(frame, -2)
        container_type = type(container)
        if container_type is dict:
            # SimpleDict won't hash the keys it's given!
            wrapped_dict = SimpleDict(list(container.items()))
            frame_stack_write(frame, -2, wrapped_dict)
        elif container_type is list:
            if isinstance(key, slice):
                if key.step not in (1, None):
                    return
                start, stop = key.start, key.stop
                if isinstance(start, SymbolicInt) or isinstance(stop, SymbolicInt):
                    view_wrapper = SliceView(container, 0, len(container))
                    frame_stack_write(frame, -2, ShellMutableSequence(view_wrapper))
            else:
                pass
                # Nothing useful to do with concrete list and symbolic numeric index.


_CONTAINMENT_OP_TYPES = tuple(
    i for (i, name) in enumerate(dis.cmp_op) if name in ("in", "not in")
)
assert len(_CONTAINMENT_OP_TYPES) in (0, 2)


class StringContainmentInterceptor(TracingModule):

    opcodes_wanted = frozenset(
        [
            COMPARE_OP,
            CONTAINS_OP,
        ]
    )

    def trace_op(self, frame, codeobj, codenum):
        if codenum == COMPARE_OP:
            compare_type = frame_op_arg(frame)
            if compare_type not in _CONTAINMENT_OP_TYPES:
                return
        container = frame_stack_read(frame, -1)
        item = frame_stack_read(frame, -2)
        if type(item) is AnySymbolicStr and type(container) is str:
            new_container = LazyIntSymbolicStr([ord(c) for c in container])
            frame_stack_write(frame, -1, new_container)


def make_registrations():
    register_opcode_patch(SymbolicSubscriptInterceptor())
    register_opcode_patch(StringContainmentInterceptor())
