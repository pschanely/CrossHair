import dis
from collections.abc import MutableMapping, Set
from sys import version_info
from types import CodeType, FrameType

from crosshair.core import CrossHairValue, register_opcode_patch
from crosshair.libimpl.builtinslib import AnySymbolicStr, SymbolicInt, SymbolicList
from crosshair.simplestructs import LinearSet, ShellMutableSet, SimpleDict, SliceView
from crosshair.tracers import (
    COMPOSITE_TRACER,
    TracingModule,
    frame_stack_read,
    frame_stack_write,
)
from crosshair.util import CrosshairInternal

BINARY_SUBSCR = dis.opmap["BINARY_SUBSCR"]
BUILD_STRING = dis.opmap["BUILD_STRING"]
COMPARE_OP = dis.opmap["COMPARE_OP"]
CONTAINS_OP = dis.opmap.get("CONTAINS_OP", 118)
FORMAT_VALUE = dis.opmap["FORMAT_VALUE"]
MAP_ADD = dis.opmap["MAP_ADD"]
SET_ADD = dis.opmap["SET_ADD"]


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
                    frame_stack_write(frame, -2, SymbolicList(view_wrapper))
            else:
                pass
                # Nothing useful to do with concrete list and symbolic numeric index.


class DeoptimizedContainer:
    def __init__(self, container):
        self.container = container

    def __contains__(self, other):
        return self.container.__contains__(other)


class FormatSatshingValue:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        self.formatted = str(self.value)
        return ""

    def __format__(self, fmt: str):
        self.formatted = format(self.value, fmt)
        return ""

    def __repr__(self) -> str:
        self.formatted = repr(self.value)
        return ""


_CONTAINMENT_OP_TYPES = tuple(
    i for (i, name) in enumerate(dis.cmp_op) if name in ("in", "not in")
)
assert len(_CONTAINMENT_OP_TYPES) in (0, 2)


class ContainmentInterceptor(TracingModule):

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
        item = frame_stack_read(frame, -2)
        if not isinstance(item, CrossHairValue):
            return
        container = frame_stack_read(frame, -1)
        containertype = type(container)
        new_container = None
        if containertype is str:
            new_container = DeoptimizedContainer(container)
        elif containertype is set:
            new_container = ShellMutableSet(LinearSet(container))
        elif containertype is dict:
            new_container = SimpleDict(list(container.items()))

        if new_container is not None:
            frame_stack_write(frame, -1, new_container)


class BuildStringInterceptor(TracingModule):
    """
    Adds symbolic handling for the BUILD_STRING opcode (used by f-strings).

    BUILD_STRING concatenates strings from the stack is a fast, but unforgiving way:
    it requires all the substrings to be real Python strings.
    We work around this by replacing the substrings with empty strings, computing the
    concatenation ourselves, and swaping our result in after the opcode completes.
    """

    opcodes_wanted = frozenset([BUILD_STRING])

    def trace_op(self, frame, codeobj, codenum):
        count = frame_op_arg(frame)
        real_result = ""
        for offset in range(-(count), 0):
            substr = frame_stack_read(frame, offset)
            if not isinstance(substr, (str, AnySymbolicStr)):
                raise CrosshairInternal
            # Because we know these are all symbolic or concrete strings, it's ok to
            # not have tracing on when we do the concatenation here:
            real_result += substr
            frame_stack_write(frame, offset, "")

        def post_op():
            frame_stack_write(frame, -1, real_result)

        COMPOSITE_TRACER.set_postop_callback(codeobj, post_op, frame)


class FormatValueInterceptor(TracingModule):
    """Avoid checks and realization during FORMAT_VALUE (used by f-strings)."""

    opcodes_wanted = frozenset([FORMAT_VALUE])

    def trace_op(self, frame, codeobj, codenum):
        flags = frame_op_arg(frame)
        value_idx = -2 if flags == 0x04 else -1
        orig_obj = frame_stack_read(frame, value_idx)

        # FORMAT_VALUE checks that results are concrete strings. So, we format via a
        # a wrapper that returns an empty str, and then swap in the actual string later:

        wrapper = FormatSatshingValue(orig_obj)
        if flags in (0x00, 0x01) and isinstance(orig_obj, AnySymbolicStr):
            # Just use the symbolic string directly (don't bother formatting at all)
            wrapper.formatted = orig_obj
            frame_stack_write(frame, value_idx, "")
        else:
            frame_stack_write(frame, value_idx, wrapper)

        def post_op():
            frame_stack_write(frame, -1, wrapper.formatted)

        COMPOSITE_TRACER.set_postop_callback(codeobj, post_op, frame)


class MapAddInterceptor(TracingModule):
    """De-optimize MAP_ADD over symbolics (used in dict comprehensions)."""

    opcodes_wanted = frozenset([MAP_ADD])

    def trace_op(self, frame: FrameType, codeobj: CodeType, codenum: int) -> None:
        dict_offset = -(frame_op_arg(frame) + 2)
        dict_obj = frame_stack_read(frame, dict_offset)
        if not isinstance(dict_obj, (dict, MutableMapping)):
            raise CrosshairInternal
        top, second = frame_stack_read(frame, -1), frame_stack_read(frame, -2)
        # Key and value were swapped in Python 3.8
        key, value = (second, top) if version_info >= (3, 8) else (top, second)
        if isinstance(dict_obj, dict):
            if isinstance(key, CrossHairValue):
                dict_obj = SimpleDict(list(dict_obj.items()))
            else:
                # Key and dict are concrete; continue as normal.
                return
        # Have the interpreter do a fake assinment, namely `{}[1] = 1`
        frame_stack_write(frame, dict_offset, {})
        frame_stack_write(frame, -1, 1)
        frame_stack_write(frame, -2, 1)

        # And do our own assignment separately:
        dict_obj[key] = value

        # Later, overwrite the interpreter's result with ours:
        def post_op():
            frame_stack_write(frame, dict_offset + 2, dict_obj)

        COMPOSITE_TRACER.set_postop_callback(codeobj, post_op, frame)


class SetAddInterceptor(TracingModule):
    """De-optimize SET_ADD over symbolics (used in set comprehensions)."""

    opcodes_wanted = frozenset([SET_ADD])

    def trace_op(self, frame: FrameType, codeobj: CodeType, codenum: int) -> None:
        set_offset = -(frame_op_arg(frame) + 1)
        set_obj = frame_stack_read(frame, set_offset)
        if not isinstance(set_obj, Set):
            raise CrosshairInternal(type(set_obj))
        item = frame_stack_read(frame, -1)
        if isinstance(set_obj, set):
            if isinstance(item, CrossHairValue):
                set_obj = ShellMutableSet(set_obj)
            else:
                # Set and value are concrete; continue as normal.
                return
        # Have the interpreter do a fake addition, namely `set().add(1)`
        frame_stack_write(frame, set_offset, set())
        frame_stack_write(frame, -1, 1)

        # And do our own addition separately:
        set_obj.add(item)

        # Later, overwrite the interpreter's result with ours:
        def post_op():
            frame_stack_write(frame, set_offset + 1, set_obj)

        COMPOSITE_TRACER.set_postop_callback(codeobj, post_op, frame)


def make_registrations():
    register_opcode_patch(SymbolicSubscriptInterceptor())
    register_opcode_patch(ContainmentInterceptor())
    register_opcode_patch(BuildStringInterceptor())
    register_opcode_patch(FormatValueInterceptor())
    register_opcode_patch(MapAddInterceptor())
    register_opcode_patch(SetAddInterceptor())
