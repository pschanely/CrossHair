import dis
import sys
import weakref
from collections import defaultdict
from collections.abc import MutableMapping, Set
from sys import version_info
from types import CodeType, FrameType
from typing import Any, Callable, Iterable, Mapping, Tuple, Union

from crosshair.core import (
    ATOMIC_IMMUTABLE_TYPES,
    register_opcode_patch,
    with_uniform_probabilities,
)
from crosshair.libimpl.builtinslib import (
    AnySymbolicStr,
    AtomicSymbolicValue,
    ModelingDirector,
    SymbolicBool,
    SymbolicInt,
    SymbolicList,
    python_types_using_atomic_symbolics,
)
from crosshair.simplestructs import LinearSet, ShellMutableSet, SimpleDict, SliceView
from crosshair.statespace import context_statespace
from crosshair.tracers import (
    COMPOSITE_TRACER,
    NoTracing,
    ResumedTracing,
    TracingModule,
    frame_stack_read,
    frame_stack_write,
)
from crosshair.util import (
    CROSSHAIR_EXTRA_ASSERTS,
    CrossHairInternal,
    CrossHairValue,
    debug,
)
from crosshair.z3util import z3Not, z3Or

BINARY_SUBSCR = dis.opmap.get("BINARY_SUBSCR", 256)
BINARY_SLICE = dis.opmap.get("BINARY_SLICE", 256)
BUILD_STRING = dis.opmap["BUILD_STRING"]
COMPARE_OP = dis.opmap["COMPARE_OP"]
CONTAINS_OP = dis.opmap.get("CONTAINS_OP", 256)
FORMAT_VALUE = dis.opmap.get("FORMAT_VALUE", 256)
CONVERT_VALUE = dis.opmap.get("CONVERT_VALUE", 256)
MAP_ADD = dis.opmap["MAP_ADD"]
SET_ADD = dis.opmap["SET_ADD"]
UNARY_NOT = dis.opmap["UNARY_NOT"]
TO_BOOL = dis.opmap.get("TO_BOOL", 256)
IS_OP = dis.opmap.get("IS_OP", 256)
BINARY_MODULO = dis.opmap.get("BINARY_MODULO", 256)
BINARY_OP = dis.opmap.get("BINARY_OP", 256)


def frame_op_arg(frame):
    return frame.f_code.co_code[frame.f_lasti + 1]  # TODO: account for EXTENDED_ARG?


_DEEPLY_CONCRETE_KEY_TYPES = (
    int,
    float,
    str,
    # Suble but important; when subscripting a Weak[Key|Value]Dictionary,
    # we need to avoid creating a SimpleDict out of the backing dictionary.
    # (because it can drop keys during iteration and fail)
    weakref.ref,
)


class MultiSubscriptableContainer:
    """Used for indexing a symbolic (non-slice) key into a concrete container"""

    def __init__(self, container: Union[list, tuple, dict]):
        self.container = container

    def __getitem__(self, key: AtomicSymbolicValue) -> object:
        with NoTracing():
            space = context_statespace()
            container = self.container
            if isinstance(container, Mapping):
                kv_pairs: Iterable[Tuple[Any, Any]] = container.items()
            else:
                in_bounds = space.smt_fork(
                    z3Or(-len(container) <= key.var, key.var < len(container)),
                    desc=f"index_in_bounds",
                    probability_true=0.9,
                )
                if not in_bounds:
                    raise IndexError
                kv_pairs = enumerate(container)

            values_by_type = defaultdict(list)
            values_by_id = {}
            keys_by_value_id = defaultdict(list)
            symbolic_for_pytype = space.extra(ModelingDirector).choose
            for cur_key, cur_value in kv_pairs:
                if (
                    isinstance(cur_value, AtomicSymbolicValue)
                    or type(cur_value) in python_types_using_atomic_symbolics()
                ):
                    pytype = (
                        cur_value._pytype()
                        if isinstance(cur_value, AtomicSymbolicValue)
                        else type(cur_value)
                    )
                    # Some types like real-based float and symbolic types don't cover all values:
                    if (
                        symbolic_for_pytype(pytype)._smt_promote_literal(cur_value)
                        is not None
                    ):
                        values_by_type[pytype].append((cur_key, cur_value))
                        continue
                # No symbolics cover this value, but we might still find repeated values:
                values_by_id[id(cur_value)] = cur_value
                keys_by_value_id[id(cur_value)].append(cur_key)
            for value_type, cur_pairs in values_by_type.items():
                hypothetical_result = symbolic_for_pytype(value_type)(
                    "item_at_" + space.uniq(), value_type
                )
                with ResumedTracing():
                    condition_pairs = []
                    for cur_key, cur_val in cur_pairs:
                        keys_equal = key == cur_key
                        values_equal = hypothetical_result == cur_val
                        with NoTracing():
                            if isinstance(keys_equal, SymbolicBool):
                                condition_pairs.append((keys_equal, values_equal))
                            elif keys_equal is False:
                                pass
                            else:
                                # (because the key must be symbolic, we don't ever expect raw True)
                                raise CrossHairInternal(
                                    f"key comparison type: {type(keys_equal)} {keys_equal}"
                                )
                    if any(keys_equal for keys_equal, _ in condition_pairs):
                        space.add(any([all(pair) for pair in condition_pairs]))
                        return hypothetical_result

            for (value_id, value), probability_true in with_uniform_probabilities(
                values_by_id.items()
            ):
                keys_for_value = keys_by_value_id[value_id]
                with ResumedTracing():
                    is_match = any([key == k for k in keys_for_value])
                if isinstance(is_match, SymbolicBool):
                    if space.smt_fork(
                        is_match.var,
                        probability_true=probability_true,
                    ):
                        return value
                elif is_match:
                    return value

            if type(container) is dict:
                raise KeyError  # ( f"Key {key} not found in dict")
            else:
                raise IndexError  # (f"Index {key} out of range for list/tuple of length {len(container)}")


class SymbolicSubscriptInterceptor(TracingModule):
    opcodes_wanted = frozenset([BINARY_SUBSCR, BINARY_OP])

    def trace_op(self, frame, codeobj, codenum):
        if codenum == BINARY_OP:
            oparg = frame_op_arg(frame)
            if oparg != 26:  # subscript operator, NB_SUBSCR
                return

        key = frame_stack_read(frame, -1)
        if isinstance(key, _DEEPLY_CONCRETE_KEY_TYPES):
            return
        # If we got this far, the index is likely symbolic (or perhaps a slice object)
        container = frame_stack_read(frame, -2)
        container_type = type(container)
        if isinstance(key, AtomicSymbolicValue) and type(container) in (
            tuple,
            list,
            dict,
        ):
            wrapped_container = MultiSubscriptableContainer(container)
            frame_stack_write(frame, -2, wrapped_container)
        elif container_type is dict:
            # SimpleDict won't hash the keys it's given!
            wrapped_dict = SimpleDict(list(container.items()))
            frame_stack_write(frame, -2, wrapped_dict)
        elif isinstance(key, slice) and container_type is list:
            step = key.step
            if isinstance(step, CrossHairValue) or step not in (None, 1):
                return
            start, stop = key.start, key.stop
            if isinstance(start, SymbolicInt) or isinstance(stop, SymbolicInt):
                view_wrapper = SliceView(container, 0, len(container))
                frame_stack_write(frame, -2, SymbolicList(view_wrapper))


class SymbolicSliceInterceptor(TracingModule):
    opcodes_wanted = frozenset([BINARY_SLICE])

    def trace_op(
        self, frame, codeobj, codenum, _concrete_index_types=(int, float, str)
    ):
        # Note that because this is called from inside a Python trace handler, tracing
        # is automatically disabled, so there's no need for a `with NoTracing():` guard.
        start = frame_stack_read(frame, -1)
        stop = frame_stack_read(frame, -2)
        if isinstance(start, _concrete_index_types) and isinstance(
            stop, _concrete_index_types
        ):
            return
        # If we got this far, the index is likely symbolic (or perhaps a slice object)
        container = frame_stack_read(frame, -3)
        container_type = type(container)
        if container_type is list:
            if isinstance(start, SymbolicInt) or isinstance(stop, SymbolicInt):
                view_wrapper = SliceView(container, 0, len(container))
                frame_stack_write(frame, -3, SymbolicList(view_wrapper))


class DeoptimizedContainer:
    def __init__(self, container):
        self.container = container

    def __contains__(self, other):
        return self.container.__contains__(other)


class SideEffectStashingHashable:
    def __init__(self, fn: Callable):
        self.fn = fn

    def __hash__(self):
        self.result = self.fn()
        return 0


class DeoptimizedPercentFormattingStr:
    def __init__(self, value):
        self.value = value

    def __mod__(self, other):
        return self.value.__mod__(other)


class FormatStashingValue:
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


class BoolStashingValue:
    def __init__(self, value, negate):
        self.value = value
        self.negate = negate

    def __bool__(self):
        stashed_bool = self.value.__bool__()
        with NoTracing():
            if self.negate:
                if isinstance(stashed_bool, SymbolicBool):
                    self.stashed_bool = SymbolicBool(z3Not(stashed_bool.var))
                else:
                    self.stashed_bool = not stashed_bool
            else:
                self.stashed_bool = stashed_bool
        return True


_CONTAINMENT_OP_TYPES = tuple(
    i for (i, name) in enumerate(dis.cmp_op) if name in ("in", "not in")
)
assert len(_CONTAINMENT_OP_TYPES) in (0, 2)

_COMPARE_ISOP_TYPES = tuple(
    i for (i, name) in enumerate(dis.cmp_op) if name in ("is", "is not")
)
assert len(_COMPARE_ISOP_TYPES) in (0, 2)


class ComparisonInterceptForwarder(TracingModule):

    opcodes_wanted = frozenset([COMPARE_OP])

    def trace_op(self, frame, codeobj, codenum):
        # Python 3.8 used a general purpose comparison opcode.
        # Forward to dedicated opcode handlers as appropriate.
        compare_type = frame_op_arg(frame)
        if compare_type in _CONTAINMENT_OP_TYPES:
            ContainmentInterceptor.trace_op(None, frame, codeobj, codenum)
        elif compare_type in _COMPARE_ISOP_TYPES:
            IdentityInterceptor.trace_op(None, frame, codeobj, codenum)


class ContainmentInterceptor(TracingModule):

    opcodes_wanted = frozenset([CONTAINS_OP])

    def trace_op(self, frame, codeobj, codenum):
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
                raise CrossHairInternal
            # Because we know these are all symbolic or concrete strings, it's ok to
            # not have tracing on when we do the concatenation here:
            real_result += substr
            frame_stack_write(frame, offset, "")

        def post_op():
            frame_stack_write(frame, -1, real_result)

        COMPOSITE_TRACER.set_postop_callback(post_op, frame)


class FormatValueInterceptor(TracingModule):
    """Avoid checks and realization during FORMAT_VALUE (used by f-strings)."""

    # TODO: don't we need to handle FORMAT_SIMPLE and FORMAT_WITH_SPEC?
    opcodes_wanted = frozenset([FORMAT_VALUE, CONVERT_VALUE])

    def trace_op(self, frame, codeobj, codenum):
        flags = frame_op_arg(frame)
        value_idx = -2 if flags == 0x04 else -1
        orig_obj = frame_stack_read(frame, value_idx)

        # FORMAT_VALUE checks that results are concrete strings. So, we format via a
        # a wrapper that returns an empty str, and then swap in the actual string later:

        wrapper = FormatStashingValue(orig_obj)
        if flags in (0x00, 0x01) and isinstance(orig_obj, AnySymbolicStr):
            # Just use the symbolic string directly (don't bother formatting at all)
            wrapper.formatted = orig_obj
            frame_stack_write(frame, value_idx, "")
        else:
            frame_stack_write(frame, value_idx, wrapper)

        def post_op():
            frame_stack_write(frame, -1, wrapper.formatted)

        COMPOSITE_TRACER.set_postop_callback(post_op, frame)


class MapAddInterceptor(TracingModule):
    """De-optimize MAP_ADD over symbolics (used in dict comprehensions)."""

    opcodes_wanted = frozenset([MAP_ADD])

    def trace_op(self, frame: FrameType, codeobj: CodeType, codenum: int) -> None:
        dict_offset = -(frame_op_arg(frame) + 2)
        dict_obj = frame_stack_read(frame, dict_offset)
        if not isinstance(dict_obj, (dict, MutableMapping)):
            raise CrossHairInternal
        # Key and value were swapped in Python 3.8
        key_offset, value_offset = (-2, -1) if version_info >= (3, 8) else (-1, -2)
        key = frame_stack_read(frame, key_offset)
        value = frame_stack_read(frame, value_offset)
        if isinstance(dict_obj, dict):
            if type(key) in ATOMIC_IMMUTABLE_TYPES:
                # Dict and key is (deeply) concrete; continue as normal.
                return
            else:
                dict_obj = SimpleDict(list(dict_obj.items()))

        # Have the interpreter do a fake assinment.
        # While the fake assignment happens, we'll perform the real assignment secretly
        # when Python hashes the fake key.
        def do_real_assignment():
            dict_obj[key] = value

        frame_stack_write(frame, dict_offset, {})
        frame_stack_write(frame, value_offset, 1)
        frame_stack_write(
            frame, key_offset, SideEffectStashingHashable(do_real_assignment)
        )

        # Afterwards, overwrite the interpreter's resulting dict with ours:
        def post_op():
            old_dict_obj = frame_stack_read(frame, dict_offset + 2)
            if CROSSHAIR_EXTRA_ASSERTS and not isinstance(
                old_dict_obj, (dict, MutableMapping)
            ):
                raise CrossHairInternal("interpreter stack corruption detected")
            frame_stack_write(frame, dict_offset + 2, dict_obj)

        COMPOSITE_TRACER.set_postop_callback(post_op, frame)


class ToBoolInterceptor(TracingModule):
    """Retain symbolic booleans across the TO_BOOL operator."""

    opcodes_wanted = frozenset([TO_BOOL])

    def trace_op(self, frame: FrameType, codeobj: CodeType, codenum: int) -> None:
        input_bool = frame_stack_read(frame, -1)
        if not isinstance(input_bool, CrossHairValue):
            return
        if isinstance(input_bool, SymbolicBool):
            # TODO: right now, we define __bool__ methods to perform realization.
            # At some point, if that isn't the case, and we can remove this specialized
            # branch for `SybolicBool`.
            frame_stack_write(frame, -1, True)

            def post_op():
                frame_stack_write(frame, -1, input_bool)

        else:
            stashing_value = BoolStashingValue(input_bool, negate=False)
            frame_stack_write(frame, -1, stashing_value)

            def post_op():
                frame_stack_write(frame, -1, stashing_value.stashed_bool)

        COMPOSITE_TRACER.set_postop_callback(post_op, frame)


class NotInterceptor(TracingModule):
    """Retain symbolic booleans across the `not` operator."""

    opcodes_wanted = frozenset([UNARY_NOT])

    def trace_op(self, frame: FrameType, codeobj: CodeType, codenum: int) -> None:
        input_bool = frame_stack_read(frame, -1)
        if not isinstance(input_bool, CrossHairValue):
            return

        if isinstance(input_bool, SymbolicBool):
            # TODO: right now, we define __bool__ methods to perform realization.
            # At some point, if that isn't the case, and we can remove this specialized
            # branch for `SybolicBool`.
            frame_stack_write(frame, -1, True)

            def post_op():
                frame_stack_write(frame, -1, SymbolicBool(z3Not(input_bool.var)))

        else:
            stashing_value = BoolStashingValue(input_bool, negate=True)
            frame_stack_write(frame, -1, stashing_value)

            def post_op():
                frame_stack_write(frame, -1, stashing_value.stashed_bool)

        COMPOSITE_TRACER.set_postop_callback(post_op, frame)


class SetAddInterceptor(TracingModule):
    """De-optimize SET_ADD over symbolics (used in set comprehensions)."""

    opcodes_wanted = frozenset([SET_ADD])

    def trace_op(self, frame: FrameType, codeobj: CodeType, codenum: int) -> None:
        set_offset = -(frame_op_arg(frame) + 1)
        set_obj = frame_stack_read(frame, set_offset)
        if not isinstance(set_obj, Set):
            raise CrossHairInternal(type(set_obj))
        item = frame_stack_read(frame, -1)
        if isinstance(set_obj, set):
            if isinstance(item, CrossHairValue):
                set_obj = ShellMutableSet(set_obj)
            else:
                # Set and value are concrete; continue as normal.
                return
        # Have the interpreter do a fake addition, namely `set().add(1)`
        dummy_set: Set = set()
        frame_stack_write(frame, set_offset, dummy_set)
        frame_stack_write(frame, -1, 1)

        # And do our own addition separately:
        set_obj.add(item)

        # Later, overwrite the interpreter's result with ours:
        def post_op():
            if CROSSHAIR_EXTRA_ASSERTS:
                to_replace = frame_stack_read(frame, set_offset + 1)
                if to_replace is not dummy_set:
                    raise CrossHairInternal(
                        f"Found an instance of {type(to_replace)} where dummy set should be."
                    )
            frame_stack_write(frame, set_offset + 1, set_obj)

        COMPOSITE_TRACER.set_postop_callback(post_op, frame)


class IdentityInterceptor(TracingModule):
    """Detect an "is" comparison to symbolics booleans"""

    opcodes_wanted = frozenset([IS_OP])
    # TODO: Adding support for an OptionalSymbolic would now be possible.
    # TODO: it would be amazing to add symbolic enums and support comparison here

    def trace_op(self, frame: FrameType, codeobj: CodeType, codenum: int) -> None:
        arg1 = frame_stack_read(frame, -1)
        arg2 = frame_stack_read(frame, -2)
        if isinstance(arg1, SymbolicBool) and isinstance(arg2, (bool, SymbolicBool)):
            frame_stack_write(frame, -1, arg1.__ch_realize__())
        if isinstance(arg2, SymbolicBool) and isinstance(arg1, (bool, SymbolicBool)):
            frame_stack_write(frame, -2, arg2.__ch_realize__())


class ModuloInterceptor(TracingModule):
    opcodes_wanted = frozenset([BINARY_MODULO, BINARY_OP])
    assert BINARY_MODULO != BINARY_OP

    def trace_op(self, frame, codeobj, codenum):
        left = frame_stack_read(frame, -2)
        from crosshair.util import debug

        if isinstance(left, str):
            if codenum == BINARY_OP:
                oparg = frame_op_arg(frame)
                if oparg != 6:  # modulo operator, NB_REMAINDER
                    return
            frame_stack_write(frame, -2, DeoptimizedPercentFormattingStr(left))


def make_registrations():
    register_opcode_patch(SymbolicSubscriptInterceptor())
    if sys.version_info >= (3, 12):
        register_opcode_patch(SymbolicSliceInterceptor())
    if sys.version_info < (3, 9):
        register_opcode_patch(ComparisonInterceptForwarder())
    register_opcode_patch(ContainmentInterceptor())
    register_opcode_patch(BuildStringInterceptor())
    register_opcode_patch(FormatValueInterceptor())
    register_opcode_patch(MapAddInterceptor())
    # register_opcode_patch(ToBoolInterceptor())
    register_opcode_patch(NotInterceptor())
    register_opcode_patch(SetAddInterceptor())
    register_opcode_patch(IdentityInterceptor())
    register_opcode_patch(ModuloInterceptor())
