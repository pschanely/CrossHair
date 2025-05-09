import re
import sys
from collections import Counter, defaultdict, deque, namedtuple
from copy import deepcopy
from inspect import Parameter, Signature
from typing import Counter, DefaultDict, Deque, NamedTuple, Tuple

import pytest

from crosshair.core import (
    deep_realize,
    get_constructor_signature,
    proxy_for_type,
    realize,
    standalone_statespace,
)
from crosshair.libimpl.collectionslib import ListBasedDeque
from crosshair.statespace import CANNOT_CONFIRM, CONFIRMED, POST_FAIL, MessageType
from crosshair.test_util import check_states
from crosshair.tracers import NoTracing, ResumedTracing
from crosshair.util import CrossHairValue


@pytest.fixture
def test_list():
    return ListBasedDeque([1, 2, 3, 4, 5])


def test_counter_symbolic_deep(space):
    d = proxy_for_type(Counter[int], "d")
    with ResumedTracing():
        deep_realize(d)
        deepcopy(d)


def test_counter_deep(space):
    d = Counter()
    with ResumedTracing():
        deep_realize(d)
        deepcopy(d)


def test_deque_appendleft(test_list) -> None:
    test_list.appendleft(0)
    assert test_list.popleft() == 0


def test_deque_appendleft_with_full_deque(test_list) -> None:
    temp_list = ListBasedDeque([1, 2, 3, 4, 5], maxlen=5)
    temp_list.appendleft(42)
    assert temp_list.popleft() == 42 and temp_list.pop() == 4


def test_deque_appendleft_doesnt_increase_size_with_maxlen(test_list) -> None:
    temp_list = ListBasedDeque([1, 2, 3, 4, 5], maxlen=5)
    temp_list.appendleft(42)
    assert len(temp_list) == temp_list.maxlen()


def test_deque_append(test_list) -> None:
    test_list.append(0)
    assert test_list.pop() == 0


def test_deque_append_with_full_deque(test_list) -> None:
    temp_list = ListBasedDeque([1, 2, 3, 4, 5], maxlen=5)
    temp_list.append(42)
    assert temp_list.pop() == 42 and temp_list.popleft() == 2


def test_deque_append_doesnt_increase_size_with_maxlen(test_list) -> None:
    temp_list = ListBasedDeque([1, 2, 3, 4, 5], maxlen=5)
    temp_list.append(42)
    assert len(temp_list) == temp_list.maxlen()


def test_deque_clear(test_list) -> None:
    test_list.clear()
    assert len(test_list) == 0


def test_deque_count(test_list) -> None:
    test_list.append(1)
    count = test_list.count(1)
    assert count == 2


def test_deque_index(test_list) -> None:
    i = test_list.index(5)
    assert i == 4


def test_deque_index_with_start_index(test_list) -> None:
    i = test_list.index(5, 1)
    assert i == 4


def test_deque_index_with_start_index_throws_correct_exception(test_list) -> None:
    with pytest.raises(ValueError) as context:
        test_list.index(1, 2)

    if sys.version_info >= (3, 14):
        # assert context.match(re.escape("list.index(x): x not in list"))
        assert context.match(re.escape("deque.index(x): x not in deque"))
    else:
        assert context.match("1 is not in list")


def test_deque_index_with_start_and_end_index(test_list) -> None:
    i = test_list.index(2, 0, 3)
    assert i == 1


def test_deque_index_with_start_and_end_index_throws_correct_exception(
    test_list,
) -> None:
    with pytest.raises(ValueError) as context:
        test_list.index(6, 0, 1)

    if sys.version_info >= (3, 14):
        assert context.match(re.escape("deque.index(x): x not in deque"))
    else:
        assert context.match("6 is not in list")


def test_deque_insert(test_list) -> None:
    test_list.insert(index=1, item=42)
    assert test_list.index(42) == 1


def test_deque_pop(test_list) -> None:
    assert test_list.pop() == 5


def test_deque_popleft(test_list) -> None:
    assert test_list.popleft() == 1


def test_deque_remove(test_list) -> None:
    original_length = len(test_list)
    test_list.remove(1)
    assert len(test_list) == original_length - 1


def test_deque_reverse(test_list) -> None:
    test_list.reverse()
    assert test_list.popleft() == 5


def test_deque_rotate(test_list) -> None:
    test_list.rotate(n=1)
    assert test_list.popleft() == 5


def test_deque_rotate_left(test_list) -> None:
    test_list.rotate(n=-1)
    assert test_list.popleft() == 2


def test_deque_maxlen(test_list) -> None:
    ls = ListBasedDeque([1, 2, 3], 5)
    assert ls.maxlen() == 5


def test_deque_len_ok() -> None:
    def f(ls: Deque[int]) -> Deque[int]:
        """
        post: len(_) == len(__old__.ls) + 1
        """
        ls.append(42)
        return ls

    check_states(f, CONFIRMED)


@pytest.mark.demo
def test_deque___len___method() -> None:
    def f(ls: Deque[int]) -> int:
        """
        Can the length of a deque equal the value of its last element?

        pre: len(ls) >= 2
        post: _ != ls[-1]
        """
        return len(ls)

    check_states(f, POST_FAIL)


@pytest.mark.demo
def test_deque_extendleft_method() -> None:
    def f(ls: Deque[int]) -> None:
        """
        Can any deque be extended by itself and form this palindrome?

        post[ls]: ls != deque([1, 2, 3, 3, 2, 1])
        """
        ls.extendleft(ls)

    check_states(f, POST_FAIL)


def test_deque_add_symbolic_to_concrete():
    with standalone_statespace as space:
        d = ListBasedDeque([1, 2]) + deque([3, 4])
        assert list(d) == [1, 2, 3, 4]


def test_deque_eq():
    with standalone_statespace as space:
        assert ListBasedDeque([1, 2]) == ListBasedDeque([1, 2])
        assert deque([1, 2]) == ListBasedDeque([1, 2])
        assert ListBasedDeque([1, 2]) != ListBasedDeque([1, 55])
        assert deque([1, 2]) != ListBasedDeque([1, 55])


def test_defaultdict_repr_equiv(test_list) -> None:
    def f(symbolic: DefaultDict[int, int]) -> Tuple[dict, dict]:
        """post: _[0] == _[1]"""
        concrete = defaultdict(symbolic.default_factory, symbolic.items())
        return (symbolic, concrete)

    check_states(f, CANNOT_CONFIRM)


def test_defaultdict_basic_fail() -> None:
    def f(a: DefaultDict[int, int], k: int, v: int) -> None:
        """
        post[a]: a[42] != 42
        """
        a[k] = v

    check_states(f, POST_FAIL)


def test_defaultdict_default_fail(test_list) -> None:
    def f(a: DefaultDict[int, int], k: int) -> None:
        """
        pre: a.default_factory is not None
        post: a[k] <= 100
        """
        if a[k] > 100:
            del a[k]

    check_states(f, POST_FAIL)


def test_defaultdict_default_ok(test_list) -> None:
    def f(a: DefaultDict[int, int], k1: int, k2: int) -> DefaultDict[int, int]:
        """
        pre: len(a) == 0 and a.default_factory is not None
        post: _[k1] == _[k2]
        """
        return a

    check_states(f, CONFIRMED)


def test_defaultdict_realize():
    with standalone_statespace:
        with NoTracing():
            d = proxy_for_type(DefaultDict[int, int], "d")
            assert type(realize(d)) is defaultdict


#
# We don't patch namedtuple, but namedtuple performs magic dynamic type
# generation, which can interfere with CrossHair.
#


def test_namedtuple_creation():
    with standalone_statespace:
        # Ensure type creation under trace doesn't raise exception:
        Color = namedtuple("Color", ("name", "hex"))


def test_namedtuple_argument_detection_untyped():
    UntypedColor = namedtuple("UntypedColor", ("name", "hex"))
    expected_signature = Signature(
        parameters=[
            Parameter("name", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("hex", Parameter.POSITIONAL_OR_KEYWORD),
        ],
        return_annotation=Signature.empty,
    )
    assert get_constructor_signature(UntypedColor) == expected_signature


def test_namedtuple_argument_detection_typed_with_subclass():
    class ClassTypedColor(NamedTuple):
        name: str
        hex: int

    expected_parameters = {
        "name": Parameter("name", Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        "hex": Parameter("hex", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
    }
    assert get_constructor_signature(ClassTypedColor).parameters == expected_parameters


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="Functional namedtuple field types supported on Python >= 3.9",
)
def test_namedtuple_argument_detection_typed_functionally():
    FunctionallyTypedColor = NamedTuple(
        "FunctionallyTypedColor", [("name", str), ("hex", int)]
    )
    expected_parameters = {
        "name": Parameter("name", Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        "hex": Parameter("hex", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
    }
    assert (
        get_constructor_signature(FunctionallyTypedColor).parameters
        == expected_parameters
    )


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="Functional namedtuple field types supported on Python >= 3.9",
)
def test_namedtuple_symbolic_creation(space):
    UntypedColor = namedtuple("Color", "name hex")
    Color = NamedTuple("Color", [("name", str), ("hex", int)])
    untyped_color = proxy_for_type(UntypedColor, "color")
    assert isinstance(untyped_color.hex, CrossHairValue)
    color = proxy_for_type(Color, "color")
    with ResumedTracing():
        assert space.is_possible(color.hex == 5)
        assert space.is_possible(color.hex == 10)
