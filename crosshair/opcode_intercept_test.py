import sys
from typing import List, Set

import pytest

from crosshair.core_and_libs import NoTracing, proxy_for_type, standalone_statespace
from crosshair.statespace import POST_FAIL, MessageType
from crosshair.test_util import check_states
from crosshair.tracers import ResumedTracing
from crosshair.z3util import z3And


def test_dict_index():
    a = {"two": 2, "four": 4, "six": 6}

    def numstr(x: str) -> int:
        """
        post: _ != 4
        raises: KeyError
        """
        return a[x]

    check_states(numstr, POST_FAIL)


def test_concrete_list_with_symbolic_index_deduplicates_values(space):
    haystack = [False] * 13 + [True] + [False] * 11

    idx = proxy_for_type(int, "idx")
    with ResumedTracing():
        space.add(0 <= idx)
        space.add(idx < len(haystack))
        ret = haystack[idx]
        assert ret
        assert not space.is_possible(idx != 13)


def test_concrete_list_with_symbolic_index_unhashable_values(space):
    o1 = dict()
    options = [o1, o1, o1]
    idx = proxy_for_type(int, "idx")
    with ResumedTracing():
        space.add(0 <= idx)
        space.add(idx < 3)
        ret = options[idx]
        assert ret is o1
        assert space.is_possible(idx == 0)
        assert space.is_possible(idx == 2)


def test_dict_key_containment():
    abc = {"two": 2, "four": 4, "six": 6}

    def numstr(x: str) -> bool:
        """
        post: _
        """
        return x not in abc

    check_states(numstr, POST_FAIL)


def test_dict_comprehension():
    with standalone_statespace as space:
        with NoTracing():
            x = proxy_for_type(int, "x")
        space.add(x >= 40)
        space.add(x < 50)
        d = {k: v for k, v in ((35, 3), (x, 4))}
        with NoTracing():
            assert type(d) is not dict
        for k in d:
            if k == 35:
                continue
            with NoTracing():
                assert type(k) is not int
            assert space.is_possible(k == 43)
            assert space.is_possible(k == 48)


def test_dict_comprehension_traces_during_custom_hash():
    class FancyCompare:
        def __init__(self, mystr: str):
            self.mystr = mystr

        def __eq__(self, other):
            return (
                isinstance(other, FancyCompare)
                and "".join([self.mystr, ""]) == other.mystr
            )

        def __hash__(self):
            return hash(self.mystr)

    with standalone_statespace as space:
        with NoTracing():
            mystr = proxy_for_type(str, "mystr")
        # NOTE: If tracing isn't on when we call FancyCompare.__eq__, we'll get an
        # exception here:
        d = {x: 42 for x in [FancyCompare(mystr), FancyCompare(mystr)]}
        # There is only one item:
        assert len(d) == 1
        # TODO: In theory, we shouldn't need to realize the string here (but we are):
        # assert space.is_possible(mystr.__len__() == 0)
        # assert space.is_possible(mystr.__len__() == 1)


def test_dict_comprehension_e2e():
    def f(ls: List[int]) -> dict:
        """
        post: 4321 not in __return__
        """
        return {i: i for i in ls}

    check_states(f, POST_FAIL)


def test_not_operator_on_bool():
    with standalone_statespace as space:
        with NoTracing():
            boolval = proxy_for_type(bool, "boolval")
        inverseval = not boolval
        assert space.is_possible(inverseval)
        with NoTracing():
            assert type(inverseval) is not bool
            assert not space.is_possible(z3And(boolval.var, inverseval.var))


def test_not_operator_on_non_bool():
    with standalone_statespace as space:
        with NoTracing():
            intlist = proxy_for_type(List[int], "intlist")
        space.add(intlist.__len__() == 0)
        notList = not intlist
        with NoTracing():
            assert notList


def test_set_comprehension():
    with standalone_statespace as space:
        with NoTracing():
            x = proxy_for_type(int, "x")
        space.add(x >= 40)
        space.add(x < 50)
        result_set = {k for k in (35, x)}
        with NoTracing():
            assert type(result_set) is not set
        for k in result_set:
            if k == 35:
                continue
            with NoTracing():
                assert type(k) is not int
            assert space.is_possible(k == 43)
            assert space.is_possible(k == 48)


def test_set_comprehension_e2e():
    def f(s: Set[int]) -> Set:
        """
        post: 4321 not in __return__
        """
        return {i for i in s}

    check_states(f, POST_FAIL)


def test_trace_disabling_at_jump_targets(space):
    # This replicates a corruption of the interpreter stack in 3.12
    # under a specific bytecode layout.
    #
    # The origial issue was caused by neglecting to keep sys.monitor probes
    # alive (for post-op callbacks) that could be jumped to from other
    # locations.
    _global_type_lookupx = {
        1: 1,
        bool: 2,
        3: 3,
    }
    with ResumedTracing():
        _ = {
            k: v
            for k, v in _global_type_lookupx.items()  # <- a new line has to be here (yes, the generated bytecode differs!)
            if k == bool  # The iteration filter needs to alternate
        }


# TODO: we could implement identity comparisons on 3.8 by intercepting COMPARE_OP
@pytest.mark.skipif(sys.version_info < (3, 9), reason="IS_OP is new in Python 3.9")
def test_identity_operator_on_booleans():
    with standalone_statespace as space:
        with NoTracing():
            b1 = proxy_for_type(bool, "b1")
        space.add(b1)
        assert b1 is True
