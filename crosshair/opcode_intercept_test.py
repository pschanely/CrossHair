from typing import List, Set

from crosshair.core_and_libs import NoTracing, proxy_for_type, standalone_statespace
from crosshair.statespace import POST_FAIL, MessageType
from crosshair.test_util import check_states


def test_dict_index():
    a = {"two": 2, "four": 4, "six": 6}

    def numstr(x: str) -> int:
        """
        post: _ != 4
        raises: KeyError
        """
        return a[x]

    check_states(numstr, POST_FAIL)


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
            space.add(x.var >= 40)
            space.add(x.var < 50)
        d = {k: v for k, v in ((35, 3), (x, 4))}
        with NoTracing():
            assert type(d) is not dict
        for k in d:
            if k == 35:
                continue
            with NoTracing():
                assert type(k) is not int
            assert space.is_possible((k == 43).var)
            assert space.is_possible((k == 48).var)


def test_dict_comprehension_e2e():
    def f(ls: List[int]) -> dict:
        """
        post: 4321 not in __return__
        """
        return {i: i for i in ls}

    check_states(f, POST_FAIL)


def test_set_comprehension():
    with standalone_statespace as space:
        with NoTracing():
            x = proxy_for_type(int, "x")
            space.add(x.var >= 40)
            space.add(x.var < 50)
        result_set = {k for k in (35, x)}
        with NoTracing():
            assert type(result_set) is not set
        for k in result_set:
            if k == 35:
                continue
            with NoTracing():
                assert type(k) is not int
            assert space.is_possible((k == 43).var)
            assert space.is_possible((k == 48).var)


def test_set_comprehension_e2e():
    def f(s: Set[int]) -> Set:
        """
        post: 4321 not in __return__
        """
        return {i for i in s}

    check_states(f, POST_FAIL)
