from typing import List

from crosshair.statespace import MessageType
from crosshair.test_util import check_states
from crosshair.core_and_libs import standalone_statespace, NoTracing, proxy_for_type


def test_dict_index():
    a = {"two": 2, "four": 4, "six": 6}

    def numstr(x: str) -> int:
        """
        post: _ != 4
        raises: KeyError
        """
        return a[x]

    assert check_states(numstr) == {MessageType.POST_FAIL}


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
    def f(l: List[int]) -> dict:
        """
        post: 4321 not in __return__
        """
        return {i: i for i in l}

    assert check_states(f) == {MessageType.POST_FAIL}
