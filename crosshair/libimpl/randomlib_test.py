import copy
import random

from crosshair.core_and_libs import proxy_for_type, standalone_statespace
from crosshair.libimpl.randomlib import ExplicitRandom
from crosshair.statespace import MessageType
from crosshair.test_util import check_states


def test_ExplicitRandom():
    rng = ExplicitRandom([1, 2])
    assert rng.randrange(0, 10) == 1
    assert rng.choice(["a", "b", "c"]) == "c"
    assert rng.choice(["a", "b", "c"]) == "a"
    assert rng.random() == 0.0
    assert repr(rng) == "crosshair.libimpl.randomlib.ExplicitRandom([1, 2, 0, 0.0])"


def test_ExplicitRandom_copy():
    rng = ExplicitRandom([1, 2])
    (rng2,) = copy.deepcopy([rng])
    assert rng.randint(0, 5) == 1
    assert rng2.randint(0, 5) == 1
    assert rng.randint(0, 5) == 2
    assert rng2.randint(0, 5) == 2


def test_proxy_random():
    with standalone_statespace as space:
        rng = proxy_for_type(random.Random, "rng")
        i = rng.randrange(5, 10)
        assert space.is_possible(i.var == 5)
        assert space.is_possible(i.var == 9)
        assert not space.is_possible(i.var == 4)


def test_global_randrange():
    assert random.randrange(10, 20, 5) in (10, 15)  # confirm we've got the args right

    def f():
        """post: _ in (10, 15)"""
        return random.randrange(10, 20, 5)

    assert check_states(f) == {MessageType.CONFIRMED}


def test_global_randrange_only_upperbound():
    assert random.randrange(2) in (0, 1)  # confirm we've got the args right

    def f():
        """post: _ in (0, 1)"""
        return random.randrange(2)

    assert check_states(f) == {MessageType.CONFIRMED}


def test_global_uniform():
    assert 10.0 <= random.uniform(10, 20) <= 20.0  # confirm we've got the args right

    def f():
        """post: _ != 20.0"""
        return random.uniform(10, 20)

    assert check_states(f) == {MessageType.POST_FAIL}


def test_global_uniform_inverted_args():
    assert -2.0 <= random.uniform(10, -2) <= 10.0  # confirm we've got the args right

    def f():
        """post: -2.0 <= _ <= 10.0"""
        return random.uniform(10, -2)

    assert check_states(f) == {MessageType.CANNOT_CONFIRM}


def test_global_getrandbits():
    assert 0 <= random.getrandbits(3) < 8

    def f():
        """post: 0<= _ < 8"""
        return random.getrandbits(3)

    assert check_states(f) == {MessageType.CONFIRMED}
