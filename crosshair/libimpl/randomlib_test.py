import copy
import random

from crosshair.core_and_libs import proxy_for_type, standalone_statespace
from crosshair.libimpl.randomlib import ExplicitRandom


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
