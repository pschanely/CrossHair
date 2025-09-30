import random

import z3  # type: ignore

from crosshair.pathing_oracle import ConstrainedOracle, PreferNegativeOracle
from crosshair.statespace import RootNode, SimpleStateSpace, WorstResultNode


def test_constrained_oracle():
    oracle = ConstrainedOracle(PreferNegativeOracle())
    x = z3.Int("x")
    root = RootNode()
    space = SimpleStateSpace()
    oracle.pre_path_hook(space)
    oracle.prefer(x >= 7)
    rand = random.Random()
    assert oracle.decide(root, WorstResultNode(rand, x < 7, space.solver), None) == 0.0
    assert oracle.decide(root, WorstResultNode(rand, x >= 3, space.solver), None) == 1.0
    assert (
        oracle.decide(root, WorstResultNode(rand, x == 7, space.solver), None) == 0.25
    )
