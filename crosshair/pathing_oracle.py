import math
from collections import defaultdict
from typing import Counter, Dict, List, Optional, Sequence, Tuple

from z3 import ExprRef  # type: ignore

from crosshair.statespace import (
    AbstractPathingOracle,
    ModelValueNode,
    NodeLike,
    RootNode,
    SearchTreeNode,
    WorstResultNode,
)
from crosshair.util import CrosshairInternal, debug, in_debug


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


CodeLoc = Tuple[str, ...]


class CoveragePathingOracle(AbstractPathingOracle):
    """
    A heuristic that attempts to target under-explored code locations.

    Code condition counts:
        {code pos: (visit count, {condition => count})}
        Conditions should be
            a count of conditions
            along non-exhaused paths
            that lead to the code position

    At root:
        Maintain a summarization level, "N"
        Summarize code condition counts to level N.
        If smallest visit count > DELVE_THRESHOLD, increment N and restart.

        Calculate a per-condition probability based on counts of positive and negative visits.
        Divide the weight on each condition by the # of visits.




    Risks:
        *   Coupled decisions aren't understood, e.g. "a xor b" is challenging with underexplored
            leaves in both branches
        *   Biases for now-impossible branches "bleed" over into our path.
            This sounds like a problem, but is at least partly intentional - we may hope to reach
            some of the same code locations under different early decisions.

    """

    def __init__(self):
        self.positions: Dict[CodeLoc, Tuple[int, Counter[ExprRef]]] = {}
        self.summarized_positions: Dict[CodeLoc, Counter[int]] = defaultdict(Counter)
        self.current_path_probabilities: Dict[ExprRef, float] = {}
        self.position_granularity = 10
        self.internalized_expressions: Tuple[Dict[ExprRef, int], Dict[int, int]] = (
            {},
            {},
        )

    _delta_probabilities = {-1: 0.1, 0: 0.25, 1: 0.9}

    def pre_path_hook(self, root: RootNode) -> None:
        _delta_probabilities = self._delta_probabilities

        tweaks: Dict[ExprRef, int] = defaultdict(int)

        loc_entries = list(self.summarized_positions.items())
        if loc_entries:
            (loc, exprs) = root._random.choice(loc_entries)
            debug("code loc", loc)
            for expr in exprs.keys():
                if expr >= 0:
                    tweaks[expr] += 1
                else:
                    tweaks[-expr] -= 1

        probabilities = {
            expr: _delta_probabilities[delta] for expr, delta in tweaks.items()
        }
        if in_debug():
            for e, t in probabilities.items():
                debug("coverage tweaked probability", e, t)
        self.current_path_probabilities = probabilities

    def internalize(self, expr):
        expr_id, id_id = self.internalized_expressions
        myid = id(expr)
        unified_id = id_id.get(myid)
        if unified_id:
            return unified_id
        unified_id = expr_id.get(expr)
        if unified_id:
            id_id[myid] = unified_id
            return unified_id
        expr_id[expr] = myid
        id_id[myid] = myid
        return myid

    def post_path_hook(self, path: Sequence[SearchTreeNode]) -> None:
        leading_locs = []
        leading_conditions: List[int] = []
        for step, node in enumerate(path):
            if not isinstance(node, NodeLike):
                continue
            node = node.simplify()  # type: ignore
            if isinstance(node, WorstResultNode):
                key = tuple(node.stacktail[-self.position_granularity :])

                if (key not in leading_locs) and (not isinstance(node, ModelValueNode)):
                    self.summarized_positions[key] += Counter(leading_conditions)
                leading_locs.append(key)
                if step + 1 < len(path):
                    (is_positive, root_expr) = node.normalized_expr
                    expr_signature = (
                        self.internalize(root_expr)
                        if is_positive
                        else -self.internalize(root_expr)
                    )
                    if path[step + 1].simplify() == node.positive.simplify():
                        leading_conditions.append(expr_signature)
                    elif path[step + 1].simplify() == node.negative.simplify():
                        leading_conditions.append(-expr_signature)
                    else:
                        raise CrosshairInternal(
                            f"{type(path[step])}{type(path[step+1])}"
                        )

    def decide(
        self,
        root: RootNode,
        node: "WorstResultNode",
        engine_probability: Optional[float],
    ) -> float:
        if engine_probability in (0.0, 1.0):  # is not None:
            return engine_probability
        default_probability = 0.25

        path_probabilities = self.current_path_probabilities
        is_positive, n_expr = node.normalized_expr
        n_expr_id = self.internalize(n_expr)
        if n_expr_id not in path_probabilities:
            return (
                default_probability
                if engine_probability is None
                else engine_probability
            )
        true_probability = path_probabilities[n_expr_id]
        return true_probability if is_positive else 1.0 - true_probability


class BreadthFirstPathingOracle(AbstractPathingOracle):
    def decide(
        self,
        root: RootNode,
        node: "WorstResultNode",
        engine_probability: Optional[float],
    ) -> float:
        branch_counter = root._open_coverage[node.stacktail]

        # If we've never taken a branch at this code location, make sure we try it!
        if bool(branch_counter.pos_ct) != bool(branch_counter.neg_ct):
            if engine_probability != 0.0 and engine_probability != 1.0:
                return 1.0 if branch_counter.neg_ct else 0.0
        if engine_probability is None:
            engine_probability = 0.25
        if engine_probability != 0.0 and engine_probability != 1.0:
            if branch_counter.pos_ct > branch_counter.neg_ct * 2 + 1:
                engine_probability /= 2.0
            elif branch_counter.neg_ct > branch_counter.pos_ct * 2 + 1:
                engine_probability = (1.0 + engine_probability) / 2.0
        return engine_probability


class PreferNegativeOracle(AbstractPathingOracle):
    def decide(
        self,
        root: RootNode,
        node: "WorstResultNode",
        engine_probability: Optional[float],
    ) -> float:
        # When both paths are unexplored, we bias for False.
        # As a heuristic, this tends to prefer early completion:
        # - Loop conditions tend to repeat on True.
        # - Optional[X] turns into Union[X, None] and False conditions
        #   biases for the last item in the union.
        # We pick a False value more than 2/3rds of the time to avoid
        # explosions while constructing binary-tree-like objects.
        if engine_probability is not None:
            return engine_probability
        return 0.25


class RotatingOracle(AbstractPathingOracle):
    def __init__(self, oracles: List[AbstractPathingOracle]):
        self.oracles = oracles
        self.index = -1

    def pre_path_hook(self, root: "RootNode") -> None:
        oracles = self.oracles
        self.index = (self.index + 1) % len(oracles)
        for oracle in oracles:
            oracle.pre_path_hook(root)

    def post_path_hook(self, path: Sequence["SearchTreeNode"]) -> None:
        for oracle in self.oracles:
            oracle.post_path_hook(path)

    def decide(
        self, root, node: "WorstResultNode", engine_probability: Optional[float]
    ) -> float:
        return self.oracles[self.index].decide(root, node, engine_probability)
