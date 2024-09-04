import math
from collections import defaultdict
from typing import Counter, Dict, List, Optional, Sequence, Tuple

from z3 import ExprRef  # type: ignore

from crosshair.statespace import (
    AbstractPathingOracle,
    DeatchedPathNode,
    ModelValueNode,
    NodeLike,
    RootNode,
    SearchTreeNode,
    WorstResultNode,
)
from crosshair.util import CrossHairInternal, debug, in_debug

CodeLoc = Tuple[str, ...]


class CoveragePathingOracle(AbstractPathingOracle):
    """
    A heuristic that attempts to target under-explored code locations.

    Code condition counts:
        {code pos: {condition => count}}
        Conditions should be
            a count of conditions
            that lead to the code location
            on some previously explored path

    When beginning an iteration:
        Pick a code location to target, biasing for those with few visits.
        Bias our decisions based on piror ones that led to the target location.

    Risks:
        *   Coupled decisions aren't understood, e.g. "a xor b" is challenging with underexplored
            leaves in both branches.
        *   We may target code locations that are impossible to reach because we've exhausted
            every path that leads to them. (TODO: visit frequency may not be the only appropriate
            metric for selecting target locations)
        *   Biases for now-impossible branches "bleed" over into our path.
            This sounds like a problem, but is at least partly intentional - we may hope to reach
            some of the same code locations under different early decisions.

    """

    def __init__(self):
        self.visits = Counter[CodeLoc]()
        self.iters_since_discovery = 0
        self.summarized_positions: Dict[CodeLoc, Counter[int]] = defaultdict(Counter)
        self.current_path_probabilities: Dict[ExprRef, float] = {}
        self.internalized_expressions: Tuple[Dict[ExprRef, int], Dict[int, int]] = (
            {},
            {},
        )

    # TODO: This falls apart for moderately sized with_equal_probabilities
    # because that has many small probability decisions.
    # (even just a 10% change could be much larger than it would be otherwise)
    _delta_probabilities = {-1: 0.1, 0: 0.25, 1: 0.9}

    def pre_path_hook(self, root: RootNode) -> None:
        visits = self.visits
        _delta_probabilities = self._delta_probabilities

        tweaks: Dict[ExprRef, int] = defaultdict(int)
        rand = root._random
        nondiscovery_iters = self.iters_since_discovery
        summarized_positions = self.summarized_positions
        num_positions = len(summarized_positions)
        recent_discovery = rand.random() > nondiscovery_iters / (nondiscovery_iters + 3)
        if recent_discovery or not num_positions:
            debug("No coverage biasing in effect. (", num_positions, " code locations)")
            self.current_path_probabilities = {}
            return

        options = list(summarized_positions.items())
        options.sort(key=lambda pair: visits[pair[0]])
        chosen_index = int((root._random.random() ** 2.5) * num_positions)
        (loc, exprs) = options[chosen_index]
        for expr in exprs.keys():
            if expr >= 0:
                tweaks[expr] += 1
            else:
                tweaks[-expr] -= 1
        probabilities = {
            expr: _delta_probabilities[delta] for expr, delta in tweaks.items()
        }
        if in_debug():
            debug("Coverage biasing for code location:", loc)
            debug("(", num_positions, " locations presently known)")
            expr_id_map, _ = self.internalized_expressions
            for expr, exprid in expr_id_map.items():
                probability = probabilities.get(exprid)
                if probability:
                    debug("coverage tweaked probability", expr, probability)
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
        for step, node in enumerate(path[:-1]):
            if not isinstance(node, NodeLike):
                continue
            node = node.simplify()  # type: ignore
            if isinstance(node, WorstResultNode):
                key = node.stacktail
                if (key not in leading_locs) and (not isinstance(node, ModelValueNode)):
                    self.summarized_positions[key] += Counter(leading_conditions)
                leading_locs.append(key)
                next_node = path[step + 1].simplify()
                if isinstance(next_node, DeatchedPathNode):
                    break
                if step + 1 < len(path):
                    (is_positive, root_expr) = node.normalized_expr
                    expr_signature = (
                        self.internalize(root_expr)
                        if is_positive
                        else -self.internalize(root_expr)
                    )
                    if next_node == node.positive.simplify():
                        leading_conditions.append(expr_signature)
                    elif next_node == node.negative.simplify():
                        leading_conditions.append(-expr_signature)
                    else:
                        raise CrossHairInternal(
                            f"{type(path[step])} was followed by {type(path[step+1])}"
                        )
        visits = self.visits
        prev_len = len(visits)
        visits += Counter(leading_locs)
        if len(visits) > prev_len:
            self.iters_since_discovery = 0
        else:
            self.iters_since_discovery += 1

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
