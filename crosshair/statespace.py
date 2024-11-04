import ast
import builtins
import copy
import enum
import functools
import random
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from sys import _getframe
from time import monotonic
from traceback import extract_stack, format_tb
from types import FrameType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NewType,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
)

import z3  # type: ignore

from crosshair import dynamic_typing
from crosshair.condition_parser import ConditionExpr
from crosshair.smtlib import parse_smtlib_literal
from crosshair.tracers import NoTracing, ResumedTracing, is_tracing
from crosshair.util import (
    CROSSHAIR_EXTRA_ASSERTS,
    CrossHairInternal,
    IgnoreAttempt,
    NotDeterministic,
    PathTimeout,
    UnknownSatisfiability,
    assert_tracing,
    ch_stack,
    debug,
    in_debug,
    name_of_type,
)
from crosshair.z3util import z3Aassert, z3Not, z3PopNot


@functools.total_ordering
class MessageType(enum.Enum):
    CONFIRMED = "confirmed"
    # The postcondition returns True over all execution paths.

    CANNOT_CONFIRM = "cannot_confirm"
    # The postcondition returns True over the execution paths that were
    # attempted.

    PRE_UNSAT = "pre_unsat"
    # No attempted execution path got past the precondition checks.

    POST_ERR = "post_err"
    # The postcondition raised an exception for some input.

    EXEC_ERR = "exec_err"
    # The body of the function raised an exception for some input.

    POST_FAIL = "post_fail"
    # The postcondition returned False for some input.

    SYNTAX_ERR = "syntax_err"
    # Pre/post conditions could not be determined.

    IMPORT_ERR = "import_err"
    # The requested module could not be imported.

    def __repr__(self):
        return f"MessageType.{self.name}"

    def __lt__(self, other):
        return self._order[self] < self._order[other]


MessageType._order = {  # type: ignore
    # This is the order that messages override each other (for the same source
    # file line).
    # For exmaple, we prefer to report a False-returning postcondition
    # (POST_FAIL) over an exception-raising postcondition (POST_ERR).
    MessageType.CONFIRMED: 0,
    MessageType.CANNOT_CONFIRM: 1,
    MessageType.PRE_UNSAT: 2,
    MessageType.POST_ERR: 3,
    MessageType.EXEC_ERR: 4,
    MessageType.POST_FAIL: 5,
    MessageType.SYNTAX_ERR: 6,
    MessageType.IMPORT_ERR: 7,
}


CONFIRMED = MessageType.CONFIRMED
CANNOT_CONFIRM = MessageType.CANNOT_CONFIRM
PRE_UNSAT = MessageType.PRE_UNSAT
POST_ERR = MessageType.POST_ERR
EXEC_ERR = MessageType.EXEC_ERR
POST_FAIL = MessageType.POST_FAIL
SYNTAX_ERR = MessageType.SYNTAX_ERR
IMPORT_ERR = MessageType.IMPORT_ERR


@dataclass(frozen=True)
class AnalysisMessage:
    state: MessageType
    message: str
    filename: str
    line: int
    column: int
    traceback: str
    test_fn: Optional[str] = None
    condition_src: Optional[str] = None


@functools.total_ordering
class VerificationStatus(enum.Enum):
    REFUTED = 0
    UNKNOWN = 1
    CONFIRMED = 2

    def __repr__(self):
        return f"VerificationStatus.{self.name}"

    def __str__(self):
        return self.name

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


@dataclass
class CallAnalysis:
    verification_status: Optional[VerificationStatus] = None  # None means "ignore"
    messages: Sequence[AnalysisMessage] = ()
    failing_precondition: Optional[ConditionExpr] = None
    failing_precondition_reason: str = ""


HeapRef = z3.DeclareSort("HeapRef")
SnapshotRef = NewType("SnapshotRef", int)


def model_value_to_python(value: z3.ExprRef) -> object:
    if z3.is_real(value):
        if isinstance(value, z3.AlgebraicNumRef):
            # Force irrational values to be rational:
            value = value.approx(precision=20)
        return float(value.as_fraction())
    elif z3.is_seq(value):
        ret = []
        while value.num_args() == 2:
            ret.append(model_value_to_python(value.arg(0).arg(0)))
            value = value.arg(1)
        if value.num_args() == 1:
            ret.append(model_value_to_python(value.arg(0)))
        return ret
    elif z3.is_int(value):
        return value.as_long()
    elif z3.is_fp(value):
        return parse_smtlib_literal(value.sexpr())
    else:
        return ast.literal_eval(repr(value))


@assert_tracing(False)
def prefer_true(v: Any) -> bool:
    if not (hasattr(v, "var") and z3.is_bool(v.var)):
        with ResumedTracing():
            v = v.__bool__()
        if not (hasattr(v, "var")):
            return v
    space = context_statespace()
    return space.choose_possible(v.var, probability_true=1.0)


def force_true(v: Any) -> None:
    with NoTracing():
        if not (hasattr(v, "var") and z3.is_bool(v.var)):
            with ResumedTracing():
                v = v.__bool__()
            if not (hasattr(v, "var")):
                raise CrossHairInternal(
                    "Attempted to call assert_true on a non-symbolic"
                )
        space = context_statespace()
        # TODO: we can improve this by making a new kind of (unary) assertion node
        # that would not create these useless forks when the space is near exhaustion.
        if not space.choose_possible(v.var, probability_true=1.0):
            raise IgnoreAttempt


class StateSpaceCounter(Counter):
    @property
    def iterations(self) -> int:
        return sum(self[s] for s in VerificationStatus) + self[None]

    @property
    def unknown_pct(self) -> float:
        return self[VerificationStatus.UNKNOWN] / (self.iterations + 1)

    def __str__(self) -> str:
        parts = []
        for k, ct in self.items():
            if isinstance(k, enum.Enum):
                k = k.name
            parts.append(f"{k}:{ct}")
        return "{" + ", ".join(parts) + "}"


class AbstractPathingOracle:
    def pre_path_hook(self, root: "RootNode") -> None:
        pass

    def post_path_hook(self, path: Sequence["SearchTreeNode"]) -> None:
        pass

    def decide(
        self, root, node: "WorstResultNode", engine_probability: Optional[float]
    ) -> float:
        raise NotImplementedError


# NOTE: CrossHair's monkey-patched getattr calls this function, so we
# force ourselves to use the builtin getattr, avoiding an infinite loop.
real_getattr = builtins.getattr

_THREAD_LOCALS = threading.local()
_THREAD_LOCALS.space = None


class StateSpaceContext:
    def __init__(self, space: "StateSpace"):
        self.space = space

    def __enter__(self):
        self.space._stack_depth_of_context_entry = len(extract_stack())
        prev = real_getattr(_THREAD_LOCALS, "space", None)
        if prev is not None:
            raise CrossHairInternal("Already in a state space context")
        space = self.space
        _THREAD_LOCALS.space = space
        space.mark_all_parent_frames()

    def __exit__(self, exc_type, exc_value, tb):
        prev = real_getattr(_THREAD_LOCALS, "space", None)
        if prev is not self.space:
            raise CrossHairInternal("State space was altered in context")
        _THREAD_LOCALS.space = None
        self.space._stack_depth_of_context_entry = None
        return False


def optional_context_statespace() -> Optional["StateSpace"]:
    return _THREAD_LOCALS.space


def context_statespace() -> "StateSpace":
    space = _THREAD_LOCALS.space
    if space is None:
        raise CrossHairInternal("Not in a statespace context")
    return space


def newrandom():
    return random.Random(1801243388510242075)


_N = TypeVar("_N", bound="SearchTreeNode")
_T = TypeVar("_T")


class NodeLike:
    def is_exhausted(self) -> bool:
        return False

    def get_result(self) -> CallAnalysis:
        """
        Get the result from the call.

        post: implies(_.verification_status == VerificationStatus.CONFIRMED, self.is_exhausted())
        """
        raise NotImplementedError

    def is_stem(self) -> bool:
        return False

    def grow_into(self, node: _N) -> _N:
        raise NotImplementedError

    def simplify(self) -> "NodeLike":
        return self

    def stats(self) -> StateSpaceCounter:
        raise NotImplementedError


class NodeStem(NodeLike):
    evolution: Optional["SearchTreeNode"] = None

    def is_exhausted(self) -> bool:
        return False if self.evolution is None else self.evolution.is_exhausted()

    def get_result(self) -> CallAnalysis:
        return (
            CallAnalysis(VerificationStatus.UNKNOWN)
            if self.evolution is None
            else self.evolution.get_result()
        )

    def is_stem(self) -> bool:
        return self.evolution is None

    def grow_into(self, node: _N) -> _N:
        self.evolution = node
        return node

    def simplify(self):
        return self if self.evolution is None else self.evolution

    def stats(self) -> StateSpaceCounter:
        return StateSpaceCounter() if self.evolution is None else self.evolution.stats()

    def __repr__(self) -> str:
        return "NodeStem()"


class SearchTreeNode(NodeLike):
    """A node in the execution path tree."""

    stacktail: Tuple[str, ...] = ()
    result: CallAnalysis = CallAnalysis()
    exhausted: bool = False
    iteration: Optional[int] = None

    def choose(
        self, space: "StateSpace", probability_true: Optional[float] = None
    ) -> Tuple[bool, float, NodeLike]:
        raise NotImplementedError

    def is_exhausted(self) -> bool:
        return self.exhausted

    def get_result(self) -> CallAnalysis:
        return self.result

    def update_result(self, leaf_analysis: CallAnalysis) -> bool:
        if not self.exhausted:
            next_result, next_exhausted = self.compute_result(leaf_analysis)
            if next_exhausted != self.exhausted or next_result != self.result:
                self.result, self.exhausted = next_result, next_exhausted
                return True
        return False

    def compute_result(self, leaf_analysis: CallAnalysis) -> Tuple[CallAnalysis, bool]:
        raise NotImplementedError


def solver_is_sat(solver, *exprs) -> bool:
    ret = solver.check(*exprs)
    if ret == z3.unknown:
        debug("Z3 Unknown satisfiability. Reason:", solver.reason_unknown())
        if solver.reason_unknown() == "interrupted from keyboard":
            raise KeyboardInterrupt
        if exprs:
            debug("While attempting to assert\n", *(e.sexpr() for e in exprs))
        debug("Solver state follows:\n", solver.sexpr())
        raise UnknownSatisfiability
    return ret == z3.sat


def node_result(node: Optional[NodeLike]) -> Optional[CallAnalysis]:
    if node is None:
        return None
    return node.get_result()


def node_status(node: Optional[NodeLike]) -> Optional[VerificationStatus]:
    result = node_result(node)
    if result is not None:
        return result.verification_status
    else:
        return None


class SearchLeaf(SearchTreeNode):
    def __init__(self, result: CallAnalysis):
        self.result = result
        self.exhausted = True
        self._stats = StateSpaceCounter({result.verification_status: 1})

    def stats(self) -> StateSpaceCounter:
        return self._stats

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.result.verification_status})"


class SinglePathNode(SearchTreeNode):
    decision: bool
    child: NodeLike
    _random: random.Random

    def __init__(self, decision: bool):
        self.decision = decision
        self.child = NodeStem()
        self._random = newrandom()

    def choose(
        self, space: "StateSpace", probability_true: Optional[float] = None
    ) -> Tuple[bool, float, NodeLike]:
        return (self.decision, 1.0, self.child)

    def compute_result(self, leaf_analysis: CallAnalysis) -> Tuple[CallAnalysis, bool]:
        self.child = self.child.simplify()
        return (self.child.get_result(), self.child.is_exhausted())

    def stats(self) -> StateSpaceCounter:
        return self.child.stats()


class BranchCounter:
    __slots__ = ["pos_ct", "neg_ct"]
    pos_ct: int
    neg_ct: int

    def __init__(self):
        self.pos_ct = 0
        self.neg_ct = 0


class RootNode(SinglePathNode):
    def __init__(self):
        super().__init__(True)
        self._open_coverage: Dict[Tuple[str, ...], BranchCounter] = defaultdict(
            BranchCounter
        )
        from crosshair.pathing_oracle import CoveragePathingOracle  # circular import

        self.pathing_oracle = CoveragePathingOracle()
        self.iteration = 0


class DeatchedPathNode(SinglePathNode):
    def __init__(self):
        super().__init__(True)
        # Seems like `exhausted` should be True, but we set to False until we can
        # collect the result from path's leaf. (exhaustion prevents caches from
        # updating)
        self.exhausted = False
        self._stats = None

    def compute_result(self, leaf_analysis: CallAnalysis) -> Tuple[CallAnalysis, bool]:
        self.child = self.child.simplify()
        return (leaf_analysis, True)

    def stats(self) -> StateSpaceCounter:
        if self._stats is None:
            self._stats = StateSpaceCounter(
                {
                    k: v
                    for k, v in self.child.stats().items()
                    # We only propagate the verification status.
                    # (we should mostly look like a SearchLeaf)
                    if isinstance(k, VerificationStatus)
                }
            )
        return self._stats


class BinaryPathNode(SearchTreeNode):
    positive: NodeLike
    negative: NodeLike

    def __init__(self):
        self._stats = StateSpaceCounter()

    def stats_lookahead(self) -> Tuple[StateSpaceCounter, StateSpaceCounter]:
        return (self.positive.stats(), self.negative.stats())

    def stats(self) -> StateSpaceCounter:
        return self._stats


class RandomizedBinaryPathNode(BinaryPathNode):
    def __init__(self, rand: random.Random):
        super().__init__()
        self._random = rand
        self.positive = NodeStem()
        self.negative = NodeStem()

    def probability_true(
        self, space: "StateSpace", requested_probability: Optional[float] = None
    ) -> float:
        raise NotImplementedError

    def choose(
        self, space: "StateSpace", probability_true: Optional[float] = None
    ) -> Tuple[bool, float, NodeLike]:
        positive_ok = not self.positive.is_exhausted()
        negative_ok = not self.negative.is_exhausted()
        assert positive_ok or negative_ok
        if positive_ok and negative_ok:
            probability_true = self.probability_true(
                space, requested_probability=probability_true
            )
            randval = self._random.uniform(0.000_001, 0.999_999)
            if randval < probability_true:
                return (True, probability_true, self.positive)
            else:
                return (False, 1.0 - probability_true, self.negative)
        else:
            return (positive_ok, 1.0, self.positive if positive_ok else self.negative)

    def _simplify(self) -> None:
        self.positive = self.positive.simplify()
        self.negative = self.negative.simplify()


class ParallelNode(RandomizedBinaryPathNode):
    """Choose either path; the first complete result will be used."""

    def __init__(self, rand: random.Random, false_probability: float, desc: str):
        super().__init__(rand)
        self._false_probability = false_probability
        self._desc = desc

    def __repr__(self):
        return f"ParallelNode(false_pct={self._false_probability}, {self._desc})"

    def compute_result(self, leaf_analysis: CallAnalysis) -> Tuple[CallAnalysis, bool]:
        self._simplify()
        positive, negative = self.positive, self.negative
        pos_exhausted = positive.is_exhausted()
        neg_exhausted = negative.is_exhausted()
        if pos_exhausted and not node_status(positive) == VerificationStatus.UNKNOWN:
            self._stats = positive.stats()
            return (positive.get_result(), True)
        if neg_exhausted and not node_status(negative) == VerificationStatus.UNKNOWN:
            self._stats = negative.stats()
            return (negative.get_result(), True)
        # it's unclear whether we want to just add stats here:
        self._stats = StateSpaceCounter(positive.stats() + negative.stats())
        return merge_node_results(
            positive.get_result(),
            pos_exhausted and neg_exhausted,
            negative,
        )

    def probability_true(
        self, space: "StateSpace", requested_probability: Optional[float] = None
    ) -> float:
        if self.negative.is_exhausted():
            return 1.0
        elif requested_probability is not None:
            return requested_probability
        else:
            return 1.0 - self._false_probability


def merge_node_results(
    left: CallAnalysis, exhausted: bool, node: NodeLike
) -> Tuple[CallAnalysis, bool]:
    """
    Merge analysis from different branches of code.

    Combines messages, take the worst verification status of the two, etc.
    """
    right = node.get_result()
    if not node.is_exhausted():
        exhausted = False
    if left.verification_status is None:
        return (right, exhausted)
    if right.verification_status is None:
        return (left, exhausted)
    if left.failing_precondition and right.failing_precondition:
        lc, rc = left.failing_precondition, right.failing_precondition
        precond_side = left if lc.line > rc.line else right
    else:
        precond_side = left if left.failing_precondition else right
    return (
        CallAnalysis(
            min(left.verification_status, right.verification_status),
            list(left.messages) + list(right.messages),
            precond_side.failing_precondition,
            precond_side.failing_precondition_reason,
        ),
        exhausted,
    )


_RE_WHITESPACE_SUB = re.compile(r"\s+").sub


class WorstResultNode(RandomizedBinaryPathNode):
    forced_path: Optional[bool] = None
    expr: Optional[z3.ExprRef] = None
    normalized_expr: Tuple[bool, z3.ExprRef]

    def __init__(self, rand: random.Random, expr: z3.ExprRef, solver: z3.Solver):
        super().__init__(rand)
        is_positive, root_expr = z3PopNot(expr)
        self.normalized_expr = (is_positive, root_expr)
        notexpr = z3Not(expr) if is_positive else root_expr
        if solver_is_sat(solver, notexpr):
            if not solver_is_sat(solver, expr):
                self.forced_path = False
        else:
            # We run into soundness issues on occasion:
            if CROSSHAIR_EXTRA_ASSERTS and not solver_is_sat(solver, expr):
                debug(" *** Reached impossible code path *** ")
                debug("Current solver state:\n", str(solver))
                raise CrossHairInternal("Reached impossible code path")
            self.forced_path = True
        self.expr = expr

    def _is_exhausted(self):
        return (
            (self.positive.is_exhausted() and self.negative.is_exhausted())
            or (self.forced_path is True and self.positive.is_exhausted())
            or (self.forced_path is False and self.negative.is_exhausted())
        )

    def __repr__(self):
        smt_expr = _RE_WHITESPACE_SUB(" ", str(self.expr))
        exhausted = " : exhausted" if self._is_exhausted() else ""
        forced = f" : force={self.forced_path}" if self.forced_path is not None else ""
        return f"{self.__class__.__name__}({smt_expr}{exhausted}{forced})"

    def choose(
        self, space: "StateSpace", probability_true: Optional[float] = None
    ) -> Tuple[bool, float, NodeLike]:
        if self.forced_path is None:
            return RandomizedBinaryPathNode.choose(self, space, probability_true)
        return (
            self.forced_path,
            1.0,
            self.positive if self.forced_path else self.negative,
        )

    def probability_true(
        self, space: "StateSpace", requested_probability: Optional[float] = None
    ) -> float:
        root = space._root
        return root.pathing_oracle.decide(
            root, self, engine_probability=requested_probability
        )

    def compute_result(self, leaf_analysis: CallAnalysis) -> Tuple[CallAnalysis, bool]:
        self._simplify()
        positive, negative = self.positive, self.negative
        exhausted = self._is_exhausted()
        if node_status(positive) == VerificationStatus.REFUTED or (
            self.forced_path is True
        ):
            self._stats = positive.stats()
            return (positive.get_result(), exhausted)
        if node_status(negative) == VerificationStatus.REFUTED or (
            self.forced_path is False
        ):
            self._stats = negative.stats()
            return (negative.get_result(), exhausted)
        self._stats = StateSpaceCounter(positive.stats() + negative.stats())
        return merge_node_results(
            positive.get_result(), positive.is_exhausted(), negative
        )


class ModelValueNode(WorstResultNode):
    condition_value: object = None

    def __init__(self, rand: random.Random, expr: z3.ExprRef, solver: z3.Solver):
        if not solver_is_sat(solver):
            debug("Solver unexpectedly unsat; solver state:", solver.sexpr())
            raise CrossHairInternal("Unexpected unsat from solver")

        self.condition_value = solver.model().evaluate(expr, model_completion=True)
        self._stats_key = f"realize_{expr}" if z3.is_const(expr) else None
        WorstResultNode.__init__(self, rand, expr == self.condition_value, solver)

    def compute_result(self, leaf_analysis: CallAnalysis) -> Tuple[CallAnalysis, bool]:
        stats_key = self._stats_key
        old_realizations = self._stats[stats_key]
        analysis, is_exhausted = super().compute_result(leaf_analysis)
        if stats_key:
            self._stats[stats_key] = old_realizations + 1
        return (analysis, is_exhausted)


def debug_path_tree(node, highlights, prefix="") -> List[str]:
    highlighted = node in highlights
    node = node.simplify()
    highlighted |= node in highlights
    if isinstance(node, BinaryPathNode):
        if isinstance(node, WorstResultNode) and node.forced_path is not None:
            return debug_path_tree(
                node.positive if node.forced_path else node.negative, highlights, prefix
            )
        lines = []
        forkstr = r"n|\ y  " if isinstance(node, WorstResultNode) else r" |\    "
        if highlighted:
            lines.append(f"{prefix}{forkstr}*{str(node)} {node.stats()}")
        else:
            lines.append(f"{prefix}{forkstr}{str(node)} {node.stats()}")
        if node.is_exhausted() and not highlighted:
            return lines  # collapse fully explored subtrees
        lines.extend(debug_path_tree(node.positive, highlights, prefix + " | "))
        lines.extend(debug_path_tree(node.negative, highlights, prefix))
        return lines
    elif isinstance(node, SinglePathNode):
        lines = []
        if highlighted:
            lines.append(f"{prefix} |     *{type(node).__name__} {node.stats()}")
        else:
            lines.append(f"{prefix} |     {type(node).__name__} {node.stats()}")
        lines.extend(debug_path_tree(node.child, highlights, prefix))
        return lines
    else:
        if highlighted:
            return [f"{prefix} -> *{str(node)} {node.stats()}"]
        else:
            return [f"{prefix} -> {str(node)} {node.stats()}"]


class StateSpace:
    """Holds various information about the SMT solver's current state."""

    _search_position: NodeLike
    _deferred_assumptions: List[Tuple[str, Callable[[], bool]]]
    _extras: Dict[Type, object]

    def __init__(
        self,
        execution_deadline: float,
        model_check_timeout: float,
        search_root: RootNode,
    ):
        smt_timeout = model_check_timeout * 1000 + 1
        smt_tactic = z3.Tactic("smt")
        if smt_timeout < 1 << 63:
            smt_tactic = z3.TryFor(smt_tactic, int(smt_timeout))
        self.solver = smt_tactic.solver()
        self.solver.set(mbqi=True)
        # turn off every randomization thing we can think of:
        self.solver.set("random-seed", 42)
        self.solver.set("smt.random-seed", 42)
        # self.solver.set('randomize', False)
        self.choices_made: List[SearchTreeNode] = []
        self.status_cap: Optional[VerificationStatus] = None
        self.heaps: List[List[Tuple[z3.ExprRef, Type, object]]] = [[]]
        self.next_uniq = 1
        self.is_detached = False
        self._extras = {}
        self._already_logged: Set[z3.ExprRef] = set()
        self._exprs_known: Dict[z3.ExprRef, bool] = {}
        self._stack_depth_of_context_entry: Optional[int] = None

        self.execution_deadline = execution_deadline
        self._root = search_root
        self._random = search_root._random
        _, _, self._search_position = search_root.choose(self)
        self._deferred_assumptions = []
        assert search_root.iteration is not None
        search_root.iteration += 1
        search_root.pathing_oracle.pre_path_hook(search_root)

    def add(self, expr) -> None:
        with NoTracing():
            if hasattr(expr, "var"):
                expr = expr.var
            elif not isinstance(expr, z3.ExprRef):
                if type(expr) is bool:
                    raise CrossHairInternal(
                        "Attempted to assert a concrete boolean (look for unexpected realization)"
                    )
                raise CrossHairInternal(
                    "Expected symbolic boolean, but supplied expression of type",
                    name_of_type(type(expr)),
                )
            # debug('Committed to ', expr)
            already_known = self._exprs_known.get(expr)
            if already_known is None:
                self.solver.add(expr)
                self._exprs_known[expr] = True
            elif already_known is not True:
                raise CrossHairInternal

    def rand(self) -> random.Random:
        return self._random

    def extra(self, typ: Type[_T]) -> _T:
        """Get an object whose lifetime is tied to that of the SMT solver."""
        value = self._extras.get(typ)
        if value is None:
            value = typ(self.solver)  # type: ignore
            self._extras[typ] = value
        return value  # type: ignore

    def stats_lookahead(self) -> Tuple[StateSpaceCounter, StateSpaceCounter]:
        node = self._search_position.simplify()
        if node.is_stem():
            return (StateSpaceCounter(), StateSpaceCounter())
        assert isinstance(
            node, BinaryPathNode
        ), f"node {node} {node.is_stem()} is not a binarypathnode"
        return node.stats_lookahead()

    def fork_parallel(self, false_probability: float, desc: str = "") -> bool:
        if self._search_position.is_stem():
            node: NodeLike = self._search_position.grow_into(
                ParallelNode(self._random, false_probability, desc)
            )
            assert isinstance(node, ParallelNode)
            self._search_position = node
        else:
            node = self._search_position.simplify()
            if not isinstance(node, ParallelNode):
                self.raise_not_deterministic(
                    node, "Wrong node type (expected ParallelNode)"
                )
            node._false_probability = false_probability
        self.choices_made.append(node)
        ret, _, next_node = node.choose(self)
        self._search_position = next_node
        return ret

    def is_possible(self, expr) -> bool:
        with NoTracing():
            if hasattr(expr, "var"):
                expr = expr.var
        return solver_is_sat(self.solver, expr)

    def mark_all_parent_frames(self):
        frames: Set[FrameType] = set()
        frame = _getframe()
        while frame and frame not in frames:
            frames.add(frame)
            frame = frame.f_back
        self.external_frames = (
            frames  # just to prevent dealllocation and keep the id()s valid
        )
        self.external_frame_ids = {id(f) for f in frames}

    def gen_stack_descriptions(self) -> Tuple[str, ...]:
        f: Any = _getframe().f_back.f_back  # type: ignore
        frames = [f := f.f_back or f for _ in range(8)]
        # TODO: To help oracles, I'd like to add sub-line resolution via f.f_lasti;
        # however, in Python >= 3.11, the instruction pointer can shift between
        # PRECALL and CALL opcodes, triggering our nondeterminism check.
        return tuple(f"{f.f_code.co_filename}:{f.f_lineno}" for f in frames)

    def check_timeout(self):
        if monotonic() > self.execution_deadline:
            debug(
                "Path execution timeout after making ",
                len(self.choices_made),
                " choices.",
            )
            raise PathTimeout

    @assert_tracing(False)
    def choose_possible(
        self, expr: z3.ExprRef, probability_true: Optional[float] = None
    ) -> bool:
        known_result = self._exprs_known.get(expr)
        if isinstance(known_result, bool):
            return known_result
        # NOTE: format_stack() is more human readable, but it pulls source file contents,
        # so it is (1) slow, and (2) unstable when source code changes while we are checking.
        stacktail = self.gen_stack_descriptions()
        if self._search_position.is_stem():
            # We only allow time outs at stems - that's because we don't want
            # to think about how mutating an existing path branch would work:
            self.check_timeout()
            node = self._search_position.grow_into(
                WorstResultNode(self._random, expr, self.solver)
            )
            node.iteration = self._root.iteration
            node.stacktail = stacktail
        else:
            node = self._search_position.simplify()  # type: ignore
            not_deterministic_reason = (
                (
                    (not isinstance(node, WorstResultNode))
                    and "Wrong node type (expected WorstResultNode)"
                )
                or (node.stacktail != stacktail and "Stack trace changed")
                or ((not z3.eq(node.expr, expr)) and "SMT expression changed")
            )
            if not_deterministic_reason:
                self.raise_not_deterministic(
                    node, not_deterministic_reason, expr=expr, stacktail=stacktail
                )
        self._search_position = node
        choose_true, chosen_probability, stem = node.choose(
            self, probability_true=probability_true
        )

        branch_counter = self._root._open_coverage[stacktail]
        if choose_true:
            branch_counter.pos_ct += 1
        else:
            branch_counter.neg_ct += 1

        self.choices_made.append(node)
        self._search_position = stem
        chosen_expr = expr if choose_true else z3Not(expr)
        if in_debug():
            debug(
                f"SMT chose: {chosen_expr} (chance: {chosen_probability}) at",
                ch_stack(),
            )
        z3Aassert(self.solver, chosen_expr)
        self._exprs_known[expr] = choose_true
        return choose_true

    def raise_not_deterministic(
        self,
        node: NodeLike,
        reason: str,
        expr: Optional[z3.ExprRef] = None,
        stacktail: Optional[Tuple[str, ...]] = None,
        currently_handling: Optional[BaseException] = None,
    ) -> NoReturn:
        lines = ["*** Begin Not Deterministic Debug ***"]
        if hasattr(node, "iteration"):
            lines.append(f"Previous iteration: {node.iteration}")
        if hasattr(node, "expr"):
            lines.append(f"Previous SMT expression: {node.expr}")
        if expr is not None:
            lines.append(f"Current SMT expression: {expr}")
        if not stacktail:
            if currently_handling is not None:
                stacktail = tuple(format_tb(currently_handling.__traceback__))
            else:
                stacktail = self.gen_stack_descriptions()
        lines.append("Current stack tail:")
        lines.extend(f"  {x}" for x in stacktail)
        if hasattr(node, "stacktail"):
            lines.append("Previous stack tail:")
            lines.extend(f"  {x}" for x in node.stacktail)
        lines.append(f"Reason: {reason}")
        lines.append("*** End Not Deterministic Debug ***")
        for line in lines:
            print(line)
        exc = NotDeterministic()
        if currently_handling:
            raise exc from currently_handling
        else:
            raise exc

    def find_model_value(self, expr: z3.ExprRef) -> Any:
        with NoTracing():
            while True:
                if self._search_position.is_stem():
                    self._search_position = self._search_position.grow_into(
                        ModelValueNode(self._random, expr, self.solver)
                    )
                node = self._search_position.simplify()
                if isinstance(node, SearchLeaf):
                    raise CrossHairInternal(
                        f"Cannot use symbolics; path is already terminated"
                    )
                if not isinstance(node, ModelValueNode):
                    debug(" *** Begin Not Deterministic Debug *** ")
                    debug(f"Model value node expected; found {type(node)} instead.")
                    debug("  Traceback: ", ch_stack())
                    debug(" *** End Not Deterministic Debug *** ")
                    raise NotDeterministic
                (chosen, _, next_node) = node.choose(self, probability_true=1.0)
                self.choices_made.append(node)
                self._search_position = next_node
                if chosen:
                    self.solver.add(expr == node.condition_value)
                    ret = model_value_to_python(node.condition_value)
                    if (
                        in_debug()
                        and not self.is_detached
                        and expr not in self._already_logged
                    ):
                        self._already_logged.add(expr)
                        debug("SMT realized symbolic:", expr, "==", repr(ret))
                        debug("Realized at", ch_stack())
                    return ret
                else:
                    self.solver.add(expr != node.condition_value)

    def find_model_value_for_function(self, expr: z3.ExprRef) -> object:
        if not solver_is_sat(self.solver):
            raise CrossHairInternal("model unexpectedly became unsatisfiable")
        # TODO: this need to go into a tree node that returns UNKNOWN or worse
        # (because it just returns one example function; it's not covering the space)

        # TODO: note this is also unsound - after completion, the solver isn't
        # bound to the returned interpretation. (but don't know how to add the
        # right constraints) Maybe just use arrays instead.
        return self.solver.model()[expr]

    def current_snapshot(self) -> SnapshotRef:
        return SnapshotRef(len(self.heaps) - 1)

    def checkpoint(self):
        self.heaps.append(
            [(ref, typ, copy.deepcopy(val)) for (ref, typ, val) in self.heaps[-1]]
        )

    def add_value_to_heaps(self, ref: z3.ExprRef, typ: Type, value: object) -> None:
        # TODO: needs more testing
        for heap in self.heaps[:-1]:
            heap.append((ref, typ, copy.deepcopy(value)))
        self.heaps[-1].append((ref, typ, value))

    def find_key_in_heap(
        self,
        ref: z3.ExprRef,
        typ: Type,
        proxy_generator: Callable[[Type], object],
        snapshot: SnapshotRef = SnapshotRef(-1),
    ) -> object:
        with NoTracing():
            # TODO: needs more testing
            for (curref, curtyp, curval) in self.heaps[snapshot]:

                # TODO: using unify() is almost certainly wrong; just because the types
                # have some instances in common does not mean that `curval` actually
                # satisfies the requirements of `typ`:
                could_match = dynamic_typing.unify(curtyp, typ)
                if not could_match:
                    continue
                if self.smt_fork(curref == ref, probability_true=0.1):
                    debug(
                        "Heap key lookup ",
                        ref,
                        ": Found existing. ",
                        "type:",
                        name_of_type(type(curval)),
                        "id:",
                        id(curval) % 1000,
                    )
                    return curval
            ret = proxy_generator(typ)
            debug(
                "Heap key lookup ",
                ref,
                ": Created new. ",
                "type:",
                name_of_type(type(ret)),
                "id:",
                id(ret) % 1000,
            )

            self.add_value_to_heaps(ref, typ, ret)
            return ret

    def uniq(self):
        self.next_uniq += 1
        return "_{:x}".format(self.next_uniq)

    def smt_fork(
        self,
        expr: Optional[z3.ExprRef] = None,
        desc: Optional[str] = None,
        probability_true: Optional[float] = None,
    ) -> bool:
        if expr is None:
            expr = z3.Bool((desc or "fork") + self.uniq())
        return self.choose_possible(expr, probability_true)

    def defer_assumption(self, description: str, checker: Callable[[], bool]) -> None:
        self._deferred_assumptions.append((description, checker))

    def detach_path(self, currently_handling: Optional[BaseException] = None) -> None:
        """
        Mark the current path exhausted.

        Also verifies all deferred assumptions.
        After detaching, the space may continue to be used (for example, to print
        realized symbolics).
        """
        assert is_tracing()
        with NoTracing():
            if self.is_detached:
                debug("Path is already detached")
                return
            else:
                # Give ourselves a time extension for deferred assumptions and
                # (likely) counterexample generation to follow.
                self.execution_deadline += 2.0
            for description, checker in self._deferred_assumptions:
                with ResumedTracing():
                    check_ret = checker()
                if not prefer_true(check_ret):
                    raise IgnoreAttempt("deferred assumption failed: " + description)
            self.is_detached = True
            if not self._search_position.is_stem():
                self.raise_not_deterministic(
                    self._search_position,
                    f"Expect to detach path at a stem node, not at this node: {self._search_position}",
                    currently_handling=currently_handling,
                )
            node = self._search_position.grow_into(DeatchedPathNode())
            assert node.child.is_stem()
            self.choices_made.append(node)
            self._search_position = node.child
            debug("Detached from search tree")

    def cap_result_at_unknown(self):
        # TODO: this doesn't seem to work as intended.
        # If any execution path is confirmed, the end result is sometimes confirmed as well.
        self.status_cap = VerificationStatus.UNKNOWN

    def bubble_status(
        self, analysis: CallAnalysis
    ) -> Tuple[Optional[CallAnalysis], bool]:
        # In some cases, we might ignore an attempt while not at a leaf.
        if self._search_position.is_stem():
            self._search_position = self._search_position.grow_into(
                SearchLeaf(analysis)
            )
        else:
            self._search_position = self._search_position.simplify()
            assert isinstance(self._search_position, SearchTreeNode)
            self._search_position.exhausted = True
            self._search_position.result = analysis
        self._root.pathing_oracle.post_path_hook(self.choices_made)
        if not self.choices_made:
            return (analysis, True)
        for node in reversed(self.choices_made):
            node.update_result(analysis)
        if False:  # this is more noise than it's worth (usually)
            if in_debug():
                for line in debug_path_tree(
                    self._root, set(self.choices_made + [self._search_position])
                ):
                    debug(line)
        # debug('Path summary:', self.choices_made)
        first = self.choices_made[0]
        analysis = first.get_result()
        verification_status = analysis.verification_status
        if self.status_cap is not None and verification_status is not None:
            analysis.verification_status = min(verification_status, self.status_cap)
        return (analysis, first.is_exhausted())


class SimpleStateSpace(StateSpace):
    def __init__(self):
        super().__init__(monotonic() + 10000.0, 10000.0, RootNode())
        self.mark_all_parent_frames()
