import ast
import copy
import enum
import itertools
import functools
import random
import time
import threading
import traceback
from dataclasses import dataclass
from typing import *

import z3  # type: ignore

from crosshair import dynamic_typing
from crosshair.util import debug
from crosshair.util import name_of_type
from crosshair.util import CrosshairInternal
from crosshair.util import IgnoreAttempt
from crosshair.util import IdentityWrapper
from crosshair.util import PathTimeout
from crosshair.util import UnknownSatisfiability
from crosshair.condition_parser import ConditionExpr
from crosshair.type_repo import SmtTypeRepository


@functools.total_ordering
class MessageType(enum.Enum):
    CONFIRMED = 'confirmed'
    CANNOT_CONFIRM = 'cannot_confirm'
    PRE_UNSAT = 'pre_unsat'
    POST_ERR = 'post_err'
    EXEC_ERR = 'exec_err'
    POST_FAIL = 'post_fail'
    SYNTAX_ERR = 'syntax_err'
    IMPORT_ERR = 'import_err'

    def __lt__(self, other):
        return self._order[self] < self._order[other]

MessageType._order = {  # type: ignore
    # This is the order that messages override each other (for the same source file line)
    MessageType.CONFIRMED: 0,
    MessageType.CANNOT_CONFIRM: 1,
    MessageType.PRE_UNSAT: 2,
    MessageType.POST_ERR: 3,
    MessageType.EXEC_ERR: 4,
    MessageType.POST_FAIL: 5,
    MessageType.SYNTAX_ERR: 6,
    MessageType.IMPORT_ERR: 7,
}

@dataclass(frozen=True)
class AnalysisMessage:
    state: MessageType
    message: str
    filename: str
    line: int
    column: int
    traceback: str
    execution_log: Optional[str] = None
    test_fn: Optional[str] = None
    condition_src: Optional[str] = None

    def toJSON(self):
        d = self.__dict__.copy()
        d['state'] = self.state.name
        return d

    @classmethod
    def fromJSON(cls, d):
        d['state'] = MessageType[d['state']]
        return AnalysisMessage(**d)

@functools.total_ordering
class VerificationStatus(enum.Enum):
    REFUTED = 0
    UNKNOWN = 1
    CONFIRMED = 2

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


@dataclass
class CallAnalysis:
    verification_status: Optional[VerificationStatus] = None  # None means "ignore"
    messages: Sequence[AnalysisMessage] = ()
    failing_precondition: Optional[ConditionExpr] = None
    failing_precondition_reason: str = ''

HeapRef = z3.DeclareSort('HeapRef')
SnapshotRef = NewType('SnapshotRef', int)


def model_value_to_python(value: z3.ExprRef) -> object:
    if z3.is_string_value(value):
        return value.as_string()
    elif z3.is_real(value):
        return float(value.as_fraction())
    else:
        return ast.literal_eval(repr(value))


class NotDeterministic(Exception):
    pass


_THREAD_LOCALS = threading.local()
class StateSpaceContext:
    def __init__(self, space: 'StateSpace'):
        self.space = space
    def __enter__(self):
        assert getattr(_THREAD_LOCALS, 'space', None) is None, 'Already in a state space context'
        _THREAD_LOCALS.space = self.space
    def __exit__(self, exc_type, exc_value, tb):
        assert getattr(_THREAD_LOCALS, 'space', None) is self.space, 'State space was altered in context'
        _THREAD_LOCALS.space = None

def optional_context_statespace() -> Optional['StateSpace']:
    return getattr(_THREAD_LOCALS, 'space', None)

def context_statespace() -> 'StateSpace':
    space = _THREAD_LOCALS.space
    assert space is not None, 'Not in a state space context'
    return space

class WithFrameworkCode:
    def __init__(self, space: 'StateSpace'):
        self.space = space
        self.previous = None

    def __enter__(self):
        assert self.previous is None  # (this context is not re-entrant)
        self.previous = self.space.running_framework_code
        self.space.running_framework_code = True

    def __exit__(self, exc_type, exc_value, tb):
        assert self.previous is not None
        self.space.running_framework_code = self.previous


class StateSpace:
    def __init__(self, model_check_timeout: float):
        smt_tactic = z3.TryFor(z3.Tactic('smt'), 1 +
                               int(model_check_timeout * 1000))
        self.solver = smt_tactic.solver()
        self.solver.set(mbqi=True)
        # turn off every randomization thing we can think of:
        self.solver.set('random-seed', 42)
        self.solver.set('smt.random-seed', 42)
        #self.solver.set('randomize', False)
        self.choices_made: List[SearchTreeNode] = []
        self.running_framework_code = False
        self.heaps: List[List[Tuple[z3.ExprRef, Type, object]]] = [[]]
        self.next_uniq = 1
        self.type_repo = SmtTypeRepository(self.solver)

    def framework(self) -> ContextManager:
        return WithFrameworkCode(self)

    def current_snapshot(self) -> SnapshotRef:
        return SnapshotRef(len(self.heaps) - 1)

    def checkpoint(self):
        self.heaps.append([])

    def add(self, expr: z3.ExprRef) -> None:
        #debug('Committed to ', expr)
        self.solver.add(expr)

    def fork_with_confirm_or_else(self, false_probabilty: float) -> bool:
        raise NotImplementedError

    def fork_parallel(self, false_probability: float) -> bool:
        raise NotImplementedError

    def choose_possible(self, expr: z3.ExprRef, favor_true=False) -> bool:
        raise NotImplementedError

    def find_model_value(self, expr: z3.ExprRef) -> object:
        value = self.solver.model().evaluate(expr, model_completion=True)
        return model_value_to_python(value)

    def find_model_value_for_function(self, expr: z3.ExprRef) -> object:
        return self.solver.model()[expr]

    def defer_assumption(self, description: str, checker: Callable[[], bool]) -> None:
        raise NotImplementedError

    def check_deferred_assumptions(self):
        pass

    def add_value_to_heaps(self, ref: z3.ExprRef, typ: Type, value: object) -> None:
        for heap in self.heaps[:-1]:
            heap.append((ref, typ, copy.deepcopy(value)))
        self.heaps[-1].append((ref, typ, value))

    def find_key_in_heap(self, ref: z3.ExprRef, typ: Type,
                         proxy_generator: Callable[[Type], object],
                         snapshot: SnapshotRef = SnapshotRef(-1)) -> object:
        with self.framework():
            for (curref, curtyp, curval) in itertools.chain(*self.heaps[snapshot:]):
                could_match = dynamic_typing.unify(curtyp, typ)
                if not could_match:
                    continue
                if self.smt_fork(curref == ref):
                    debug('HEAP key lookup ', ref, ': Found existing. ',
                          'type:', name_of_type(type(curval)), 'id:', id(curval)%1000)
                    return curval
            ret = proxy_generator(typ)
            debug('HEAP key lookup ', ref, ': Created new. ',
                  'type:', name_of_type(type(ret)), 'id:', id(ret) % 1000)

            #assert dynamic_typing.unify(python_type(ret), typ), 'proxy type was {} and type required was {}'.format(type(ret), typ)
            self.add_value_to_heaps(ref, typ, ret)
            return ret

    def find_val_in_heap(self, value: object) -> z3.ExprRef:
        lastheap = self.heaps[-1]
        with self.framework():
            for (curref, curtyp, curval) in lastheap:
                if curval is value:
                    debug('HEAP value lookup for ', type(
                        value), ' value type; found', curref)
                    return curref
            ref = z3.Const('heapkey' + str(value) + self.uniq(), HeapRef)
            for (curref, _, _) in lastheap:
                self.add(ref != curref)
            self.add_value_to_heaps(ref, type(value), value)
            debug('HEAP value lookup for ', type(value),
                  ' value type; created new ', ref)
            return ref

    def uniq(self):
        self.next_uniq += 1
        return '_{:x}'.format(self.next_uniq)

    def smt_fork(self, expr: Optional[z3.ExprRef] = None) -> bool:
        if expr is None:
            expr = z3.Bool('fork' + self.uniq())
        return self.choose_possible(expr)

    def proxy_for_type(self, typ: Type, varname: str) -> object:
        raise NotImplementedError


def newrandom():
    return random.Random(1801243388510242075)


class NodeLike:
    def is_exhausted(self) -> bool:
        return False
    def get_result(self) -> CallAnalysis:
        '''
        post: implies(_.verification_status == VerificationStatus.CONFIRMED, self.is_exhausted())
        '''
        raise NotImplementedError
    def is_stem(self) -> bool:
        return False
    def grow_into(self, node: 'SearchTreeNode') -> 'SearchTreeNode':
        raise NotImplementedError
    def simplify(self) -> 'NodeLike':
        return self

class NodeStem(NodeLike):
    evolution: Optional['SearchTreeNode'] = None
    def is_exhausted(self) -> bool:
        return False if self.evolution is None else self.evolution.is_exhausted()
    def get_result(self) -> CallAnalysis:
        return (CallAnalysis(VerificationStatus.UNKNOWN)
                if self.evolution is None
                else self.evolution.get_result())
    def is_stem(self) -> bool:
        return self.evolution is None
    def grow_into(self, node: 'SearchTreeNode') -> 'SearchTreeNode':
        self.evolution = node
        return node
    def simplify(self):
        return self if self.evolution is None else self.evolution

class SearchTreeNode(NodeLike):
    '''
    Abstract helper class for TrackingStateSpace.
    Represents a single decision point.
    '''
    statehash: Optional[str] = None
    result: CallAnalysis = CallAnalysis()
    exhausted: bool = False

    def choose(self, favor_true=False) -> Tuple[bool, NodeLike]:
        raise NotImplementedError
    def is_exhausted(self) -> bool:
        return self.exhausted
    def get_result(self) -> CallAnalysis:
        return self.result
    def update_result(self) -> bool:
        if not self.exhausted:
            next_result, next_exhausted = self.compute_result()
            if next_exhausted != self.exhausted or next_result != self.result:
                self.result, self.exhausted = next_result, next_exhausted
                return True
        return False
    def compute_result(self) -> Tuple[CallAnalysis, bool]:
        raise NotImplementedError

def solver_is_sat(solver, *a) -> bool:
    ret = solver.check(*a)
    if ret == z3.unknown:
        debug('Unknown satisfiability. Solver state follows:\n', solver)
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

class SinglePathNode(SearchTreeNode):
    decision: bool
    child: NodeLike
    def __init__(self, decision: bool):
        self.decision = decision
        self.child = NodeStem()
    def choose(self, favor_true=False) -> Tuple[bool, NodeLike]:
        return (self.decision, self.child)
    def compute_result(self) -> Tuple[CallAnalysis, bool]:
        self.child = self.child.simplify()
        return (self.child.get_result(), self.child.is_exhausted())
        
class BinaryPathNode(SearchTreeNode):
    positive: NodeLike
    negative: NodeLike

class RandomizedBinaryPathNode(BinaryPathNode):
    _random: random.Random

    def __init__(self, rand=None):
        self._random = rand if rand else newrandom()
        self.positive = NodeStem()
        self.negative = NodeStem()

    def false_probability(self):
        return 0.5
    
    def choose(self, favor_true=False) -> Tuple[bool, NodeLike]:
        positive_ok = not self.positive.is_exhausted()
        negative_ok = not self.negative.is_exhausted()
        assert positive_ok or negative_ok
        if positive_ok and negative_ok:
            if favor_true:
                choice = True
            else:
                choice = self._random.uniform(0.0, 1.0) > self.false_probability()
        else:
            choice = positive_ok
        return (choice, self.positive if choice else self.negative)
    def _simplify(self) -> None:
        self.positive = self.positive.simplify()
        self.negative = self.negative.simplify()

class ConfirmOrElseNode(RandomizedBinaryPathNode):
    def __init__(self, false_probability: float):
        super().__init__()
        self._false_probability = false_probability
    def compute_result(self) -> Tuple[CallAnalysis, bool]:
        self._simplify()
        if node_status(self.positive) == VerificationStatus.CONFIRMED:
            return (self.positive.get_result(), True)
        positive_exhausted = self.positive.is_exhausted()
        negative_exhausted = self.negative.is_exhausted()
        exhausted = ((negative_exhausted and positive_exhausted) or
                     (negative_exhausted and node_status(self.negative) in (VerificationStatus.CONFIRMED, VerificationStatus.REFUTED)))
        return (self.negative.get_result(), exhausted)
    def false_probability(self) -> float:
        return 1.0 if self.positive.is_exhausted() else self._false_probability

class ParallelNode(RandomizedBinaryPathNode):
    '''
    Chooses either path; the first complete result will be used.
    '''
    def __init__(self, false_probability: float):
        super().__init__()
        self._false_probability = false_probability
    def compute_result(self) -> Tuple[CallAnalysis, bool]:
        self._simplify()
        pos_exhausted = self.positive.is_exhausted()
        neg_exhausted = self.negative.is_exhausted()
        if pos_exhausted and not node_status(self.positive) == VerificationStatus.UNKNOWN:
            return (self.positive.get_result(), True)
        if neg_exhausted and not node_status(self.negative) == VerificationStatus.UNKNOWN:
            return (self.negative.get_result(), True)
        return merge_node_results(self.positive.get_result(), pos_exhausted and neg_exhausted, self.negative)
    def false_probability(self) -> float:
        return 1.0 if self.positive.is_exhausted() else self._false_probability

def merge_node_results(left: CallAnalysis, exhausted: bool, node: NodeLike) -> Tuple[CallAnalysis, bool]:
    '''
    Merges analysis from different branches of code. (combines messages, takes
    the worst verification status of the two, etc)
    '''
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
    return (CallAnalysis(
        min(left.verification_status, right.verification_status),
        list(left.messages) + list(right.messages),
        precond_side.failing_precondition,
        precond_side.failing_precondition_reason), exhausted)

class WorstResultNode(RandomizedBinaryPathNode):
    forced_path: Optional[bool] = None
    def __init__(self, rand: random.Random, expr: z3.ExprRef, solver: z3.Solver):
        RandomizedBinaryPathNode.__init__(self, rand)
        notexpr = z3.Not(expr)
        could_be_true = solver_is_sat(solver, expr)
        could_be_false = solver_is_sat(solver, notexpr)
        #debug(expr, ' true?', could_be_true, ' false?', could_be_false)
        if (not could_be_true) and (not could_be_false):
            debug(' *** Reached impossible code path *** ')
            debug('Current solver state:\n', str(solver))
            raise CrosshairInternal('Reached impossible code path')
        elif not could_be_true:
            self.forced_path = False
        elif not could_be_false:
            self.forced_path = True
        #self._dbgstr = str(expr) + ' forced=' + str(self.forced_path)

    #def __str__(self):
    #    return f'WorstResultNode({self._dbgstr})'
    
    def choose(self, favor_true=False) -> Tuple[bool, NodeLike]:
        if self.forced_path is None:
            return RandomizedBinaryPathNode.choose(self, favor_true)
        return (self.forced_path, self.positive if self.forced_path else self.negative)
        
    def false_probability(self):
        # When both paths are unexplored, we bias for False.
        # As a heuristic, this tends to prefer early completion:
        # - Loop conditions tend to repeat on True.
        # - Optional[X] turns into Union[X, None] and False conditions
        #   biases for the last item in the union.
        # We pick a False value more than 2/3rds of the time to avoid
        # explosions while constructing binary-tree-like objects.
        return 0.75

    def compute_result(self) -> Tuple[CallAnalysis, bool]:
        self._simplify()
        exhausted = (
            (self.positive.is_exhausted() and self.negative.is_exhausted()) or
            (self.forced_path is True and self.positive.is_exhausted()) or
            (self.forced_path is False and self.negative.is_exhausted()))
        if node_status(self.positive) == VerificationStatus.REFUTED or (self.forced_path is True):
            return (self.positive.get_result(), exhausted)
        if node_status(self.negative) == VerificationStatus.REFUTED or (self.forced_path is False):
            return (self.negative.get_result(), exhausted)
        return merge_node_results(self.positive.get_result(), self.positive.is_exhausted(), self.negative)

class ModelValueNode(WorstResultNode):
    condition_value: object = None
    def __init__(self, rand: random.Random, expr: z3.ExprRef, solver: z3.Solver):
        if self.condition_value is None:
            if not solver_is_sat(solver):
                debug('bad solver', solver.sexpr())
                raise CrosshairInternal('unexpected un sat')
            self.condition_value = solver.model().evaluate(expr, model_completion=True)
        WorstResultNode.__init__(self, rand, expr == self.condition_value, solver)

class TrackingStateSpace(StateSpace):
    search_position: NodeLike
    _deferred_assumptions: List[Tuple[str, Callable[[], bool]]]
    def __init__(self,
                 execution_deadline: float,
                 model_check_timeout: float,
                 search_root: SinglePathNode):
        StateSpace.__init__(self, model_check_timeout)
        self.execution_deadline = execution_deadline
        self._random = newrandom()
        _, self.search_position = search_root.choose()
        self._deferred_assumptions = []

    def fork_with_confirm_or_else(self, false_probability: float) -> bool:
        if self.search_position.is_stem():
            self.search_position = self.search_position.grow_into(ConfirmOrElseNode(false_probability))
        node = self.search_position.simplify()
        assert isinstance(node, SearchTreeNode)
        self.choices_made.append(node)
        ret, next_node = node.choose()
        self.search_position = next_node
        return ret

    def fork_parallel(self, false_probability: float) -> bool:
        if self.search_position.is_stem():
            self.search_position = self.search_position.grow_into(ParallelNode(false_probability))
        node = self.search_position.simplify()
        assert isinstance(node, SearchTreeNode)
        self.choices_made.append(node)
        ret, next_node = node.choose()
        self.search_position = next_node
        return ret

    def choose_possible(self, expr: z3.ExprRef, favor_true=False) -> bool:
        with self.framework():
            if time.monotonic() > self.execution_deadline:
                debug('Path execution timeout after making ',
                      len(self.choices_made), ' choices.')
                raise PathTimeout
            notexpr = z3.Not(expr)
            if self.search_position.is_stem():
                self.search_position = self.search_position.grow_into(
                    WorstResultNode(self._random, expr, self.solver))

            self.search_position = self.search_position.simplify()
            node = self.search_position
            # NOTE: format_stack() is more human readable, but it pulls source file contents,
            # so it is (1) slow, and (2) unstable when source code changes while we are checking.
            statedesc = '\n'.join(map(str, traceback.extract_stack()))
            assert isinstance(node, SearchTreeNode)
            if node.statehash is None:
                node.statehash = statedesc
            else:
                if node.statehash != statedesc:
                    debug(self.choices_made)
                    debug(' *** Begin Not Deterministic Debug *** ')
                    debug('     First state: ', len(node.statehash))
                    debug(node.statehash)
                    debug('     Last state: ', len(statedesc))
                    debug(statedesc)
                    debug('     Stack Diff: ')
                    import difflib
                    debug('\n'.join(difflib.context_diff(
                        node.statehash.split('\n'), statedesc.split('\n'))))
                    debug(' *** End Not Deterministic Debug *** ')
                    raise NotDeterministic()
            choose_true, stem = node.choose(favor_true=favor_true)
            assert isinstance(self.search_position, SearchTreeNode)
            self.choices_made.append(self.search_position)
            self.search_position = stem
            expr = expr if choose_true else notexpr
            debug('SMT chose:', expr)
            self.add(expr)
            return choose_true

    def find_model_value(self, expr: z3.ExprRef) -> object:
        with self.framework():
            while True:
                if self.search_position.is_stem():
                    self.search_position = self.search_position.grow_into(ModelValueNode(self._random, expr, self.solver))
                node = self.search_position.simplify()
                assert isinstance(node, ModelValueNode)
                (chosen, next_node) = node.choose(favor_true=True)
                self.choices_made.append(node)
                self.search_position = next_node
                if chosen:
                    self.solver.add(expr == node.condition_value)
                    debug('SMT realized symbolic:', expr == node.condition_value)
                    return model_value_to_python(node.condition_value)
                else:
                    self.solver.add(expr != node.condition_value)
    
    def find_model_value_for_function(self, expr: z3.ExprRef) -> object:
        # TODO: this need to go into a tree node that returns UNKNOWN or worse
        # (because it just returns one example function; it's not covering the space)
        if not solver_is_sat(self.solver):
            raise CrosshairInternal(
                'model unexpectedly became unsatisfiable')
        return self.solver.model()[expr]
    
    def defer_assumption(self, description: str, checker: Callable[[], bool]) -> None:
        self._deferred_assumptions.append((description, checker))

    def check_deferred_assumptions(self) -> None:
        for description, checker in self._deferred_assumptions:
            if not checker():
                raise IgnoreAttempt('deferred assumption failed: '+description)

    def execution_log(self) -> str:
        log = []
        choices = self.choices_made
        for idx, node in enumerate(choices[:-1]):
            next_node = choices[idx + 1]
            if isinstance(node, BinaryPathNode):
                assert next_node is node.positive or next_node is node.negative
                log.append('1' if node.positive is next_node else '0')
        return ''.join(log)

    def bubble_status(self, analysis: CallAnalysis) -> Tuple[
            Optional[CallAnalysis], bool]:
        # In some cases, we might ignore an attempt while not at a leaf.
        if self.search_position.is_stem():
            self.search_position = self.search_position.grow_into(SearchLeaf(analysis))
        else:
            self.search_position = self.search_position.simplify()
            assert isinstance(self.search_position, SearchTreeNode)
            self.search_position.exhausted = True
            self.search_position.result = analysis
        if not self.choices_made:
            return (analysis, True)
        for node in reversed(self.choices_made):
            node.update_result()
        first = self.choices_made[0]
        return (first.get_result(), first.is_exhausted())

class SimpleStateSpace(TrackingStateSpace):
    def __init__(self):
        search_root = SinglePathNode(True)
        super().__init__(time.monotonic() + 10000.0, 10000.0, search_root)

