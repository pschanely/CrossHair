import ast
import copy
import enum
import itertools
import functools
import random
import time
import traceback
from dataclasses import dataclass
from typing import *

import z3  # type: ignore

from crosshair import dynamic_typing
from crosshair.util import debug, PathTimeout, UnknownSatisfiability, CrosshairInternal, IgnoreAttempt, IdentityWrapper
from crosshair.condition_parser import ConditionExpr


@functools.total_ordering
class MessageType(enum.Enum):
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
    MessageType.CANNOT_CONFIRM: 0,
    MessageType.PRE_UNSAT: 1,
    MessageType.POST_ERR: 2,
    MessageType.EXEC_ERR: 3,
    MessageType.POST_FAIL: 4,
    MessageType.SYNTAX_ERR: 5,
    MessageType.IMPORT_ERR: 6,
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
    if z3.is_string(value):
        return value.as_string()
    elif z3.is_real(value):
        return float(value.as_fraction())
    else:
        return ast.literal_eval(repr(value))


class NotDeterministic(CrosshairInternal):
    pass


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
                               int(model_check_timeout * 1000 * 0.75))
        nlsat_tactic = z3.TryFor(
            z3.Tactic('qfnra-nlsat'), 1 + int(model_check_timeout * 1000 * 0.25))
        self.solver = z3.OrElse(smt_tactic, nlsat_tactic).solver()
        self.solver.set(mbqi=True)
        # turn off every randomization thing we can think of:
        self.solver.set('random-seed', 42)
        self.solver.set('smt.random-seed', 42)
        self.solver.set('randomize', False)
        self.choices_made: List[SearchTreeNode] = []
        self.running_framework_code = False
        self.heaps: List[List[Tuple[z3.ExprRef, Type, object]]] = [[]]
        self.next_uniq = 1

    def framework(self) -> ContextManager:
        return WithFrameworkCode(self)

    def current_snapshot(self) -> SnapshotRef:
        return SnapshotRef(len(self.heaps) - 1)

    def checkpoint(self):
        debug('checkpoint', len(self.heaps) + 1)
        self.heaps.append([])

    def add(self, expr: z3.ExprRef) -> None:
        #debug('Committed to ', expr)
        self.solver.add(expr)

    def check(self, expr: z3.ExprRef) -> z3.CheckSatResult:
        solver = self.solver
        solver.push()
        solver.add(expr)
        #debug('CHECK ? ' + str(solver.sexpr()))
        ret = solver.check()
        #debug('CHECK => ' + str(ret))
        if ret not in (z3.sat, z3.unsat):
            debug('Solver cannot decide satisfiability')
            raise UnknownSatisfiability(str(ret) + ': ' + str(solver))
        solver.pop()
        return ret

    def fork_with_confirm_or_else(self) -> bool:
        raise NotImplementedError

    def choose_possible(self, expr: z3.ExprRef, favor_true=False) -> bool:
        raise NotImplementedError

    def find_model_value(self, expr: z3.ExprRef) -> object:
        value = self.solver.model().evaluate(expr, model_completion=True)
        return model_value_to_python(value)

    def find_model_value_for_function(self, expr: z3.ExprRef) -> object:
        return self.solver.model()[expr]

    def add_value_to_heaps(self, ref: z3.ExprRef, typ: Type, value: object) -> None:
        for heap in self.heaps[:-1]:
            heap.append((ref, typ, copy.deepcopy(value)))
        self.heaps[-1].append((ref, typ, value))

    def find_key_in_heap(self, ref: z3.ExprRef, typ: Type,
                         proxy_generator: Callable[[Type], object],
                         snapshot: SnapshotRef = SnapshotRef(-1)) -> object:
        with self.framework():
            for (curref, curtyp, curval) in itertools.chain(*self.heaps[snapshot:]):
                could_match = dynamic_typing.unify(
                    curtyp, typ) or dynamic_typing.value_matches(curval, typ)
                if not could_match:
                    continue
                if self.smt_fork(curref == ref):
                    debug('HEAP key lookup ', ref, 'from snapshot', snapshot)
                    return curval
            ret = proxy_generator(typ)
            debug('HEAP key lookup ', ref, ' items. Created new',
                  type(ret), 'from snapshot', snapshot)

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
        if self.next_uniq >= 1000000:
            raise CrosshairInternal('Exhausted var space')
        return '{:06d}'.format(self.next_uniq)

    def smt_fork(self, expr: Optional[z3.ExprRef] = None) -> bool:
        if expr is None:
            expr = z3.Bool('fork' + self.uniq())
        return self.choose_possible(expr)

    def proxy_for_type(self, typ: Type, varname: str) -> object:
        raise NotImplementedError


def newrandom():
    return random.Random(1801243388510242075)


def stem_or_attr(parent: 'SearchTreeNode', attr: str):
    if getattr(parent, attr, None) is not None:
        return getattr(parent, attr)
    return NodeStem(parent=parent, attr=attr)

class NodeStem:
    def __init__(self,
                 parent: Optional['SearchTreeNode'] = None,
                 attr: Optional[str] = None,
                 setter: Optional[Callable[['SearchTreeNode'], None]] = None):
        if setter:
            self.setter = setter
        else:
            def attr_setter(node: SearchTreeNode):
                assert parent is not None
                assert attr is not None
                assert getattr(parent, attr, None) is None
                setattr(parent, attr, node)
            self.setter = attr_setter
    def get_result(self) -> Optional[CallAnalysis]:
        return None
    def grow_into(self, node: 'SearchTreeNode') -> 'SearchTreeNode':
        self.setter(node)
        return node

class SearchTreeNode:
    '''
    Abstract helper class for TrackingStateSpace.
    Represents a single decision point.
    '''
    statehash: Optional[str] = None
    result: Optional[CallAnalysis] = None
    exhausted: bool = False

    def choose(self, favor_true=False) -> Tuple[bool, Union[NodeStem, 'SearchTreeNode']]:
        raise NotImplementedError
    def get_result(self) -> Optional[CallAnalysis]:
        return self.result
    def update_result(self) -> bool:
        if self.result is not None:
            return False
        self.result = self.compute_result()
        return self.result is not None
    def compute_result(self) -> Optional[CallAnalysis]:
        raise NotImplementedError

def node_result(node: Optional[SearchTreeNode]) -> Optional[CallAnalysis]:
    if node is None:
        return None
    return node.get_result()

def node_has_status(node: Optional[SearchTreeNode], status: VerificationStatus) -> bool:
    result = node_result(node)
    if result is not None and result.verification_status is not None:
        return result.verification_status == status
    else:
        return False

class SearchLeaf(SearchTreeNode):
    def __init__(self, result: CallAnalysis):
        self.result = result

class SinglePathNode(SearchTreeNode):
    decision: bool
    child: Optional[SearchTreeNode] = None
    def __init__(self, decision: bool):
        self.decision = decision
    def choose(self, favor_true=False) -> Tuple[bool, Union['NodeStem', SearchTreeNode]]:
        return (self.decision, stem_or_attr(self, 'child'))
    def compute_result(self) -> Optional[CallAnalysis]:
        if self.child is None:
            return None
        return self.child.get_result()
        
class BinaryPathNode(SearchTreeNode):
    positive: Optional['SearchTreeNode'] = None
    negative: Optional['SearchTreeNode'] = None

class RandomizedBinaryPathNode(BinaryPathNode):
    _random: random.Random

    def __init__(self, rand=None):
        self._random = rand if rand else newrandom()

    def false_probability(self):
        return 0.5
    
    def choose(self, favor_true=False) -> Tuple[bool, Union['NodeStem', SearchTreeNode]]:
        positive_ok = node_result(self.positive) is None
        negative_ok = node_result(self.negative) is None
        assert positive_ok or negative_ok
        if positive_ok and negative_ok:
            if favor_true:
                choice = True
            else:
                choice = self._random.uniform(0.0, 1.0) > self.false_probability()
        else:
            choice = positive_ok
        return (choice, stem_or_attr(self, 'positive' if choice else 'negative'))

class ConfirmOrElseNode(RandomizedBinaryPathNode):
    def compute_result(self) -> Optional[CallAnalysis]:
        if node_has_status(self.positive, VerificationStatus.CONFIRMED):
            assert self.positive is not None
            return self.positive.get_result()
        return node_result(self.negative)

def merge_analysis(left: CallAnalysis, right: CallAnalysis) -> CallAnalysis:
    '''
    Merges analysis from different branches of code. (combines messages, takes
    the worst verification status of the two, etc)
    '''
    if left.verification_status is None:
        return right
    if right.verification_status is None:
        return left
    if left.failing_precondition and right.failing_precondition:
        lc, rc = left.failing_precondition, right.failing_precondition
        precond_side = left if lc.line > rc.line else right
    else:
        precond_side = left if left.failing_precondition else right
    return CallAnalysis(
        min(left.verification_status, right.verification_status),
        list(left.messages) + list(right.messages),
        precond_side.failing_precondition,
        precond_side.failing_precondition_reason)

class WorstResultNode(RandomizedBinaryPathNode):
    def false_probability(self):
        # When both paths are unexplored, we bias for False.
        # As a heuristic, this tends to prefer early completion:
        # - Loop conditions tend to repeat on True.
        # - Optional[X] turns into Union[X, None] and False conditions
        #   biases for the last item in the union.
        # We pick a False value more than 2/3rds of the time to avoid
        # explosions while constructing binary-tree-like objects.
        return 0.75

    def compute_result(self) -> Optional[CallAnalysis]:
        if node_has_status(self.positive, VerificationStatus.REFUTED):
            return node_result(self.positive)
        if node_has_status(self.negative, VerificationStatus.REFUTED):
            return node_result(self.negative)
        positive_result = node_result(self.positive)
        negative_result = node_result(self.negative)
        if (positive_result is None) or (negative_result is None):
            return None
        return merge_analysis(positive_result, negative_result)

class ModelValueNode(SearchTreeNode):
    values_so_far: Dict[z3.ExprRef, SearchTreeNode]
    def __init__(self, rand: random.Random):
        self.rand = rand
        self.completed = False
        self.values_so_far: Dict[z3.ExprRef, Union[NodeStem, SearchTreeNode]] = {}
    def choose_model_value(self, expr: z3.ExprRef, solver: z3.Solver) -> Tuple[object, Union['NodeStem', SearchTreeNode]]:
        values_so_far = self.values_so_far
        if solver.check() != z3.sat:
            raise CrosshairInternal(
                'model unexpectedly became unsatisfiable')
        not_any_completed = [expr != value for value, node in values_so_far.items()
                             if node.get_result() is not None]
        solver.add(*not_any_completed)
        sat_result = solver.check()
        if sat_result == z3.unsat:
            # Most spaces are infinte, but sometimes we'll exhaust the space.
            self.completed = True
            raise IgnoreAttempt('cannot find any more model values')
        elif sat_result != z3.sat:
            raise UnknownSatisfiability(str(sat_result) + ': ' + str(solver) + str(not_any_completed))
        smtval = solver.model().evaluate(expr, model_completion=True)
        solver.add(expr == smtval)
        if solver.check() != z3.sat:
            # this should never happen, but seems like (perhaps with sequences) it does
            raise UnknownSatisfiability('model().evaluate produced value incompatable with solver state')
            #raise CrosshairInternal(
            #    'model unexpectedly became unsatisfiable')
        pyval = model_value_to_python(smtval)
        if smtval in values_so_far:
            return (pyval, values_so_far[smtval])
        else:
            def stem_setter(n: SearchTreeNode):
                assert smtval not in values_so_far
                values_so_far[smtval] = n
            return (pyval, NodeStem(setter=stem_setter))
    def compute_result(self) -> Optional[CallAnalysis]:
        if not self.completed:
            return None
        sub_results = [node.get_result() for node in self.values_so_far.values()
                       if node.get_result() is not None]
        if any(r is None for r in sub_results):
            return None
        return functools.reduce(merge_analysis, sub_results)  # type:ignore

class TrackingStateSpace(StateSpace):
    search_position: Union[NodeStem, SearchTreeNode]
    def __init__(self,
                 execution_deadline: float,
                 model_check_timeout: float,
                 search_root: SinglePathNode):
        StateSpace.__init__(self, model_check_timeout)
        self.execution_deadline = execution_deadline
        self._random = newrandom()
        _, self.search_position = search_root.choose()

    def fork_with_confirm_or_else(self) -> bool:
        if isinstance(self.search_position, NodeStem):
            self.search_position = self.search_position.grow_into(
                ConfirmOrElseNode())
        self.choices_made.append(self.search_position)
        ret, next_node = self.search_position.choose()
        self.search_position = next_node
        return ret
        
    def choose_possible(self, expr: z3.ExprRef, favor_true=False) -> bool:
        with self.framework():
            if time.time() > self.execution_deadline:
                debug('Path execution timeout after making ',
                      len(self.choices_made), ' choices.')
                raise PathTimeout
            notexpr = z3.Not(expr)
            if isinstance(self.search_position, NodeStem):
                true_sat, false_sat = self.check(expr), self.check(notexpr)
                could_be_true = (true_sat == z3.sat)
                could_be_false = (false_sat == z3.sat)
                if (not could_be_true) and (not could_be_false):
                    debug(' *** Reached impossible code path *** ',
                          true_sat, false_sat, expr)
                    debug('Current solver state:\n', str(self.solver))
                    raise CrosshairInternal('Reached impossible code path')
                if could_be_true and could_be_false:
                    self.search_position = self.search_position.grow_into(
                        WorstResultNode(self._random))
                else:
                    self.search_position = self.search_position.grow_into(
                        SinglePathNode(could_be_true))

            node = self.search_position
            # NOTE: format_stack() is more human readable, but it pulls source file contents,
            # so it is (1) slow, and (2) unstable when source code changes while we are checking.
            statedesc = '\n'.join(map(str, traceback.extract_stack()))
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
            self.choices_made.append(self.search_position)
            self.search_position = stem
            expr = expr if choose_true else notexpr
            #debug('CHOOSE', expr)
            self.add(expr)
            return choose_true

    def find_model_value(self, expr: z3.ExprRef) -> object:
        with self.framework():
            if isinstance(self.search_position, NodeStem):
                self.search_position = self.search_position.grow_into(ModelValueNode(self._random))
            node = self.search_position
            assert isinstance(node, ModelValueNode)
            self.choices_made.append(node)
            value, next_node = node.choose_model_value(expr, self.solver)
            self.search_position = next_node
            return value
    
    def find_model_value_for_function(self, expr: z3.ExprRef) -> object:
        # TODO: this need to go into a tree node that returns UNKNOWN or worse
        # (because it just returns one example function; it's not covering the space)
        if self.solver.check() != z3.sat:
            raise CrosshairInternal(
                'model unexpectedly became unsatisfiable')
        finterp = self.solver.model()[expr]
        if self.solver.check() != z3.sat:
            raise CrosshairInternal(
                'could not confirm model satisfiability after fixing value')
        return finterp
    
    def execution_log(self) -> str:
        log = []
        choices = self.choices_made
        for idx, node in enumerate(choices[:-1]):
            next_node = choices[idx + 1]
            if isinstance(node, BinaryPathNode):
                assert next_node is node.positive or next_node is node.negative
                log.append('1' if node.positive is next_node else '0')
        return ''.join(log)

    def bubble_status(self, analysis: CallAnalysis) -> Optional[CallAnalysis]:
        # In some cases, we might ignore an attempt while not at a leaf.
        if isinstance(self.search_position, NodeStem):
            self.search_position = self.search_position.grow_into(SearchLeaf(analysis))
        for node in reversed(self.choices_made):
            node.update_result()
        if not self.choices_made:
            return analysis
        return self.choices_made[0].get_result()


class ReplayStateSpace(StateSpace):
    def __init__(self, execution_log: str):
        StateSpace.__init__(self, model_check_timeout=5.0)
        self.execution_log = execution_log
        self.log_index = 0

    def choose_possible(self, expr: z3.ExprRef, favor_true=False) -> bool:
        with self.framework():
            notexpr = z3.Not(expr)
            true_sat, false_sat = self.check(expr), self.check(notexpr)
            could_be_true = (true_sat == z3.sat)
            could_be_false = (false_sat == z3.sat)
            if (not could_be_true) and (not could_be_false):
                raise CrosshairInternal('Reached impossible code path')
            else:
                log, idx = self.execution_log, self.log_index
                if idx >= len(log):
                    if idx == len(log):
                        debug('Precise path replay unsuccessful.')
                    return False
                debug('decide_true = ', self.execution_log[self.log_index])
                decide_true = (self.execution_log[self.log_index] == '1')
                self.log_index += 1
            expr = expr if decide_true else notexpr
            debug('REPLAY CHOICE', expr)
            self.add(expr)
            if not self.solver.check():
                debug('Precise path replay unsuccessful.')
            return decide_true
