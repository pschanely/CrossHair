import ast
import copy
import itertools
import random
import time
import traceback
from typing import *

import z3  # type: ignore

from crosshair import dynamic_typing
from crosshair.util import debug, PathTimeout, UnknownSatisfiability, CrosshairInternal, IdentityWrapper

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
    def __init__(self, space:'StateSpace'):
        self.space = space
        self.previous = None
    def __enter__(self):
        assert self.previous is None # (this context is not re-entrant)
        self.previous = self.space.running_framework_code
        self.space.running_framework_code = True
    def __exit__(self, exc_type, exc_value, tb):
        assert self.previous is not None
        self.space.running_framework_code = self.previous

_MISSING = object()

class StateSpace:
    def __init__(self, model_check_timeout: float):
        smt_tactic = z3.TryFor(z3.Tactic('smt'), 1 + int(model_check_timeout * 1000 * 0.75))
        nlsat_tactic = z3.TryFor(z3.Tactic('qfnra-nlsat'), 1 + int(model_check_timeout * 1000 * 0.25))
        self.solver = z3.OrElse(smt_tactic, nlsat_tactic).solver()
        self.solver.set(mbqi=True)
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
            
    def add(self, expr:z3.ExprRef) -> None:
        #debug('Committed to ', expr)
        self.solver.add(expr)

    def check(self, expr:z3.ExprRef) -> z3.CheckSatResult:
        solver = self.solver
        solver.push()
        solver.add(expr)
        #debug('CHECK ? ' + str(solver))
        ret = solver.check()
        #debug('CHECK => ' + str(ret))
        if ret not in (z3.sat, z3.unsat):
            debug('Solver cannot decide satisfiability')
            raise UnknownSatisfiability(str(ret)+': '+str(solver))
        solver.pop()
        return ret

    def choose_possible(self, expr:z3.ExprRef, favor_true=False) -> bool:
        raise NotImplementedError

    def find_model_value(self, expr:z3.ExprRef) -> object:
        value = self.solver.model().evaluate(expr, model_completion=True)
        return model_value_to_python(value)

    def find_model_value_for_function(self, expr:z3.ExprRef) -> object:
        return self.solver.model()[expr]

    def add_value_to_heaps(self, ref: z3.ExprRef, typ: Type, value: object) -> None:
        for heap in self.heaps[:-1]:
            heap.append((ref, typ, copy.deepcopy(value)))
        self.heaps[-1].append((ref, typ, value))

    def find_key_in_heap(self, ref: z3.ExprRef, typ: Type,
                         proxy_generator: Callable[[Type], object],
                         snapshot: SnapshotRef=SnapshotRef(-1)) -> object:
        with self.framework():
            for (curref, curtyp, curval) in itertools.chain(*self.heaps[snapshot:]):
                could_match = dynamic_typing.unify(curtyp, typ) or dynamic_typing.value_matches(curval, typ)
                if not could_match:
                    continue
                if self.smt_fork(curref == ref):
                    debug('HEAP key lookup ', ref, 'from snapshot', snapshot)
                    return curval
            ret = proxy_generator(typ)
            debug('HEAP key lookup ', ref, ' items. Created new', type(ret), 'from snapshot', snapshot)
            
            #assert dynamic_typing.unify(python_type(ret), typ), 'proxy type was {} and type required was {}'.format(type(ret), typ)
            self.add_value_to_heaps(ref, typ, ret)
            return ret

    def find_val_in_heap(self, value:object) -> z3.ExprRef:
        lastheap = self.heaps[-1]
        with self.framework():
            for (curref, curtyp, curval) in lastheap:
                if curval is value:
                    debug('HEAP value lookup for ', type(value), ' value type; found', curref)
                    return curref
            ref = z3.Const('heapkey'+str(value) + self.uniq(), HeapRef)
            for (curref, _, _) in lastheap:
                self.add(ref != curref)
            self.add_value_to_heaps(ref, type(value), value)
            debug('HEAP value lookup for ', type(value), ' value type; created new ', ref)
            return ref

    def uniq(self):
        self.next_uniq += 1
        if self.next_uniq >= 1000000:
            raise CrosshairInternal('Exhausted var space')
        return '{:06d}'.format(self.next_uniq)

    def smt_fork(self, expr:Optional[z3.ExprRef]=None) -> bool:
        if expr is None:
            expr = z3.Bool('fork' + self.uniq())
        return self.choose_possible(expr)

    def proxy_for_type(self, typ: Type, varname: str) -> object:
        raise NotImplementedError



def newrandom():
    return random.Random(1801243388510242075)

class SearchTreeNode:
    '''
    Helper class for TrackingStateSpace.
    Represents a single decision point.
    '''
    _random: random.Random
    exhausted :bool = False
    positive :Optional['SearchTreeNode'] = None
    negative :Optional['SearchTreeNode'] = None
    model_condition :Any = _MISSING
    statehash :Optional[str] = None
    def __init__(self, rand=None):
        self._random = rand if rand else newrandom()
        
    def choose(self, favor_true=False) -> Tuple[bool, 'SearchTreeNode']:
        assert self.positive is not None
        assert self.negative is not None
        positive_ok = not self.positive.exhausted
        negative_ok = not self.negative.exhausted
        if positive_ok and negative_ok:
            if favor_true:
                choice = True
            else:
                choice = bool(self._random.randint(0, 1))
        else:
            choice = positive_ok
        if choice:
            return (True, self.positive)
        else:
            return (False, self.negative)

class TrackingStateSpace(StateSpace):
    def __init__(self, 
                 execution_deadline: float,
                 model_check_timeout: float,
                 previous_searches: Optional[SearchTreeNode]=None):
        StateSpace.__init__(self, model_check_timeout)
        self.execution_deadline = execution_deadline
        self._random = newrandom()
        if previous_searches is None:
            previous_searches = SearchTreeNode(self._random)
        self.search_position = previous_searches

    def choose_possible(self, expr:z3.ExprRef, favor_true=False) -> bool:
        with self.framework():
            if time.time()  > self.execution_deadline:
                debug('Path execution timeout after making ', len(self.choices_made), ' choices.')
                raise PathTimeout
            notexpr = z3.Not(expr)
            node = self.search_position
            # NOTE: format_stack() is more human readable, but it pulls source file contents,
            # so it is (1) slow, and (2) unstable when source code changes while we are checking.
            statedesc = '\n'.join(map(str, traceback.extract_stack()))
            if node.statehash is None:
                node.statehash = statedesc
            else:
                if node.statehash != statedesc:
                    debug(' *** Begin Not Deterministic Debug *** ')
                    debug('     First state: ', len(node.statehash))
                    debug(node.statehash)
                    debug('     Last state: ', len(statedesc))
                    debug(statedesc)
                    debug('     Stack Diff: ')
                    import difflib
                    debug('\n'.join(difflib.context_diff(node.statehash.split('\n'), statedesc.split('\n'))))
                    debug(' *** End Not Deterministic Debug *** ')
                    raise NotDeterministic()
            if node.positive is None and node.negative is None:
                node.positive = SearchTreeNode(self._random)
                node.negative = SearchTreeNode(self._random)
                true_sat, false_sat = self.check(expr), self.check(notexpr)
                could_be_true = (true_sat == z3.sat)
                could_be_false = (false_sat == z3.sat)
                if (not could_be_true) and (not could_be_false):
                    debug(' *** Reached impossible code path *** ', true_sat, false_sat, expr)
                    debug('Current solver state:\n', str(self.solver))
                    raise CrosshairInternal('Reached impossible code path')
                if not could_be_true:
                    node.positive.exhausted = True
                if not could_be_false:
                    node.negative.exhausted = True

            (choose_true, new_search_node) = self.search_position.choose(favor_true=favor_true)
            self.choices_made.append(self.search_position)
            self.search_position = new_search_node
            expr = expr if choose_true else notexpr
            #debug('CHOOSE', expr)
            self.add(expr)
            return choose_true

    def find_model_value(self, expr:z3.ExprRef) -> object:
        with self.framework():
            while True:
                node = self.search_position
                if node.model_condition is _MISSING:
                    if self.solver.check() != z3.sat:
                        raise CrosshairInternal('model unexpectedly became unsatisfiable')
                    node.model_condition = self.solver.model().evaluate(expr, model_completion=True)
                value = node.model_condition
                if self.choose_possible(expr == value, favor_true=True):
                    if self.solver.check() != z3.sat:
                        raise CrosshairInternal('could not confirm model satisfiability after fixing value')
                    return model_value_to_python(value)

    def find_model_value_for_function(self, expr:z3.ExprRef) -> object:
        wrapper = IdentityWrapper(expr)
        while True:
            node = self.search_position
            if node.model_condition is _MISSING:
                if self.solver.check() != z3.sat:
                    raise CrosshairInternal('model unexpectedly became unsatisfiable')
                finterp = self.solver.model()[expr]
                node.model_condition = (wrapper, finterp)
            cmpvalue, finterp = node.model_condition
            if self.choose_possible(wrapper == cmpvalue, favor_true=True):
                if self.solver.check() != z3.sat:
                    raise CrosshairInternal('could not confirm model satisfiability after fixing value')
                return finterp

    def execution_log(self) -> str:
        log = []
        choices = self.choices_made
        for idx, node in enumerate(choices[:-1]):
            next_node = choices[idx + 1]
            assert next_node is node.positive or next_node is node.negative
            log.append('1' if node.positive is next_node else '0')
        return ''.join(log)

    def check_exhausted(self) -> bool:
        self.search_position.exhausted = True
        for node in reversed(self.choices_made):
            if (node.positive and node.positive.exhausted and
                node.negative and node.negative.exhausted):
                node.exhausted = True
            else:
                return False
        return True


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
