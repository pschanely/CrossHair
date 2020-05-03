import collections
import copy
import dataclasses
import re
import sys
import unittest
from typing import *

from crosshair.core import make_fake_object
from crosshair.core_and_libs import *
from crosshair.test_util import check_ok
from crosshair.test_util import check_exec_err
from crosshair.test_util import check_post_err
from crosshair.test_util import check_fail
from crosshair.test_util import check_unknown
from crosshair.test_util import check_messages
from crosshair.util import set_debug
from crosshair.statespace import SimpleStateSpace














#
# Begin fixed line number area.
# Tests depend on the line number of the following section.
#

class Pokeable:
    '''
    inv: self.x >= 0
    '''
    x: int = 1

    def poke(self) -> None:
        '''
        post[self]: True
        '''
        self.x += 1

    def wild_pokeby(self, amount: int) -> None:
        '''
        post[self]: True
        '''
        self.x += amount

    def safe_pokeby(self, amount: int) -> None:
        '''
        pre: amount >= 0
        post[self]: True
        '''
        self.x += amount

    def __repr__(self) -> str:
        return 'Pokeable(' + repr(self.x) + ')'

    def __init__(self, x: int) -> None:
        '''
        pre: x >= 0
        '''
        self.x = x


#
# End fixed line number area.
#

class Cat:
    def size(self) -> int:
        return 1
class BiggerCat(Cat):
    def size(self) -> int:
        return 2

class PersonTuple(NamedTuple):
    name: str
    age: int

class PersonWithoutAttributes:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


NOW = 1000

@dataclasses.dataclass(
    repr=False  # make checking faster (repr has an infinite search tree)
)
class Person:
    '''
    Contains various features that we expect to be successfully checkable.

    inv: True # TODO: test that NameError in invariant does the right thing
    '''
    name: str
    birth: int

    def _getage(self):
        return NOW - self.birth

    def _setage(self, newage):
        self.birth = NOW - newage

    def _delage(self):
        del self.birth
    age = property(_getage, _setage, _delage, 'Age of person')

    def abstract_operation(self):
        '''
        post: False # doesn't error because the method is "abstract"
        '''
        raise NotImplementedError


class SmokeDetector:
    ''' inv: not (self._is_plugged_in and self._in_original_packaging) '''
    _in_original_packaging: bool
    _is_plugged_in: bool

    def signaling_alarm(self, air_samples: List[str]) -> bool:
        '''
        pre: self._is_plugged_in
        post: implies('smoke' in air_samples, _ == True)
        '''
        return 'smoke' in air_samples


class Measurer:
    def measure(self, x: int) -> str:
        '''
        post: _ == self.measure(-x)
        '''
        return 'small' if x <= 10 else 'large'


A_REFERENCED_THING = 42
@dataclasses.dataclass(repr=False)
class ReferenceHoldingClass:
    '''
    inv: self.item != A_REFERENCED_THING
    '''
    item: str


def fibb(x: int) -> int:
    '''
    pre: x>=0
    post: _ < 10
    '''
    if x <= 2:
        return 1
    r1, r2 = fibb(x - 1), fibb(x - 2)
    ret = r1 + r2
    return ret


def recursive_example(x: int) -> bool:
    '''
    pre: x >= 0
    post[]:
        __old__.x >= 0  # just to confirm __old__ works in recursive cases
        _ == True
    '''
    if x == 0:
        return True
    else:
        return recursive_example(x - 1)


class ProxiedObjectTest(unittest.TestCase):
    def test_proxy_type(self) -> None:
        poke = make_fake_object(SimpleStateSpace(), Pokeable, 'ppoke')
        self.assertIs(type(poke), Pokeable)

    def test_copy(self) -> None:
        poke1 = make_fake_object(SimpleStateSpace(), Pokeable, 'ppoke')
        poke1.poke()
        poke2 = copy.copy(poke1)
        self.assertIsNot(poke1, poke2)
        self.assertEqual(type(poke1), type(poke2))
        self.assertIs(poke1.x, poke2.x)
        poke1.poke()
        self.assertIsNot(poke1.x, poke2.x)
        self.assertNotEqual(str(poke1.x.var), str(poke2.x.var))

    def test_proxy_alone(self) -> None:
        def f(pokeable: Pokeable) -> None:
            '''
            post[pokeable]: pokeable.x > 0
            '''
            pokeable.poke()
        self.assertEqual(*check_ok(f))

    def test_proxy_in_list(self) -> None:
        def f(pokeables: List[Pokeable]) -> None:
            '''
            pre: len(pokeables) == 1
            post: all(p.x > 0 for p in pokeables)
            '''
            for pokeable in pokeables:
                pokeable.poke()
        self.assertEqual(*check_ok(f))


class ObjectsTest(unittest.TestCase):

    def test_obj_member_fail(self) -> None:
        def f(foo: Pokeable) -> int:
            '''
            pre: 0 <= foo.x <= 4
            post[foo]: _ < 5
            '''
            foo.poke()
            foo.poke()
            return foo.x
        self.assertEqual(*check_fail(f))

    def test_obj_member_nochange_ok(self) -> None:
        def f(foo: Pokeable) -> int:
            ''' post: _ == foo.x '''
            return foo.x
        self.assertEqual(*check_ok(f))

    def test_obj_member_change_ok(self) -> None:
        def f(foo: Pokeable) -> int:
            '''
            pre: foo.x >= 0
            post[foo]: foo.x >= 2
            '''
            foo.poke()
            foo.poke()
            return foo.x
        self.assertEqual(*check_ok(f))

    def test_obj_member_change_detect(self) -> None:
        def f(foo: Pokeable) -> int:
            '''
            pre: foo.x > 0
            post[]: True
            '''
            foo.poke()
            return foo.x
        self.assertEqual(*check_post_err(f))

    def test_example_second_largest(self) -> None:
        def second_largest(items: List[int]) -> int:
            '''
            pre: len(items) == 3  # (length is to cap runtime)
            post: _ == sorted(items)[-2]
            '''
            next_largest, largest = items[:2]
            if largest < next_largest:
                next_largest, largest = largest, next_largest

            for item in items[2:]:
                if item > largest:
                    largest, next_largest = (item, largest)
                elif item > next_largest:
                    next_largest = item
            return next_largest
        self.assertEqual(*check_ok(second_largest))

    def test_pokeable_class(self) -> None:
        messages = analyze_class(Pokeable)
        self.assertEqual(*check_messages(messages,
                                         state=MessageType.POST_FAIL,
                                         line=50,
                                         column=0))

    def test_person_class(self) -> None:
        messages = analyze_class(Person)
        self.assertEqual(*check_messages(messages, state=MessageType.CONFIRMED))

    def test_extend_namedtuple(self) -> None:
        def f(p: PersonTuple) -> PersonTuple:
            '''
            post: _.age != 222
            '''
            return PersonTuple(p.name, p.age + 1)
        self.assertEqual(*check_fail(f))

    def test_without_typed_attributes(self) -> None:
        def f(p: PersonWithoutAttributes) -> PersonWithoutAttributes:
            '''
            post: _.age != 222
            '''
            return PersonTuple(p.name, p.age + 1)
        self.assertEqual(*check_fail(f))

    def test_property(self) -> None:
        def f(p: Person) -> None:
            '''
            pre: 0 <= p.age < 100
            post[p]: p.birth + p.age == NOW
            '''
            assert p.age == NOW - p.birth
            oldbirth = p.birth
            p.age = p.age + 1
            assert oldbirth == p.birth + 1
        self.assertEqual(*check_ok(f))

    def test_typevar(self) -> None:
        T = TypeVar('T')

        class MaybePair(Generic[T]):
            '''
            inv: (self.left is None) == (self.right is None)
            '''
            left: Optional[T]
            right: Optional[T]

            def setpair(self, left: Optional[T], right: Optional[T]):
                '''post[self]: True'''
                if (left is None) ^ (right is None):
                    raise ValueError('Populate both values or neither value in the pair')
                self.left, self.right = left, right

        messages = analyze_class(MaybePair)
        self.assertEqual(*check_messages(messages, state=MessageType.EXEC_ERR))

    def test_bad_invariant(self):
        class Foo:
            '''
            inv: self.item == 7
            '''

            def do_a_thing(self) -> None:
                pass
        self.assertEqual(*check_messages(analyze_class(Foo),
                                         state=MessageType.PRE_UNSAT))

    def test_expr_name_resolution(self):
        '''
        dataclass() generates several methods. It can be tricky to ensure
        that invariants for these methods can resolve names in the 
        correct namespace.
        '''
        self.assertEqual(*check_messages(analyze_class(ReferenceHoldingClass), state=MessageType.CONFIRMED))

    def test_inheritance_base_class_ok(self):
        self.assertEqual(*check_messages(analyze_class(SmokeDetector), state=MessageType.CONFIRMED))

    def test_super(self):
        class FooDetector(SmokeDetector):
            def signaling_alarm(self, air_samples: List[str]):
                return super().signaling_alarm(air_samples)
        self.assertEqual(*check_messages(analyze_class(FooDetector), state=MessageType.CONFIRMED))

    def test_use_inherited_postconditions(self):
        class CarbonMonoxideDetector(SmokeDetector):
            def signaling_alarm(self, air_samples: List[str]) -> bool:
                '''
                post: implies('carbon_monoxide' in air_samples, _ == True)
                '''
                return 'carbon_monoxide' in air_samples  # fails: does not detect smoke
        self.assertEqual(*check_messages(analyze_class(CarbonMonoxideDetector),
                                         state=MessageType.POST_FAIL))

    def test_inherited_preconditions_overridable(self):
        class SmokeDetectorWithBattery(SmokeDetector):
            _battery_power: int

            def signaling_alarm(self, air_samples: List[str]) -> bool:
                '''
                pre: self._battery_power > 0 or self._is_plugged_in
                '''
                return 'smoke' in air_samples
        self.assertEqual(*check_messages(analyze_class(SmokeDetectorWithBattery),
                                         state=MessageType.CONFIRMED))

    def test_use_subclasses_of_arguments(self):
        # Even though the argument below is typed as the base class, the fact
        # that a faulty implementation exists is enough to produce a
        # counterexample:
        def f(foo: Cat) -> int:
            ''' post: _ == 1 '''
            return foo.size()
        self.assertEqual(*check_fail(f))

    def test_check_parent_conditions(self):
        # Ensure that conditions of parent classes are checked in children
        # even when not overridden.
        class Parent:
            def size(self) -> int:
                return 1
            def amount_smaller(self, other_size: int) -> int:
                '''
                pre: other_size >= 1
                post: _ >= 0
                '''
                return other_size - self.size()
        class Child(Parent):
            def size(self) -> int:
                return 2
        messages = analyze_class(Child)
        self.assertEqual(*check_messages(messages, state=MessageType.POST_FAIL))

    # TODO: precondition strengthening check
    def TODO_test_cannot_strengthen_inherited_preconditions(self):
        class PowerHungrySmokeDetector(SmokeDetector):
            _battery_power: int

            def signaling_alarm(self, air_samples: List[str]) -> bool:
                '''
                pre: self._is_plugged_in
                pre: self._battery_power > 0
                '''
                return 'smoke' in air_samples
        self.assertEqual(*check_messages(analyze_class(PowerHungrySmokeDetector),
                                         state=MessageType.PRE_INVALID))

    def test_container_typevar(self) -> None:
        T = TypeVar('T')

        def f(s: Sequence[T]) -> Dict[T, T]:
            ''' post: len(_) == len(s) '''
            return dict(zip(s, s))
        # (sequence could contain duplicate items)
        self.assertEqual(*check_fail(f))

    def test_typevar_bounds_fail(self) -> None:
        T = TypeVar('T')

        def f(x: T) -> int:
            ''' post:True '''
            return x + 1  # type: ignore
        self.assertEqual(*check_exec_err(f))

    def test_typevar_bounds_ok(self) -> None:
        B = TypeVar('B', bound=int)

        def f(x: B) -> int:
            ''' post:True '''
            return x + 1
        self.assertEqual(*check_ok(f))

    def test_any(self) -> None:
        def f(x: Any) -> bool:
            ''' post: True '''
            return x is None
        self.assertEqual(*check_ok(f))

    def test_meeting_class_preconditions(self) -> None:
        def f() -> int:
            '''
            post: _ == -1
            '''
            pokeable = Pokeable(0)
            pokeable.safe_pokeby(-1)
            return pokeable.x
        result = analyze_function(f)

    def test_enforced_fn_preconditions(self) -> None:
        def f(x: int) -> bool:
            ''' post: _ == True '''
            return bool(fibb(x)) or True
        self.assertEqual(*check_exec_err(f))

    def test_generic_object(self) -> None:
        def f(thing: object):
            ''' post: True '''
            if isinstance(thing, SmokeDetector):
                return thing._is_plugged_in
            return False
        self.assertEqual(*check_ok(f))


class BehaviorsTest(unittest.TestCase):
    def test_syntax_error(self) -> None:
        def f(x: int) -> int:
            ''' pre: x && x '''
        self.assertEqual(*check_messages(analyze_function(f),
                                         state=MessageType.SYNTAX_ERR))

    def test_invalid_raises(self) -> None:
        def f(x: int) -> int:
            ''' raises: NotExistingError '''
            return x
        self.assertEqual(*check_messages(analyze_function(f),
                                         state=MessageType.SYNTAX_ERR))

    def test_raises_ok(self) -> None:
        def f() -> bool:
            '''
            raises: IndexError, NameError
            post: __return__
            '''
            raise IndexError()
            return True
        self.assertEqual(*check_ok(f))

    def test_optional_can_be_none_fail(self) -> None:
        def f(n: Optional[Pokeable]) -> bool:
            ''' post: _ '''
            return isinstance(n, Pokeable)
        self.assertEqual(*check_fail(f))

    def test_implicit_heapref_conversions(self) -> None:
        def f(foo: List[List]) -> None:
            '''
            pre: len(foo) > 0
            post: True
            '''
            foo[0].append(42)
        self.assertEqual(*check_ok(f))

    def test_nonuniform_list_types_1(self) -> None:
        def f(a: List[object], b: List[int]) -> List[object]:
            '''
            pre: len(b) == 5  # constraint for performance
            post: b[0] not in _
            '''
            ret = (a + b[1:])
            return ret
        self.assertEqual(*check_fail(f))

    def test_nonuniform_list_types_2(self) -> None:
        def f(a: List[object], b: List[int]) -> List[object]:
            '''
            pre: len(b) == 5  # constraint for performance
            post: b[-1] not in _
            '''
            return (a + b[:-1])
        self.assertEqual(*check_fail(f))

    def test_varargs_fail(self) -> None:
        def f(x: int, *a: str, **kw: bool) -> int:
            ''' post: _ > x '''
            return x + len(a) + (42 if kw else 0)
        self.assertEqual(*check_fail(f))

    def test_varargs_ok(self) -> None:
        def f(x: int, *a: str, **kw: bool) -> int:
            ''' post: _ >= x '''
            return x + len(a) + (42 if kw else 0)
        self.assertEqual(*check_unknown(f))

    def test_recursive_fn_fail(self) -> None:
        self.assertEqual(*check_fail(fibb))

    def test_recursive_fn_ok(self) -> None:
        self.assertEqual(*check_ok(recursive_example))

    def test_recursive_postcondition_ok(self) -> None:
        def f(x: int) -> int:
            ''' post: _ == f(-x) '''
            return x * x
        self.assertEqual(*check_ok(f))

    def test_recursive_postcondition_enforcement_suspension(self) -> None:
        messages = analyze_class(Measurer)
        self.assertEqual(*check_messages(messages,
                                         state=MessageType.POST_FAIL))

    def test_error_message_has_unmodified_args(self) -> None:
        def f(foo: List[Pokeable]) -> None:
            '''
            pre: len(foo) == 1
            pre: foo[0].x == 10
            post[foo]: foo[0].x == 12
            '''
            foo[0].poke()
        self.assertEqual(*check_messages(
            analyze_function(f),
            state=MessageType.POST_FAIL,
            message='false when calling f(foo = [Pokeable(10)])'))

    # TODO: List[List] involves no HeapRefs
    def TODO_test_potential_circular_references(self) -> None:
        # TODO?: potential aliasing of input argument data?
        def f(foo: List[List], thing: object) -> None:
            '''
            pre: len(foo) == 2
            pre: len(foo[0]) == 1
            pre: len(foo[1]) == 1
            post: len(foo[1]) == 1
            '''
            foo[0].append(object())  # TODO: using 42 yields a z3 sort error
        self.assertEqual(*check_ok(f))

    def test_nonatomic_comparison(self) -> None:
        def f(x: int, l: List[str]) -> bool:
            ''' post: not _ '''
            return l == x
        self.assertEqual(*check_ok(f))

    def test_difficult_equality(self) -> None:
        def f(x: Dict[FrozenSet[float], int]) -> bool:
            ''' post: not _ '''
            return x == {frozenset({10.0}): 1}
        self.assertEqual(*check_fail(f))

    def test_nondeterminisim_detected(self) -> None:
        _GLOBAL_THING = [True]
        def f(i: int) -> int:
            ''' post: True '''
            if i > 0:
                _GLOBAL_THING[0] = not _GLOBAL_THING[0]
            else:
                _GLOBAL_THING[0] = not _GLOBAL_THING[0]
            if _GLOBAL_THING[0]:
                return -i if i < 0 else i
            else:
                return -i if i < 0 else i
        self.assertEqual(*check_exec_err(f, 'NotDeterministic'))

    def test_old_works_in_invariants(self) -> None:
        class FrozenApples:
            ''' inv: self.count == __old__.self.count '''
            count: int
            def add_one(self):
                self.count += 1
        messages = analyze_class(FrozenApples)
        self.assertEqual(*check_messages(messages, state=MessageType.POST_FAIL))

    def test_fallback_when_smt_values_out_themselves(self) -> None:
        def f(items: List[str]) -> str:
            ''' post: True '''
            return ','.join(items)
        self.assertEqual(*check_unknown(f))

    def test_fallback_when_regex_is_used(self) -> None:
        def f(s: str) -> bool:
            ''' post: True '''
            return bool(re.match('(\d+)', s))
        self.assertEqual(*check_unknown(f))


def profile():
    # This is a scratch area to run quick profiles.
    class ProfileTest(unittest.TestCase):
        def test_nonuniform_list_types_2(self) -> None:
            def f(a: List[object], b: List[int]) -> List[object]:
                ...
            self.assertEqual(*check_fail(f))
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(ProfileTest)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    if ('-v' in sys.argv) or ('--verbose' in sys.argv):
        set_debug(True)
    if ('-p' in sys.argv):
        profile()
    else:
        unittest.main()
