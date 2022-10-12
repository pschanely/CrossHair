import dataclasses
import inspect
import re
import sys
import time
import unittest
from typing import *

import pytest  # type: ignore

import crosshair
from crosshair import type_repo
from crosshair.core import (
    deep_realize,
    get_constructor_signature,
    is_deeply_immutable,
    proxy_for_class,
    proxy_for_type,
    run_checkables,
)
from crosshair.core_and_libs import (
    AnalysisKind,
    List,
    MessageType,
    analyze_any,
    analyze_class,
    analyze_function,
    standalone_statespace,
)
from crosshair.fnutil import FunctionInfo, walk_qualname
from crosshair.options import DEFAULT_OPTIONS, AnalysisOptionSet
from crosshair.statespace import (
    CANNOT_CONFIRM,
    CONFIRMED,
    EXEC_ERR,
    POST_ERR,
    POST_FAIL,
)
from crosshair.test_util import check_exec_err, check_messages, check_states
from crosshair.tracers import NoTracing
from crosshair.util import CrosshairInternal, set_debug

try:
    import icontract
except BaseException:
    icontract = None  # type: ignore

try:
    import hypothesis
except BaseException:
    hypothesis = None  # type: ignore


@dataclasses.dataclass
class Pokeable:
    """
    inv: self.x >= 0
    """

    x: int = 1

    def poke(self) -> None:
        """
        post[self]: True
        """
        self.x += 1

    def wild_pokeby(self, amount: int) -> None:
        """
        post[self]: True
        """
        self.x += amount

    def safe_pokeby(self, amount: int) -> None:
        """
        pre: amount >= 0
        post[self]: True
        """
        self.x += amount


def remove_smallest_with_asserts(numbers: List[int]) -> None:
    assert len(numbers) > 0
    smallest = min(numbers)
    numbers.remove(smallest)
    assert len(numbers) == 0 or min(numbers) > smallest


if icontract:

    @icontract.snapshot(lambda lst: lst[:])
    @icontract.ensure(lambda OLD, lst, value: lst == OLD.lst + [value])
    def icontract_appender(lst: List[int], value: int) -> None:
        lst.append(value)
        lst.append(1984)  # bug

    @icontract.invariant(lambda self: self.x > 0)
    class IcontractA(icontract.DBC):
        def __init__(self) -> None:
            self.x = 10

        @icontract.require(lambda x: x % 2 == 0)
        def weakenedfunc(self, x: int) -> None:
            pass

    @icontract.invariant(lambda self: self.x < 100)
    class IcontractB(IcontractA):
        def break_parent_invariant(self):
            self.x = -1

        def break_my_invariant(self):
            self.x = 101

        @icontract.require(lambda x: x % 3 == 0)
        def weakenedfunc(self, x: int) -> None:
            pass


class ShippingContainer:
    container_weight = 4

    def total_weight(self) -> int:
        """post: _ < 10"""
        return self.cargo_weight() + self.container_weight

    def cargo_weight(self) -> int:
        return 0

    def __repr__(self):
        return type(self).__name__ + "()"


class OverloadedContainer(ShippingContainer):
    """
    We use this example to demonstrate messaging when an override breaks
    the contract of a different method.
    """

    def cargo_weight(self) -> int:
        return 9


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


@dataclasses.dataclass
class Person:
    """
    Contains various features that we expect to be successfully checkable.

    inv: True
    """

    name: str
    birth: int

    def _getage(self):
        return NOW - self.birth

    def _setage(self, newage):
        self.birth = NOW - newage

    def _delage(self):
        del self.birth

    age = property(_getage, _setage, _delage, "Age of person")

    def abstract_operation(self):
        """
        post: False # doesn't error because the method is "abstract"
        """
        raise NotImplementedError

    def a_regular_method(self):
        """post: True"""

    @classmethod
    def a_class_method(cls, x):
        """post: cls == Person"""

    @staticmethod
    def a_static_method():
        """post: True"""


class AirSample:
    # NOTE: we don't use an enum here because we want to use pure symbolic containers
    # in our tests.
    CLEAN = 0
    SMOKE = 1
    CO2 = 2


@dataclasses.dataclass
class SmokeDetector:
    """inv: not (self._is_plugged_in and self._in_original_packaging)"""

    _in_original_packaging: bool
    _is_plugged_in: bool

    def signaling_alarm(self, air_samples: List[int]) -> bool:
        """
        pre: self._is_plugged_in
        post: implies(AirSample.SMOKE in air_samples, _ == True)
        """
        return AirSample.SMOKE in air_samples


class Measurer:
    def measure(self, x: int) -> str:
        """
        post: _ == self.measure(-x)
        """
        return "small" if x <= 10 else "large"


def _unused_fn(x: int) -> "ClassWithExplicitSignature":
    ...


class ClassWithExplicitSignature:
    __signature__ = inspect.signature(_unused_fn)

    def __init__(self, *a):
        self.x = a[0]


A_REFERENCED_THING = 42


@dataclasses.dataclass(repr=False)
class ReferenceHoldingClass:
    """
    inv: self.item != A_REFERENCED_THING
    """

    item: str


def fibb(x: int) -> int:
    """
    pre: x>=0
    post[]: _ < 3
    """
    if x <= 2:
        return 1
    r1, r2 = fibb(x - 1), fibb(x - 2)
    ret = r1 + r2
    return ret


def reentrant_precondition(minx: int):
    """pre: reentrant_precondition(minx - 1)"""
    return minx <= 10


def recursive_example(x: int) -> bool:
    """
    pre: x >= 0
    post[]:
        __old__.x >= 0  # just to confirm __old__ works in recursive cases
        _ == True
    """
    if x == 0:
        return True
    else:
        return recursive_example(x - 1)


class RegularInt:
    def __new__(self, num: "int"):
        return num


class UnitTests(unittest.TestCase):
    def test_get_constructor_signature_with_new(self):
        self.assertIs(RegularInt(7), 7)
        params = get_constructor_signature(RegularInt).parameters
        self.assertEqual(len(params), 1)
        self.assertEqual(params["num"].name, "num")
        self.assertEqual(params["num"].annotation, int)


def test_proxy_alone() -> None:
    def f(pokeable: Pokeable) -> None:
        """
        post[pokeable]: pokeable.x > 0
        """
        pokeable.poke()

    check_states(f, CONFIRMED)


def test_proxy_in_list() -> None:
    def f(pokeables: List[Pokeable]) -> None:
        """
        pre: len(pokeables) == 1
        post: all(p.x > 0 for p in pokeables)
        """
        for pokeable in pokeables:
            pokeable.poke()

    check_states(f, CONFIRMED)


def test_class_with_explicit_signature() -> None:
    def f(c: ClassWithExplicitSignature) -> int:
        """post: _ != 42"""
        return c.x

    # pydantic sets __signature__ on the class, so we look for that as well as on
    # __init__ (see https://github.com/samuelcolvin/pydantic/pull/1034)
    check_states(f, POST_FAIL)


def test_preconditioned_init():
    class Penguin:
        _age: int

        def __init__(self, age: int):
            """pre: age >= 1"""
            self._age = age

    def f(p: Penguin) -> int:
        """post: _ != 0"""
        return p._age

    check_states(f, CONFIRMED)


def test_class_proxies_are_created_through_constructor():
    class Penguin:
        can_swim: bool

        def __init__(self):
            self.can_swim = True

    with standalone_statespace as space:
        with NoTracing():  # (because following function resumes tracing)
            p = proxy_for_class(Penguin, "p")
        # `can_swim` is locked to True
        assert p.can_swim is True


class ObjectsTest(unittest.TestCase):
    def test_obj_member_fail(self) -> None:
        def f(foo: Pokeable) -> int:
            """
            pre: 0 <= foo.x <= 4
            post[foo]: _ < 5
            """
            foo.poke()
            foo.poke()
            return foo.x

        check_states(f, POST_FAIL)

    def test_obj_member_nochange_ok(self) -> None:
        def f(foo: Pokeable) -> int:
            """post: _ == foo.x"""
            return foo.x

        check_states(f, CONFIRMED)

    def test_obj_member_change_ok(self) -> None:
        def f(foo: Pokeable) -> int:
            """
            pre: foo.x >= 0
            post[foo]: foo.x >= 2
            """
            foo.poke()
            foo.poke()
            return foo.x

        check_states(f, CONFIRMED)

    def test_obj_member_change_detect(self) -> None:
        def f(foo: Pokeable) -> int:
            """
            pre: foo.x > 0
            post[]: True
            """
            foo.poke()
            return foo.x

        check_states(f, POST_ERR)

    def test_example_second_largest(self) -> None:
        def second_largest(items: List[int]) -> int:
            """
            pre: len(items) == 3  # (length is to cap runtime)
            post: _ == sorted(items)[-2]
            """
            next_largest, largest = items[:2]
            if largest < next_largest:
                next_largest, largest = largest, next_largest

            for item in items[2:]:
                if item > largest:
                    largest, next_largest = (item, largest)
                elif item > next_largest:
                    next_largest = item
            return next_largest

        check_states(second_largest, CONFIRMED)

    def test_pokeable_class(self) -> None:
        messages = analyze_class(Pokeable)
        line = Pokeable.wild_pokeby.__code__.co_firstlineno
        self.assertEqual(
            *check_messages(messages, state=MessageType.POST_FAIL, line=line, column=0)
        )

    def test_person_class(self) -> None:
        messages = analyze_class(Person)
        self.assertEqual(*check_messages(messages, state=MessageType.CONFIRMED))

    def test_methods_directly(self) -> None:
        # Running analysis on individual methods directly works a little
        # differently, especially for staticmethod/classmethod. Confirm these
        # don't explode:
        messages = analyze_any(
            walk_qualname(Person, "a_regular_method"),
            AnalysisOptionSet(per_condition_timeout=5),
        )
        self.assertEqual(*check_messages(messages, state=MessageType.CONFIRMED))

    def test_class_method(self) -> None:
        messages = analyze_any(
            walk_qualname(Person, "a_class_method"),
            AnalysisOptionSet(per_condition_timeout=5),
        )
        self.assertEqual(*check_messages(messages, state=MessageType.CONFIRMED))

    def test_static_method(self) -> None:
        messages = analyze_any(
            walk_qualname(Person, "a_static_method"),
            AnalysisOptionSet(per_condition_timeout=5),
        )
        self.assertEqual(*check_messages(messages, state=MessageType.CONFIRMED))

    def test_extend_namedtuple(self) -> None:
        def f(p: PersonTuple) -> PersonTuple:
            """
            post: _.age != 222
            """
            return PersonTuple(p.name, p.age + 1)

        check_states(f, POST_FAIL)

    def test_without_typed_attributes(self) -> None:
        def f(p: PersonWithoutAttributes) -> PersonWithoutAttributes:
            """
            post: _.age != 222
            """
            return PersonTuple(p.name, p.age + 1)  # type: ignore

        check_states(f, POST_FAIL)

    def test_property(self) -> None:
        def f(p: Person) -> None:
            """
            pre: 0 <= p.age < 100
            post[p]: p.birth + p.age == NOW
            """
            assert p.age == NOW - p.birth
            oldbirth = p.birth
            p.age = p.age + 1
            assert oldbirth == p.birth + 1

        check_states(f, CONFIRMED)

    def test_readonly_property_contract(self) -> None:
        class Clock:
            @property
            def time(self) -> int:
                """post: _ == self.time"""
                return 120

        messages = analyze_class(Clock)
        self.assertEqual(*check_messages(messages, state=MessageType.CONFIRMED))

    def test_typevar(self) -> None:
        T = TypeVar("T")

        @dataclasses.dataclass
        class MaybePair(Generic[T]):
            """
            inv: (self.left is None) == (self.right is None)
            """

            left: Optional[T]
            right: Optional[T]

            def setpair(self, left: Optional[T], right: Optional[T]):
                """post[self]: True"""
                if (left is None) ^ (right is None):
                    raise ValueError(
                        "Populate both values or neither value in the pair"
                    )
                self.left, self.right = left, right

        messages = analyze_function(
            FunctionInfo(MaybePair, "setpair", MaybePair.__dict__["setpair"])
        )
        self.assertEqual(*check_messages(messages, state=MessageType.EXEC_ERR))

    def test_bad_invariant(self):
        class WithBadInvariant:
            """
            inv: self.item == 7
            """

            def do_a_thing(self) -> None:
                pass

        self.assertEqual(
            *check_messages(
                analyze_class(WithBadInvariant), state=MessageType.PRE_UNSAT
            )
        )

    def test_expr_name_resolution(self):
        """
        dataclass() generates several methods. It can be tricky to ensure
        that invariants for these methods can resolve names in the
        correct namespace.
        """
        self.assertEqual(
            *check_messages(
                analyze_class(ReferenceHoldingClass), state=MessageType.CONFIRMED
            )
        )

    def test_inheritance_base_class_ok(self):
        self.assertEqual(
            *check_messages(analyze_class(SmokeDetector), state=MessageType.CONFIRMED)
        )

    def test_super(self):
        class FooDetector(SmokeDetector):
            def signaling_alarm(self, air_samples: List[int]):
                return super().signaling_alarm(air_samples)

        self.assertEqual(
            *check_messages(analyze_class(FooDetector), state=MessageType.CONFIRMED)
        )

    def test_use_inherited_postconditions(self):
        class CarbonMonoxideDetector(SmokeDetector):
            def signaling_alarm(self, air_samples: List[int]) -> bool:
                """
                post: implies(AirSample.CO2 in air_samples, _ == True)
                """
                return AirSample.CO2 in air_samples  # fails: does not detect smoke

        self.assertEqual(
            *check_messages(
                analyze_class(CarbonMonoxideDetector), state=MessageType.POST_FAIL
            )
        )

    def test_inherited_preconditions_overridable(self):
        @dataclasses.dataclass
        class SmokeDetectorWithBattery(SmokeDetector):
            _battery_power: int

            def signaling_alarm(self, air_samples: List[int]) -> bool:
                """
                pre: self._battery_power > 0 or self._is_plugged_in
                post: self._battery_power > 0
                """
                return AirSample.SMOKE in air_samples

        self.assertEqual(
            *check_messages(
                analyze_class(SmokeDetectorWithBattery), state=MessageType.POST_FAIL
            )
        )

    def test_use_subclasses_of_arguments(self):
        # Even though the argument below is typed as the base class, the fact
        # that a faulty implementation exists is enough to produce a
        # counterexample:
        def f(foo: Cat) -> int:
            """post: _ == 1"""
            return foo.size()

        # Type repo doesn't load crosshair classes by default; load manually:
        type_repo._add_class(Cat)
        type_repo._add_class(BiggerCat)
        check_states(f, POST_FAIL)

    def test_does_not_report_with_actual_repr(self):
        def f(foo: BiggerCat) -> int:
            """post: False"""
            return foo.size()

        (actual, expected) = check_messages(
            analyze_function(f),
            state=MessageType.POST_FAIL,
            message="false when calling f(BiggerCat()) " "(which returns 2)",
        )
        assert expected == actual

    def test_check_parent_conditions(self):
        # Ensure that conditions of parent classes are checked in children
        # even when not overridden.
        class Parent:
            def size(self) -> int:
                return 1

            def amount_smaller(self, other_size: int) -> int:
                """
                pre: other_size >= 1
                post: _ >= 0
                """
                return other_size - self.size()

        class Child(Parent):
            def size(self) -> int:
                return 2

        messages = analyze_class(Child)
        self.assertEqual(*check_messages(messages, state=MessageType.POST_FAIL))

    @pytest.mark.skipif(sys.version_info < (3, 8), reason="Python 3.8+ required")
    def test_final_with_concrete_proxy(self):
        from typing import Final

        class FinalCat:
            legs: Final[int] = 4

            def __repr__(self):
                return f"FinalCat with {self.legs} legs"

        def f(cat: FinalCat, strides: int) -> int:
            """
            pre: strides > 0
            post: __return__ >= 4
            """
            return strides * cat.legs

        check_states(f, CONFIRMED)

    # TODO: precondition strengthening check
    def TODO_test_cannot_strengthen_inherited_preconditions(self):
        class PowerHungrySmokeDetector(SmokeDetector):
            _battery_power: int

            def signaling_alarm(self, air_samples: List[int]) -> bool:
                """
                pre: self._is_plugged_in
                pre: self._battery_power > 0
                """
                return AirSample.SMOKE in air_samples

        self.assertEqual(
            *check_messages(
                analyze_class(PowerHungrySmokeDetector), state=MessageType.PRE_INVALID
            )
        )

    def test_container_typevar(self) -> None:
        T = TypeVar("T")

        def f(s: Sequence[T]) -> Dict[T, T]:
            """post: len(_) == len(s)"""
            return dict(zip(s, s))

        # (sequence could contain duplicate items)
        check_states(f, POST_FAIL)

    def test_typevar_bounds_fail(self) -> None:
        T = TypeVar("T")

        def f(x: T) -> int:
            """post:True"""
            return x + 1  # type: ignore

        check_states(f, EXEC_ERR)

    def test_typevar_bounds_ok(self) -> None:
        B = TypeVar("B", bound=int)

        def f(x: B) -> int:
            """post:True"""
            return x + 1

        check_states(f, CONFIRMED)

    def test_any(self) -> None:
        def f(x: Any) -> bool:
            """post: True"""
            return x is None

        check_states(f, CONFIRMED)

    def test_meeting_class_preconditions(self) -> None:
        def f() -> int:
            """
            post: _ == -1
            """
            pokeable = Pokeable(0)
            pokeable.safe_pokeby(-1)
            return pokeable.x

        analyze_function(f)
        # TODO: this doesn't test anything?

    def test_enforced_fn_preconditions(self) -> None:
        def f(x: int) -> bool:
            """post: _ == True"""
            return bool(fibb(x)) or True

        check_states(f, EXEC_ERR)

    def test_generic_object(self) -> None:
        def f(thing: object):
            """post: True"""
            if isinstance(thing, SmokeDetector):
                return thing._is_plugged_in
            return False

        check_states(f, CANNOT_CONFIRM)


def get_natural_number() -> int:
    """post: _ >= 0"""
    # crosshair: specs_complete=True
    return 1


def test_specs_complete():
    def f() -> int:
        """post: _"""
        return get_natural_number()

    (actual, expected) = check_messages(
        analyze_function(f),
        state=MessageType.POST_FAIL,
        message="false when calling f() "
        "with crosshair.patch_to_return({"
        "crosshair.core_test.get_natural_number: [0]}) "
        "(which returns 0)",
    )
    assert actual == expected

    # also check that it reproduces!:
    assert get_natural_number() == 1
    with crosshair.patch_to_return({crosshair.core_test.get_natural_number: [0]}):
        assert get_natural_number() == 0


def test_access_class_method_on_symbolic_type():
    with standalone_statespace as space:
        person = proxy_for_type(Type[Person], "p")
        person.a_class_method(42)  # Just check that this don't explode


class BehaviorsTest(unittest.TestCase):
    def test_syntax_error(self) -> None:
        def f(x: int) -> int:
            """pre: x && x"""

        self.assertEqual(
            *check_messages(analyze_function(f), state=MessageType.SYNTAX_ERR)
        )

    def test_raises_ok(self) -> None:
        def f() -> bool:
            """
            raises: IndexError, NameError
            post: __return__
            """
            raise IndexError()
            return True

        check_states(f, CONFIRMED)

    def test_optional_can_be_none_fail(self) -> None:
        def f(n: Optional[Pokeable]) -> bool:
            """post: _"""
            return isinstance(n, Pokeable)

        check_states(f, POST_FAIL)

    def test_implicit_heapref_conversions(self) -> None:
        def f(foo: List[List]) -> None:
            """
            pre: len(foo) > 0
            post: True
            """
            foo[0].append(42)

        check_states(f, CONFIRMED)

    def test_nonuniform_list_types_1(self) -> None:
        def f(a: List[object], b: List[int]) -> List[object]:
            """
            pre: len(b) == 5  # constraint for performance
            post: b[0] not in _
            """
            ret = a + b[1:]  # type: ignore
            return ret

        check_states(f, POST_FAIL)

    def test_nonuniform_list_types_2(self) -> None:
        def f(a: List[object], b: List[int]) -> List[object]:
            """
            pre: len(b) == 5  # constraint for performance
            post: b[-1] not in _
            """
            return a + b[:-1]  # type: ignore

        check_states(f, POST_FAIL)

    def test_varargs_fail(self) -> None:
        def f(x: int, *a: str, **kw: bool) -> int:
            """post: _ > x"""
            return x + len(a) + (42 if kw else 0)

        check_states(f, POST_FAIL)

    def test_varargs_ok(self) -> None:
        def f(x: int, *a: str, **kw: bool) -> int:
            """post: _ >= x"""
            return x + len(a) + (42 if kw else 0)

        check_states(f, CANNOT_CONFIRM)

    def test_recursive_fn_fail(self) -> None:
        check_states(fibb, POST_FAIL)

    def test_recursive_fn_ok(self) -> None:
        check_states(recursive_example, CONFIRMED)

    def test_recursive_postcondition_ok(self) -> None:
        def f(x: int) -> int:
            """post: _ == f(-x)"""
            return x * x

        check_states(f, CONFIRMED)

    def test_reentrant_precondition(self) -> None:
        # Really, we're just ensuring that we don't stack-overflow here.
        check_states(reentrant_precondition, CONFIRMED)

    def test_recursive_postcondition_enforcement_suspension(self) -> None:
        messages = analyze_class(Measurer)
        self.assertEqual(*check_messages(messages, state=MessageType.POST_FAIL))

    def test_short_circuiting(self) -> None:
        # Some operations are hard to deal with symbolically, like hashes.
        # CrossHair will sometimes "short-circuit" functions, in hopes that the
        # function body isn't required to prove the postcondition.
        # This is an example of such a case.
        def f(x: str) -> int:
            """post: _ == 0"""
            a = hash(x)
            b = 7
            # This is zero no matter what the hashes are:
            return (a + b) - (b + a)

        check_states(f, CONFIRMED)

    def test_error_message_in_unrelated_method(self) -> None:
        messages = analyze_class(OverloadedContainer)
        line = ShippingContainer.total_weight.__code__.co_firstlineno + 1
        self.assertEqual(
            *check_messages(
                messages,
                state=MessageType.POST_FAIL,
                message="false when calling total_weight(OverloadedContainer()) (which returns 13)",
                line=line,
            )
        )

    def test_error_message_has_unmodified_args(self) -> None:
        def f(foo: List[Pokeable]) -> None:
            """
            pre: len(foo) == 1
            pre: foo[0].x == 10
            post[foo]: foo[0].x == 12
            """
            foo[0].poke()

        self.assertEqual(
            *check_messages(
                analyze_function(f),
                state=MessageType.POST_FAIL,
                message="false when calling f([Pokeable(x = 10)])",
            )
        )

    # TODO: List[List] involves no HeapRefs
    def TODO_test_potential_circular_references(self) -> None:
        # TODO?: potential aliasing of input argument data?
        def f(foo: List[List], thing: object) -> None:
            """
            pre: len(foo) == 2
            pre: len(foo[0]) == 1
            pre: len(foo[1]) == 1
            post: len(foo[1]) == 1
            """
            foo[0].append(object())  # TODO: using 42 yields a z3 sort error

        check_states(f, CONFIRMED)

    def test_nonatomic_comparison(self) -> None:
        def f(x: int, ls: List[str]) -> bool:
            """post: not _"""
            return ls == x

        check_states(f, CONFIRMED)

    def test_difficult_equality(self) -> None:
        def f(x: Dict[FrozenSet[float], int]) -> bool:
            """post: not _"""
            return x == {frozenset({10.0}): 1}

        check_states(f, POST_FAIL)

    def test_nondeterminisim_detected(self) -> None:
        _GLOBAL_THING = [True]

        def f(i: int) -> int:
            """post: True"""
            _GLOBAL_THING[0] = not _GLOBAL_THING[0]
            if _GLOBAL_THING[0]:
                return -i if i < 0 else i
            else:
                return -i if i < 0 else i

        self.assertEqual(*check_exec_err(f, "NotDeterministic"))

    def test_old_works_in_invariants(self) -> None:
        @dataclasses.dataclass
        class FrozenApples:
            """inv: self.count == __old__.self.count"""

            count: int

            def add_one(self):
                self.count += 1

        messages = analyze_class(FrozenApples)
        self.assertEqual(*check_messages(messages, state=MessageType.POST_FAIL))

        # Also confirm we can create one as an argument:
        def f(a: FrozenApples) -> int:
            """post: True"""
            return 0

        check_states(f, CONFIRMED)

    def test_class_patching_is_undone(self) -> None:
        # CrossHair does a lot of monkey matching of classes
        # with contracts. Ensure that gets undone.
        original_container = ShippingContainer.__dict__.copy()
        original_overloaded = OverloadedContainer.__dict__.copy()
        run_checkables(analyze_class(OverloadedContainer))
        for k, v in original_container.items():
            self.assertIs(ShippingContainer.__dict__[k], v)
        for k, v in original_overloaded.items():
            self.assertIs(OverloadedContainer.__dict__[k], v)

    def test_fallback_when_smt_values_out_themselves(self) -> None:
        def f(items: List[str]) -> str:
            """post: True"""
            return ",".join(items)

        check_states(f, CANNOT_CONFIRM)

    def test_unrelated_regex(self) -> None:
        def f(s: str) -> bool:
            """post: True"""
            return bool(re.match(r"(\d+)", s))

        check_states(f, CANNOT_CONFIRM)

    @pytest.mark.skip("Python 3.9+ is not supported yet")
    def test_new_style_type_hints(self):
        def f(ls: list[int]) -> List[int]:
            """
            pre: len(ls) == 2
            post: _[0] != 'a'
            """
            return ls

        check_states(f, CONFIRMED)


if icontract:

    class TestIcontract(unittest.TestCase):
        def test_icontract_basic(self):
            @icontract.ensure(lambda result, x: result > x)
            def some_func(x: int, y: int = 5) -> int:
                return x - y

            check_states(some_func, POST_FAIL)

        def test_icontract_snapshots(self):
            messages = analyze_function(
                icontract_appender,
                DEFAULT_OPTIONS,
            )
            line = icontract_appender.__wrapped__.__code__.co_firstlineno + 1
            self.assertEqual(
                *check_messages(
                    messages, state=MessageType.POST_FAIL, line=line, column=0
                )
            )

        def test_icontract_weaken(self):
            @icontract.require(lambda x: x in (2, 3))
            @icontract.ensure(lambda: True)
            def trynum(x: int):
                IcontractB().weakenedfunc(x)

            check_states(trynum, CONFIRMED)

        def test_icontract_class(self):
            messages = run_checkables(
                analyze_class(
                    IcontractB,
                    # TODO: why is this required?
                    DEFAULT_OPTIONS.overlay(analysis_kind=[AnalysisKind.icontract]),
                )
            )
            messages = {
                (m.state, m.line, m.message)
                for m in messages
                if m.state != MessageType.CONFIRMED
            }
            line_gt0 = (
                IcontractB.break_parent_invariant.__wrapped__.__code__.co_firstlineno
            )
            line_lt100 = (
                IcontractB.break_my_invariant.__wrapped__.__code__.co_firstlineno
            )
            self.assertEqual(
                messages,
                {
                    (
                        MessageType.POST_FAIL,
                        line_gt0,
                        '"@icontract.invariant(lambda self: self.x > 0)" yields false '
                        "when calling break_parent_invariant(IcontractB())",
                    ),
                    (
                        MessageType.POST_FAIL,
                        line_lt100,
                        '"@icontract.invariant(lambda self: self.x < 100)" yields false '
                        "when calling break_my_invariant(IcontractB())",
                    ),
                },
            )

        def test_icontract_nesting(self):
            @icontract.require(lambda name: name.startswith("a"))
            def innerfn(name: str):
                pass

            @icontract.ensure(lambda: True)
            @icontract.require(lambda name: len(name) > 0)
            def outerfn(name: str):
                innerfn("00" + name)

            self.assertEqual(
                *check_exec_err(
                    outerfn,
                    message_prefix="PreconditionFailed",
                )
            )


if hypothesis:

    @hypothesis.given(hypothesis.strategies.booleans())
    def foo(x):
        assert x

    def test_hypothesis_counterexample_text():
        messages = analyze_function(
            foo,
            DEFAULT_OPTIONS.overlay(
                analysis_kind=[AnalysisKind.hypothesis],
                max_iterations=10,
                per_condition_timeout=20,
            ),
        )
        actual, expected = check_messages(
            messages,
            state=MessageType.EXEC_ERR,
            message="AssertionError: assert False when calling foo(False)",
        )
        assert actual == expected


class TestAssertsMode(unittest.TestCase):
    def test_asserts(self):
        messages = analyze_function(
            remove_smallest_with_asserts,
            DEFAULT_OPTIONS.overlay(
                analysis_kind=[AnalysisKind.asserts],
                max_iterations=10,
                per_condition_timeout=5,
            ),
        )
        line = remove_smallest_with_asserts.__code__.co_firstlineno + 4
        self.assertEqual(
            *check_messages(messages, state=MessageType.EXEC_ERR, line=line, column=0)
        )


def test_unpickable_args() -> None:
    from threading import RLock  # RLock objects aren't copyable

    @dataclasses.dataclass
    class WithUnpickleableArg:
        x: int
        lock: RLock

    def dothing(foo: WithUnpickleableArg) -> int:
        """
        post: __return__ != 42
        """
        return foo.x

    check_states(dothing, POST_FAIL)


def test_deep_realize():
    with standalone_statespace as space:
        x = proxy_for_type(int, "x")
        space.add(x.var == 4)

    @dataclasses.dataclass
    class Woo:
        stuff: Dict[str, int]

    woo = Woo({"": x})
    assert type(woo.stuff[""]) is not int
    realized = deep_realize(woo)
    assert type(realized.stuff[""]) is int
    assert realized.stuff[""] == 4


@pytest.mark.parametrize(
    "o", (4, "foo", 23.1, None, (12,), frozenset({1, 2}), ((), (4,)))
)
def test_is_deeply_immutable(o):
    with standalone_statespace:
        assert is_deeply_immutable(o)


@pytest.mark.parametrize("o", ({}, {1: 2}, [], (3, []), ("foo", (3, []))))
def test_is_not_deeply_immutable(o):
    with standalone_statespace:
        assert not is_deeply_immutable(o)


def profile():
    # This is a scratch area to run quick profiles.
    def f(x: int) -> int:
        """
        post: _ != 123456
        """
        return hash(x)

    check_states(f, AnalysisOptionSet(max_iterations=20)) == {
        MessageType.CANNOT_CONFIRM
    }


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    if "-p" in sys.argv:
        import time

        t0 = time.time()
        profile()
        print("check seconds:", time.time() - t0)
    elif "-t" in sys.argv:
        import cProfile

        cProfile.run("profile()", "out.pprof")
    else:
        unittest.main()
