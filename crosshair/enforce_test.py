import unittest

from crosshair.condition_parser import Pep316Parser
from crosshair.enforce import *
from crosshair.tracers import COMPOSITE_TRACER
from crosshair.util import set_debug
from typing import IO


def setup_module():
    COMPOSITE_TRACER.__enter__()


def teardown_module():
    COMPOSITE_TRACER.__exit__()


def foo(x: int) -> int:
    """
    pre: 0 <= x <= 100
    post: _ > x
    """
    return x * 2


class Pokeable:
    """
    inv: self.x >= 0
    """

    x: int = 1

    def poke(self) -> None:
        self.x += 1

    def pokeby(self, amount: int) -> None:
        """
        pre: amount >= 0
        """
        self.x += amount


def same_thing(thing: object) -> object:
    """ post: __old__.thing == _ """
    # If `thing` isn't copyable, it won't be available in `__old__`.
    # In this case, enforcement will fail with an AttributeError.
    return thing


class CoreTest(unittest.TestCase):
    def test_enforce_conditions(self) -> None:
        env = {"foo": foo}
        self.assertEqual(foo(-1), -2)  # unchecked
        with EnforcedConditions(Pep316Parser(), env) as enf, enf.enabled_enforcement():
            self.assertEqual(env["foo"](50), 100)
            with self.assertRaises(PreconditionFailed):
                env["foo"](-1)
            with self.assertRaises(PostconditionFailed):
                env["foo"](0)

    def test_class_enforce(self) -> None:
        env = {"Pokeable": Pokeable}
        old_id = id(Pokeable.poke)
        Pokeable().pokeby(-1)  # no exception (yet!)
        with EnforcedConditions(Pep316Parser(), env) as enf, enf.enabled_enforcement():
            Pokeable().poke()
            with self.assertRaises(PreconditionFailed):
                Pokeable().pokeby(-1)

    def test_enforce_on_uncopyable_value(self) -> None:
        class NotCopyable:
            def __copy__(self):
                raise TypeError("not copyable")

        not_copyable = NotCopyable()
        env = {"same_thing": same_thing}
        with EnforcedConditions(Pep316Parser(), env) as enf, enf.enabled_enforcement():
            with self.assertRaises(AttributeError):
                env["same_thing"](not_copyable)


class BaseFooable:
    def foo(self, x: int):
        """ pre: x > 100 """

    def foo_only_in_super(self, x: int):
        """ pre: x > 100 """

    @classmethod
    def class_foo(cls, x: int):
        """ pre: x > 100 """

    @staticmethod
    def static_foo(x: int):
        """ pre: x > 100 """


class DerivedFooable(BaseFooable):
    def foo(self, x: int):
        """ pre: x > 0 """

    @classmethod
    def class_foo(cls, x: int):
        """ pre: x > 0 """

    @staticmethod
    def static_foo(x: int):
        """ pre: x > 0 """


class TrickyCasesTest(unittest.TestCase):
    def test_attrs_restored_properly(self) -> None:
        orig_class_dict = DerivedFooable.__dict__.copy()
        env = {"DerivedFooable": DerivedFooable}
        with EnforcedConditions(Pep316Parser(), env) as enf, enf.enabled_enforcement():
            pass
        for k, v in orig_class_dict.items():
            self.assertIs(
                DerivedFooable.__dict__[k], v, f'member "{k}" changed afer encforcement'
            )

    def test_enforcement_of_class_methods(self) -> None:
        env = {"DerivedFooable": BaseFooable}
        with EnforcedConditions(Pep316Parser(), env) as enf, enf.enabled_enforcement():
            with self.assertRaises(PreconditionFailed):
                BaseFooable.class_foo(50)
        env = {"DerivedFooable": DerivedFooable}
        with EnforcedConditions(Pep316Parser(), env) as enf, enf.enabled_enforcement():
            DerivedFooable.class_foo(50)

    def test_enforcement_of_static_methods(self) -> None:
        env = {"DerivedFooable": DerivedFooable}
        with EnforcedConditions(Pep316Parser(), env) as enf, enf.enabled_enforcement():
            DerivedFooable.static_foo(50)
            with self.assertRaises(PreconditionFailed):
                BaseFooable.static_foo(50)

    def test_super_method_enforced(self) -> None:
        env = {"DerivedFooable": DerivedFooable}
        with EnforcedConditions(Pep316Parser(), env) as enf, enf.enabled_enforcement():
            with self.assertRaises(PreconditionFailed):
                DerivedFooable().foo_only_in_super(50)
            with self.assertRaises(PreconditionFailed):
                DerivedFooable().foo(-1)
            # Derived class has a weaker precondition, so this is OK:
            DerivedFooable().foo(50)


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
