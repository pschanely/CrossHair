import builtins
import inspect
import sys
import unittest
from dataclasses import dataclass
from typing import Generic

from crosshair.fnutil import (
    FunctionInfo,
    fn_globals,
    load_function_at_line,
    resolve_signature,
    set_first_arg_type,
)
from crosshair.util import set_debug


def with_invalid_type_annotation(x: "TypeThatIsNotDefined"):  # type: ignore  # noqa: F821
    pass


class FnutilTest(unittest.TestCase):
    def test_fn_globals_on_builtin(self) -> None:
        self.assertIs(fn_globals(zip), builtins.__dict__)

    def test_resolve_signature_invalid_annotations(self) -> None:
        sig = resolve_signature(with_invalid_type_annotation)
        self.assertEqual(sig, "name 'TypeThatIsNotDefined' is not defined", sig)

    def test_resolve_signature_c_function(self) -> None:
        sig = resolve_signature(map)
        self.assertEqual(sig, "No signature available")

    def test_set_first_arg_type(self) -> None:
        sig = inspect.signature(with_invalid_type_annotation)
        typed_sig = set_first_arg_type(sig, int)
        self.assertEqual(typed_sig.parameters["x"].annotation, int)


def toplevelfn():
    pass


# NOTE: We test a dataclass because those will have eval()'d members that appear to
# come from the file "<string>".
@dataclass
class Outer:
    def outerfn(self):
        pass

    class Inner:
        def innerfn(self):
            pass


def test_load_function_at_line():
    mymodule = sys.modules[__name__]
    myfile = __file__
    outerfnline = Outer.outerfn.__code__.co_firstlineno
    innerfnline = Outer.Inner.innerfn.__code__.co_firstlineno
    toplevelfnline = toplevelfn.__code__.co_firstlineno
    assert load_function_at_line(mymodule, myfile, 1) is None
    assert load_function_at_line(mymodule, myfile, outerfnline).name == "outerfn"
    assert load_function_at_line(mymodule, myfile, innerfnline).name == "innerfn"
    assert load_function_at_line(mymodule, myfile, toplevelfnline).name == "toplevelfn"


def test_FunctionInfo_get_callable_on_generic():
    assert FunctionInfo.from_class(Generic, "__class_getitem__").get_callable() is None


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
