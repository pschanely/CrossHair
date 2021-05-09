import inspect
import unittest

from crosshair.fnutil import *
from crosshair.util import set_debug, debug
from crosshair.test_util import this_line_number


def with_invalid_type_annotation(x: "TypeThatIsNotDefined"):  # type: ignore
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


_TOP_LINE = this_line_number()  # returns the line number where it's called


class Outer:
    def outerfn(self):  # (line offset 4)
        pass

    class Inner:
        def innerfn(self):  # (line offset 8)
            pass


def toplevelfn():
    pass  # (line offset 13)


def test_load_function_at_line():
    mymodule = sys.modules[__name__]
    myfile = __file__
    assert load_function_at_line(mymodule, myfile, _TOP_LINE + 1) is None
    assert load_function_at_line(mymodule, myfile, _TOP_LINE + 4).name == "outerfn"
    assert load_function_at_line(mymodule, myfile, _TOP_LINE + 8).name == "innerfn"
    assert load_function_at_line(mymodule, myfile, _TOP_LINE + 13).name == "toplevelfn"


if __name__ == "__main__":
    if ("-v" in sys.argv) or ("--verbose" in sys.argv):
        set_debug(True)
    unittest.main()
