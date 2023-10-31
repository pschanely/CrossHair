import codecs
import sys
from io import BytesIO

import pytest  # type: ignore

from crosshair.core_and_libs import MessageType, analyze_function, run_checkables
from crosshair.options import AnalysisOptionSet
from crosshair.test_util import ResultComparison, compare_results

_ERROR_HANDLERS = ["strict", "replace", "ignore"]

# crosshair: max_iterations=20


def check_encode_ascii(string: str, errors: str) -> ResultComparison:
    """
    pre: errors in _ERROR_HANDLERS
    post: _
    """
    return compare_results(lambda s, e: s.encode("ascii", e), string, errors)


def check_encode_latin1(string: str, errors: str) -> ResultComparison:
    """
    pre: errors in _ERROR_HANDLERS
    post: _
    """
    return compare_results(lambda s, e: s.encode("latin1", e), string, errors)


def check_encode_utf8(string: str, errors: str) -> ResultComparison:
    """
    pre: errors in _ERROR_HANDLERS
    post: _
    """
    return compare_results(lambda s, e: s.encode("utf8", e), string, errors)


def check_decode_ascii(bytestring: bytes, errors: str) -> ResultComparison:
    """
    pre: errors in _ERROR_HANDLERS
    post: _
    """
    return compare_results(lambda b, e: b.decode("ascii", e), bytestring, errors)


def check_decode_latin1(bytestring: bytes, errors: str) -> ResultComparison:
    """
    pre: errors in _ERROR_HANDLERS
    post: _
    """
    return compare_results(lambda b, e: b.decode("latin1", e), bytestring, errors)


def check_decode_utf8(bytestring: bytes, errors: str) -> ResultComparison:
    """
    pre: errors in _ERROR_HANDLERS
    post: _
    """
    # crosshair: max_iterations=200
    return compare_results(lambda b, e: b.decode("utf8", e), bytestring, errors)


# TODO add handling for BytesIO(SymbolicBytes())
# def check_stream_decode_utf8(bytestring: bytes, errors: str) -> ResultComparison:
#     """
#     pre: errors in _ERROR_HANDLERS
#     post: _
#     """
#     def read_stream(b, e):
#         return codecs.getreader("utf8")(BytesIO(b), e).read()
#     return compare_results(read_stream, bytestring, errors)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    this_module = sys.modules[__name__]
    messages = run_checkables(analyze_function(getattr(this_module, fn_name)))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
