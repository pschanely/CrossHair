import sys
from io import StringIO
from typing import List, Optional, Union

import pytest  # type: ignore

from crosshair.core_and_libs import MessageType, analyze_function, run_checkables
from crosshair.libimpl.iolib import BackedStringIO
from crosshair.options import AnalysisOptionSet
from crosshair.test_util import compare_returns
from crosshair.tracers import is_tracing

# crosshair: max_iterations=40
# crosshair: per_condition_timeout=100


def _maybe_stringio(s: BackedStringIO) -> Union[StringIO, BackedStringIO]:
    if not is_tracing():
        sio = StringIO(s._contents, s._newline_mode)
        sio.seek(s._pos)
        return sio
    return s


def _do_something(s: Union[StringIO, BackedStringIO], opname: str) -> object:
    if opname == "closed":
        return s.closed
    elif opname == "flush":
        s.flush()
    elif opname == "read":
        return s.read()
    elif opname == "readlines":
        return s.readlines()
    elif opname == "readable":
        return s.readable()
    elif opname == "seek":
        return s.seek(1)
    elif opname == "seekable":
        return s.seekable()
    elif opname == "tell":
        return s.tell()
    elif opname == "truncate":
        return s.truncate()
    elif opname == "writable":
        return s.writable()
    elif opname == "write":
        return s.write("")
    return None


def check_stringio_readlines(s: BackedStringIO, hint: int):
    """post: _"""

    def readlines(s, hint: int):
        return _maybe_stringio(s).readlines(hint)

    return compare_returns(readlines, s, hint)


def check_stringio_writelines(s: BackedStringIO, lines: List[str]):
    """post: _"""

    def writelines(s, lines: List[str]):
        return _maybe_stringio(s).writelines(lines)

    return compare_returns(writelines, s, lines)


def check_stringio_seek1(s: BackedStringIO, o1: int, w1: int):
    """post: _"""

    def seek_double(s, o1: int, w1: int) -> int:
        s = _maybe_stringio(s)
        s.seek(o1, w1)
        return s.tell()

    return compare_returns(seek_double, s, o1, w1)


def check_stringio_seek_seek(s: BackedStringIO, o1: int, w1: int, o2: int, w2: int):
    """post: _"""

    def seek_seek(s, o1: int, w1: int, o2: int, w2: int) -> int:
        s = _maybe_stringio(s)
        s.seek(o1, w1)
        s.seek(o2, w2)
        return s.tell()

    return compare_returns(seek_seek, s, o1, w1, o2, w2)


def check_stringio_seek_write(s: BackedStringIO, o1: int, w1: int, ws: str):
    """post: _"""

    def seek_write(s, o1: int, w1: int, ws: str) -> int:
        s = _maybe_stringio(s)
        s.seek(o1, w1)
        return s.write(ws)

    return compare_returns(seek_write, s, o1, w1, ws)


def check_stringio_write_newlines(newline_mode: str, ws: str):
    """post: _"""

    def write_newlines(newline_mode: Optional[str], ws: str):
        if newline_mode in (None, "", "\n", "\r", "\r\n"):
            s = _maybe_stringio(BackedStringIO("", newline_mode))
            s.write(ws)
            return s.newlines

    return compare_returns(write_newlines, newline_mode, ws)


def check_stringio_simple_operation(s: BackedStringIO, opname: str):
    """post: _"""

    def simple_operation(s, opname: str) -> object:
        return _do_something(_maybe_stringio(s), opname)

    return compare_returns(simple_operation, s, opname)


def check_stringio_operation_while_closed(s: BackedStringIO, opname: str):
    """post: _"""

    def closed_operation(s, opname: str) -> object:
        s = _maybe_stringio(s)
        s.close()
        return _do_something(s, opname)

    return compare_returns(closed_operation, s, opname)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    fn = getattr(sys.modules[__name__], fn_name)
    messages = run_checkables(analyze_function(fn))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
