import re
from io import SEEK_CUR, SEEK_END, SEEK_SET, StringIO, TextIOBase
from typing import Optional, Tuple, Union

from crosshair import ResumedTracing, SymbolicFactory, register_type
from crosshair.util import IgnoreAttempt

_UNIVERSAL_NEWLINE_RE = re.compile(r"(\r\n|\r|\n)")


class BackedStringIO(TextIOBase):
    _contents: str
    _pos: int
    _discovered_newlines: set
    _newline_mode: Optional[str]

    def __init__(
        self, initial_value: str, newline_mode: Optional[str] = "\n", pos: int = 0
    ):
        r"""
        pre: newline_mode in (None, "", "\n", "\r", "\r\n")
        pre: pos >= 0
        """
        if not (isinstance(initial_value, str)):
            raise TypeError
        if not (isinstance(newline_mode, (str, type(None)))):
            raise TypeError
        if newline_mode not in (None, "", "\n", "\r", "\r\n"):
            raise ValueError
        self._newline_mode = newline_mode
        self._discovered_newlines = set()
        self._pos = pos
        self._contents = self._replace_newlines(initial_value) if initial_value else ""

    def __repr__(self):
        contents, newline_mode, pos = self._contents, self._newline_mode, self._pos
        if pos == 0:
            if newline_mode == "\n":
                return f"BackedStringIO({contents!r})"
            else:
                return f"BackedStringIO({contents!r}, {newline_mode!r})"
        return (
            f"BackedStringIO({contents!r}, newline_mode={newline_mode!r}, pos={pos!r})"
        )

    @property
    def newlines(self) -> Union[None, str, Tuple[str, ...]]:  # type: ignore
        discovered = self._discovered_newlines
        # Fiddly! Trying to preserve the static tuple ordering that CPython has:
        ret = tuple(nl for nl in ("\r", "\n", "\r\n") if nl in discovered)
        if len(ret) > 1:
            return ret
        if len(ret) == 1:
            return ret[0]
        return None

    def _replace_newlines(self, string: str) -> str:
        newline_mode = self._newline_mode
        if newline_mode is None:

            def replace(match: re.Match) -> str:
                self._discovered_newlines.add(match.group())
                return "\n"

            return _UNIVERSAL_NEWLINE_RE.sub(replace, string)
        elif newline_mode == "":
            self._discovered_newlines.update(_UNIVERSAL_NEWLINE_RE.findall(string))
            return string
        else:
            return string.replace("\n", newline_mode)

    def flush(self) -> None:
        return

    def getvalue(self) -> str:
        return self._contents

    def read(self, amount: Optional[int] = None) -> str:
        if self.closed:
            raise ValueError
        if amount is None:
            ret = self._contents[self._pos :]
        else:
            ret = self._contents[self._pos : self._pos + amount]
        self._pos += len(ret)
        return ret

    def readable(self) -> bool:
        if self.closed:
            raise ValueError
        return True

    def readline(self, limit: Optional[int] = None) -> str:  # type: ignore
        if self.closed:
            raise ValueError
        contents, pos = self._contents, self._pos
        if limit is None:
            limit = len(contents)
        if self._newline_mode == "":
            # All other modes would have normalized the contents to \n already.
            for match in _UNIVERSAL_NEWLINE_RE.finditer(contents, pos, limit):
                self._pos = match.end()
                return contents[pos : match.end()]
            self._pos = limit
            return contents[pos:limit]
        else:
            idx = contents.find("\n", pos, limit)
            if idx == -1:
                self._pos = limit
                return contents[pos:limit]
            else:
                self._pos = idx + 1
                return contents[pos : idx + 1]

    def write(self, string: str) -> int:
        if self.closed:
            raise ValueError
        contents, pos = self._contents, self._pos
        contentslen = len(contents)
        if not string:
            return 0
        writestr = self._replace_newlines(string)
        writelen = len(writestr)
        if pos > contentslen:
            self._contents += "\u0000" * (pos - contentslen)
            pos = contentslen
        if pos == contentslen:
            self._contents += writestr
        else:
            self._contents = contents[:pos] + writestr + contents[pos + writelen :]
        self.pos = pos + writelen
        return writelen

    def seek(self, amount: int, whence: int = SEEK_SET) -> int:
        if self.closed:
            raise ValueError
        if whence == SEEK_CUR:
            if amount != 0:
                raise OSError
            pos = self._pos + amount
        elif whence == SEEK_END:
            if amount != 0:
                raise OSError
            pos = len(self._contents) + amount
        elif whence == SEEK_SET:
            if amount < 0:
                raise ValueError
            pos = amount
        else:
            raise ValueError
        self._pos = pos
        return pos

    def seekable(self) -> bool:
        if self.closed:
            raise ValueError
        return True

    def tell(self) -> int:
        if self.closed:
            raise ValueError
        return self._pos

    def truncate(self, size: Optional[int] = None) -> int:
        if self.closed:
            raise ValueError
        if size is None:
            size = self._pos
        self._contents = self._contents[:size]
        return size

    def writable(self) -> bool:
        if self.closed:
            raise ValueError
        return True


def make_string_io(factory: SymbolicFactory) -> BackedStringIO:
    contents = factory(str)
    newline_mode = factory(Optional[str])
    with ResumedTracing():
        if newline_mode not in (None, "", "\n", "\r", "\r\n"):
            raise IgnoreAttempt
        return BackedStringIO(contents, newline_mode)


def make_registrations() -> None:
    register_type(StringIO, make_string_io)
    # TODO: register_type(io.TextIO, )
