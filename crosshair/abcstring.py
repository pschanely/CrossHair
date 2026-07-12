import collections.abc
import sys
from collections import UserString
from numbers import Integral
from typing import Mapping, Union

from crosshair.tracers import NoTracing

# Similar to UserString, but allows you to lazily supply the contents
# when accessed.

# Sadly, this illusion doesn't fully work: various Python operations
# require a actual strings or subclasses.
# (see related issue: https://bugs.python.org/issue16397)

# TODO: Our symbolic strings likely already override most of these methods.
# Consider removing this class.

_MISSING = object()


def _real_string(thing: object):
    with NoTracing():
        return thing.data if isinstance(thing, (UserString, AbcString)) else thing


def _real_affix(thing: object):
    # startswith/endswith accept a single prefix/suffix OR a tuple of them
    if isinstance(thing, tuple):
        return tuple(_real_string(t) for t in thing)
    return _real_string(thing)


def _real_int(thing: object):
    return thing.__int__() if isinstance(thing, Integral) else thing


def _unfindable_range(start, end, mylen: int) -> bool:
    """Emulate the preliminary bounds checks CPython makes before searching for a
    substring (in str.find, str.startswith, etc)."""
    if start is None or start == 0 or start <= -mylen:
        return False
    # `start` is defined and points past 0:
    if end is None or end >= mylen:
        return start > mylen
    # `end` is defined and points before the end of the string:
    if start < 0:
        start += mylen
    if end < 0:
        end += mylen
    return start > end


class AbcString(collections.abc.Sequence, collections.abc.Hashable):
    """
    Implement just ``__str__``.

    Useful for making lazy strings.
    """

    data = property(lambda s: s.__str__())

    def __repr__(self):
        return repr(self.data)

    def __hash__(self):
        return hash(self.data)

    def __eq__(self, string):
        return self.data == _real_string(string)

    def __lt__(self, string):
        return self.data < _real_string(string)

    def __le__(self, string):
        return self.data <= _real_string(string)

    def __gt__(self, string):
        return self.data > _real_string(string)

    def __ge__(self, string):
        return self.data >= _real_string(string)

    def __contains__(self, char):
        return _real_string(char) in self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __add__(self, other):
        other = _real_string(other)
        if isinstance(other, str):
            return self.data + other
        return self.data + str(other)

    def __radd__(self, other):
        other = _real_string(other)
        if isinstance(other, str):
            return other + self.data
        return str(other) + self.data

    def __mul__(self, n):
        return self.data * n

    def __rmul__(self, n):
        return self.data * n

    def __mod__(self, args):
        return self.data % args

    def __rmod__(self, template):
        return str(template) % self.data

    # the following methods are defined in alphabetical order:
    def capitalize(self):
        return self.data.capitalize()

    def casefold(self):
        return self.data.casefold()

    def center(self, width, *args):
        return self.data.center(width, *args)

    def count(self, sub, start=None, end=None):
        subpoints = self._ch_search_operand_points(sub)
        sliced = self[start:end]
        if len(subpoints) == 0:
            return len(sliced) + 1
        total, pos, step = 0, 0, len(subpoints)
        while True:
            idx = sliced.find(sub, pos)
            if idx == -1:
                return total
            total += 1
            pos = idx + step

    def encode(self, encoding=_MISSING, errors=_MISSING):
        if encoding is not _MISSING:
            if errors is not _MISSING:
                return self.data.encode(encoding, errors)
            return self.data.encode(encoding)
        return self.data.encode()

    def endswith(self, suffix, start=None, end=None):
        if isinstance(suffix, tuple):
            return any(self.endswith(s, start, end) for s in suffix)
        suffixpoints = self._ch_operand_points(suffix)
        slen = len(suffixpoints)
        matchable = self if start is None and end is None else self[start:end]
        if slen == 0:
            return not _unfindable_range(start, end, len(self))
        if slen > len(matchable):
            return False
        tail = matchable._ch_codepoints[-slen:]
        return all(a == b for a, b in zip(tail, suffixpoints))

    def expandtabs(self, tabsize=8):
        return self.data.expandtabs(_real_int(tabsize))

    # ---- shared symbolic string/bytes algorithms -------------------------
    # These work on the underlying codepoint/byte sequence, so a symbolic
    # receiver stays symbolic (no realization).  str and bytes differ only in
    # element domain and result type; subclasses supply the three hooks below.
    @property
    def _ch_codepoints(self):
        raise NotImplementedError

    def _ch_make(self, codepoints):
        """Build an instance of the receiver's own type from a codepoint seq."""
        raise NotImplementedError

    def _ch_operand_points(self, operand):
        """Validate a needle/separator operand; return its codepoint seq."""
        raise NotImplementedError

    def _ch_search_operand_points(self, operand):
        """Like ``_ch_operand_points`` but for the find/index/count family, which
        accepts a superset of operands (bytes/bytearray also take a single int
        byte value).  Defaults to the strict form."""
        return self._ch_operand_points(operand)

    def _find(self, sub, start=None, end=None, from_right=False):
        # Search the codepoint sequence directly (not via partition, which is
        # strict about operand type): find/index accept an int byte value that
        # partition rejects.
        subpoints = self._ch_search_operand_points(sub)
        mylen = len(self)
        if start is None:
            start = 0
        elif start < 0:
            start += mylen
        if end is None:
            end = mylen
        elif end < 0:
            end += mylen
        matchable = self[start:end] if start != 0 or end != mylen else self
        if len(subpoints) == 0:
            # CPython oddity: the empty string is findable when over-slicing off
            # the left side but not the right ('' .find('', 3, 4) == -1).
            if len(matchable) == 0 and start > min(mylen, max(end, 0)):
                return -1
            return max(min(end, mylen), 0) if from_right else max(start, 0)
        points = matchable._ch_codepoints
        sublen = len(subpoints)
        span = len(matchable) - sublen
        positions = range(span, -1, -1) if from_right else range(0, span + 1)
        for i in positions:
            if all(a == b for a, b in zip(points[i : i + sublen], subpoints)):
                return start + i
        return -1

    def find(self, sub, start=None, end=None):
        return self._find(sub, start, end, from_right=False)

    def format(self, *args, **kwds):
        return self.data.format(*args, **kwds)

    def format_map(self, mapping):
        return self.data.format_map(mapping)

    def index(self, sub, start=None, end=None):
        idx = self.find(sub, start, end)
        if idx == -1:
            raise ValueError
        return idx

    def isalpha(self):
        return self.data.isalpha()

    def isalnum(self):
        return self.data.isalnum()

    def isascii(self):
        return self.data.isascii()

    def isdecimal(self):
        return self.data.isdecimal()

    def isdigit(self):
        return self.data.isdigit()

    def isidentifier(self):
        return self.data.isidentifier()

    def islower(self):
        return self.data.islower()

    def isnumeric(self):
        return self.data.isnumeric()

    def isprintable(self):
        return self.data.isprintable()

    def isspace(self):
        return self.data.isspace()

    def istitle(self):
        return self.data.istitle()

    def isupper(self):
        return self.data.isupper()

    def join(self, seq):
        return self.data.join(seq)

    def ljust(self, width, *args):
        return self.data.ljust(width, *args)

    def lower(self):
        return self.data.lower()

    def lstrip(self, chars=None):
        return self.data.lstrip(_real_string(chars))

    maketrans = str.maketrans

    def partition(self, sep):
        seppoints = self._ch_operand_points(sep)
        if len(seppoints) == 0:
            raise ValueError
        mypoints = self._ch_codepoints
        seplen = len(seppoints)
        for start in range(1 + len(mypoints) - seplen):
            # all()/zip defers the character comparisons into one SMT query.
            if all(a == b for a, b in zip(mypoints[start : start + seplen], seppoints)):
                return (
                    self._ch_make(mypoints[:start]),
                    self._ch_make(seppoints),
                    self._ch_make(mypoints[start + seplen :]),
                )
        return (self, self._ch_make([]), self._ch_make([]))

    def replace(self, old, new, count=-1):
        oldpoints = self._ch_operand_points(old)
        self._ch_operand_points(new)  # validate the replacement's type
        if count == 0:
            return self
        if len(self) == 0:
            if len(oldpoints) == 0:
                return self._ch_make(self._ch_operand_points(new))
            return self
        if len(oldpoints) == 0:
            return new + self[:1] + self[1:].replace(old, new, count - 1)
        prefix, match, suffix = self.partition(old)
        if len(match) == 0:
            return self
        return prefix + new + suffix.replace(old, new, count - 1)

    def rfind(self, sub, start=None, end=None):
        return self._find(sub, start, end, from_right=True)

    def rindex(self, sub, start=None, end=None):
        idx = self.rfind(sub, start, end)
        if idx == -1:
            raise ValueError
        return idx

    def rjust(self, width, *args):
        return self.data.rjust(width, *args)

    def rpartition(self, sep):
        seppoints = self._ch_operand_points(sep)
        if len(seppoints) == 0:
            raise ValueError
        mypoints = self._ch_codepoints
        seplen = len(seppoints)
        for start in range(len(mypoints) - seplen, -1, -1):
            if all(a == b for a, b in zip(mypoints[start : start + seplen], seppoints)):
                return (
                    self._ch_make(mypoints[:start]),
                    self._ch_make(seppoints),
                    self._ch_make(mypoints[start + seplen :]),
                )
        return (self._ch_make([]), self._ch_make([]), self)

    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)

    def rstrip(self, chars=None):
        return self.data.rstrip(_real_string(chars))

    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)

    def splitlines(self, keepends=False):
        return self.data.splitlines(keepends)

    def startswith(self, prefix, start=None, end=None):
        if isinstance(prefix, tuple):
            return any(self.startswith(p, start, end) for p in prefix)
        prefixpoints = self._ch_operand_points(prefix)
        if start is None and end is None:
            matchable = self
        else:
            # The empty string is findable off the left side but not the right.
            if _unfindable_range(start, end, len(self)):
                return False
            matchable = self[start:end]
        plen = len(prefixpoints)
        if plen > len(matchable):
            return False
        head = matchable._ch_codepoints[:plen]
        return all(a == b for a, b in zip(head, prefixpoints))

    def strip(self, chars=None):
        return self.data.strip(_real_string(chars))

    def swapcase(self):
        return self.data.swapcase()

    def title(self):
        return self.data.title()

    def translate(self, table: Mapping[int, Union[int, str, None]]) -> str:
        return self.data.translate(table)

    def upper(self):
        return self.data.upper()

    def zfill(self, width):
        return self.data.zfill(width)

    if sys.version_info >= (3, 9):

        def removeprefix(self, prefix: str) -> "AbcString":
            if self.startswith(prefix):
                return self[len(prefix) :]
            return self

        def removesuffix(self, suffix: str) -> "AbcString":
            if self.endswith(suffix):
                suffixlen = len(suffix)
                if suffixlen == 0:
                    return self
                return self[:-suffixlen]
            return self
