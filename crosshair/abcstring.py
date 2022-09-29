import collections.abc
import sys
from collections import UserString
from numbers import Integral

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


def _real_int(thing: object):
    return thing.__int__() if isinstance(thing, Integral) else thing


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

    def count(self, sub, start=0, end=sys.maxsize):
        return self.data.count(_real_string(sub), start, end)

    def encode(self, encoding=_MISSING, errors=_MISSING):
        if encoding is not _MISSING:
            if errors is not _MISSING:
                return self.data.encode(encoding, errors)
            return self.data.encode(encoding)
        return self.data.encode()

    def endswith(self, suffix, start=0, end=sys.maxsize):
        return self.data.endswith(suffix, start, end)

    def expandtabs(self, tabsize=8):
        return self.data.expandtabs(_real_int(tabsize))

    def find(self, sub, start=0, end=sys.maxsize):
        return self.data.find(_real_string(sub), start, end)

    def format(self, *args, **kwds):
        return self.data.format(*args, **kwds)

    def format_map(self, mapping):
        return self.data.format_map(mapping)

    def index(self, sub, start=0, end=sys.maxsize):
        return self.data.index(_real_string(sub), start, end)

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
        return self.data.partition(_real_string(sep))

    def replace(self, old, new, maxsplit=-1):
        return self.data.replace(_real_string(old), _real_string(new), maxsplit)

    def rfind(self, sub, start=0, end=sys.maxsize):
        return self.data.rfind(_real_string(sub), start, end)

    def rindex(self, sub, start=0, end=sys.maxsize):
        return self.data.rindex(_real_string(sub), start, end)

    def rjust(self, width, *args):
        return self.data.rjust(width, *args)

    def rpartition(self, sep):
        return self.data.rpartition(sep)

    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)

    def rstrip(self, chars=None):
        return self.data.rstrip(_real_string(chars))

    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)

    def splitlines(self, keepends=False):
        return self.data.splitlines(keepends)

    def startswith(self, prefix, start=0, end=sys.maxsize):
        return self.data.startswith(prefix, start, end)

    def strip(self, chars=None):
        return self.data.strip(_real_string(chars))

    def swapcase(self):
        return self.data.swapcase()

    def title(self):
        return self.data.title()

    def translate(self, *args):
        return self.data.translate(*args)

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
