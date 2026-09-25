import numbers
import sys
from array import array
from typing import BinaryIO, Dict, Iterable, List, Sequence, Tuple

import z3  # type: ignore

from crosshair import SymbolicFactory, realize, register_patch
from crosshair.core import register_type
from crosshair.libimpl.builtinslib import SymbolicArrayBasedUniformTuple
from crosshair.simplestructs import ShellMutableSequence, check_idx
from crosshair.statespace import StateSpace
from crosshair.tracers import NoTracing
from crosshair.util import CrossHairValue, name_of_type


def _int_bounds(typecode: str) -> Tuple[int, int]:
    """Return the (inclusive min, exclusive max) values for an integer typecode."""
    bits = array(typecode).itemsize * 8
    if typecode.isupper():
        return (0, 1 << bits)
    return (-(1 << (bits - 1)), 1 << (bits - 1))


# Order is significant - we choose earlier codes more readily.
INT_TYPE_BOUNDS: Dict[str, Tuple[int, int]] = {
    code: _int_bounds(code) for code in "LBlbQqIiHh"
}

INT_TYPE_SIZE = {c: array(c).itemsize for c in INT_TYPE_BOUNDS.keys()}


def pick_code(space: StateSpace) -> Tuple[str, int, int]:
    last_idx = len(INT_TYPE_BOUNDS) - 1
    for idx, (code, rng) in enumerate(INT_TYPE_BOUNDS.items()):
        if idx < last_idx:
            if space.smt_fork(desc=f"not_{code}_array"):
                continue
        return (code, *rng)
    assert False, "Not Reachable"


def make_array(creator: SymbolicFactory) -> object:
    space = creator.space
    code, minval, maxval = pick_code(space)
    nums = SymbolicArrayBasedUniformTuple(creator.varname, Tuple[int, ...])
    z3_array = nums._arr()
    qvar = z3.Int("arrvar" + space.uniq())
    space.add(z3.ForAll([qvar], minval <= z3.Select(z3_array, qvar)))
    space.add(z3.ForAll([qvar], z3.Select(z3_array, qvar) < maxval))
    return SymbolicArray(code, nums)


def check_int(item, minval, maxval):
    if not (minval <= item < maxval):
        raise OverflowError
    return item


def _array(typecode: str, iterable: Iterable = ()):
    realized_type = realize(typecode)
    bounds = INT_TYPE_BOUNDS.get(typecode)
    if bounds:
        args = [check_int(x, *bounds) for x in iterable]
        return SymbolicArray(realized_type, args)
    return array(realized_type, realize(iterable))


class SymbolicArray(
    ShellMutableSequence,
    CrossHairValue,
):
    def __init__(self, typecode: str, items: Sequence = ()):
        # All arguments are presumed valid here
        self.typecode = typecode
        self.itemsize = INT_TYPE_SIZE[typecode]
        self._snapshots: List[array] = []
        super().__init__(items)

    def _realized_inner(self) -> array:
        with NoTracing():
            realized = self.__ch_realize__()
            self.inner = realized
            return realized

    def _iter_checker(self, items: Iterable[int]) -> Iterable[int]:
        bounds = INT_TYPE_BOUNDS.get(self.typecode)
        if bounds:
            return (check_int(i, *bounds) for i in items)
        else:
            return items

    __hash__ = None  # type: ignore

    def __ch_realize__(self):
        return array(self.typecode, self.inner)

    def __ch_pytype__(self):
        return array

    def __add__(self, other):
        if not isinstance(other, array):
            raise TypeError(
                f'can only append array (not "{name_of_type(type(other))}") to array'
            )
        if self.typecode != other.typecode:
            raise TypeError("bad argument type for built-in operation")
        return super().__add__(other)

    def __radd__(self, other):
        return NotImplemented

    def __iadd__(self, other):
        if not isinstance(other, array):
            raise TypeError(
                f'can only extend array with array (not "{name_of_type(type(other))}")'
            )
        self.extend(other)
        return self

    def __eq__(self, other):
        if not isinstance(other, array):
            return False
        return ShellMutableSequence.__eq__(self, other)

    def __setitem__(self, k, v):
        bounds = INT_TYPE_BOUNDS.get(self.typecode)
        if bounds is not None:
            if isinstance(k, slice):
                v = self._iter_checker(v)
            elif isinstance(k, numbers.Integral):
                k = check_idx(k, len(self))
                check_int(v, *bounds)
        return super().__setitem__(k, v)

    def _spawn(self, items: Sequence) -> ShellMutableSequence:
        return SymbolicArray(self.typecode, items)

    def append(self, value) -> None:
        bounds = INT_TYPE_BOUNDS.get(self.typecode)
        if bounds:
            check_int(value, *bounds)
        return super().append(value)

    def buffer_info(self) -> Tuple[int, int]:
        return self._realized_inner().buffer_info()

    def byteswap(self) -> None:
        self._realized_inner().byteswap()

    # count() handled by superclass

    def extend(self, nums: Iterable) -> None:
        if isinstance(nums, array) and nums.typecode != self.typecode:
            raise TypeError("can only extend with array of same kind")
        super().extend(self._iter_checker(nums))

    def frombytes(self, b: bytes) -> None:
        if not isinstance(b, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"a bytes-like object is required, not '{type(b).__name__}'"
            )
        itemsize = self.itemsize
        if len(b) % itemsize != 0:
            raise ValueError("bytes length not a multiple of item size")
        signed = INT_TYPE_BOUNDS[self.typecode][0] < 0
        self.extend(
            int.from_bytes(b[i : i + itemsize], sys.byteorder, signed=signed)
            for i in range(0, len(b), itemsize)
        )

    def fromfile(self, fd: BinaryIO, num_bytes: int) -> None:
        self._realized_inner().fromfile(fd, num_bytes)

    def fromlist(self, nums: List) -> None:
        self.extend(nums)

    def fromunicode(self, s: str) -> None:
        self._realized_inner().fromunicode(s)

    # index() handled by superclass
    # insert() handled by superclass
    # pop() handled by superclass
    # remove() handled by superclass
    # reverse() handled by superclass

    def tobytes(self) -> bytes:
        return self._realized_inner().tobytes()

    def tofile(self, fh: BinaryIO) -> None:
        self._realized_inner().tofile(fh)

    def tolist(self) -> List:
        return list(self.inner)

    def tounicode(self) -> str:
        return self._realized_inner().tounicode()

    # TODO: test repr


def make_registrations():
    register_type(array, make_array)
    register_patch(array, _array)
