from array import array
from typing import BinaryIO, ByteString, Dict, Iterable, List, Sequence, Tuple

import z3  # type: ignore

from crosshair import SymbolicFactory, realize, register_patch
from crosshair.core import CrossHairValue, register_type
from crosshair.libimpl.builtinslib import SymbolicArrayBasedUniformTuple
from crosshair.simplestructs import ShellMutableSequence
from crosshair.statespace import StateSpace
from crosshair.tracers import NoTracing

INT_TYPE_BOUNDS: Dict[str, Tuple[int, int]] = {
    # (min, max) ranges - inclusive on min, exclusive on max.
    # Order is significant - we choose earlier codes more readily.
    "L": (0, 1 << 32),
    "B": (0, 1 << 8),
    "l": (-(1 << 31), (1 << 31)),
    "b": (-(1 << 7), (1 << 7)),
    "Q": (0, 1 << 64),
    "q": (-(1 << 63), (1 << 63)),
    "I": (0, 1 << 16),
    "i": (-(1 << 15), (1 << 15)),
    "H": (0, 1 << 16),
    "h": (-(1 << 15), (1 << 15)),
}

INT_TYPE_SIZE = {c: array(c).itemsize for c in INT_TYPE_BOUNDS.keys()}


def is_bytes_like(obj: object) -> bool:
    return isinstance(obj, (ByteString, array))


def pick_code(space: StateSpace) -> Tuple[str, int, int]:
    last_idx = len(INT_TYPE_BOUNDS) - 1
    for (idx, (code, rng)) in enumerate(INT_TYPE_BOUNDS.items()):
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
        super().extend(self._iter_checker(nums))

    def from_bytes(self, b: Sequence) -> None:
        self.extend(b)

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
