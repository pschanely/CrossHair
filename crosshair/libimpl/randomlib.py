from functools import partial
import random
from typing import Any, Callable, List, Optional, Union

from crosshair.libimpl.builtinslib import SymbolicInt
from crosshair import debug
from crosshair import register_type
from crosshair import NoTracing
from crosshair import SymbolicFactory


class ExplicitRandom(random.Random):
    def __new__(cls, *a):
        return super().__new__(ExplicitRandom, 0)

    def __init__(
        self,
        future_values: Optional[List[Union[int, float]]] = None,
        intgen: Callable[[int], int] = lambda c: 0,
        floatgen: Callable[[], float] = lambda: 0.0,
    ):
        self._future_values = future_values if future_values else []
        self._idx = 0
        self._intgen = intgen
        self._floatgen = floatgen
        super().__init__()

    def __copy__(self):
        # Just a regular copy. (otherwise, we'd be deferring to getstate/setstate)
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        # We pretend this is a deepcopy, but it isn't.
        # That way, the values lazily added to _future_values will be the same in each
        # instance, even though we don't know what they are at the time of the deep
        # copy.
        return self.__copy__()

    def __repr__(self) -> str:
        return f"crosshair.libimpl.randomlib.ExplicitRandom({self._future_values!r})"

    def getstate(self):
        raise NotImplementedError

    def setstate(self, o):
        raise NotImplementedError

    def random(self) -> float:
        idx = self._idx
        future_values = self._future_values
        if idx >= len(future_values):
            future_values.append(self._floatgen())
        intorfloat = self._future_values[idx]
        if isinstance(intorfloat, int):
            ret = 1.0 / abs(intorfloat) if intorfloat != 0 else 0.0
        else:
            ret = intorfloat
        self._idx += 1
        return ret

    def _randbelow(self, cap: int) -> int:
        idx = self._idx
        future_values = self._future_values
        if idx >= len(future_values):
            future_values.append(self._intgen(cap))
        intorfloat = future_values[idx]
        if isinstance(intorfloat, float):
            ret = abs(hash(intorfloat)) % cap
        else:
            ret = intorfloat
        self._idx += 1
        return ret

    def getrandbits(self, k: int) -> int:
        return self._randbelow(2 ** k)


def genint(factory: SymbolicFactory, cap: int):
    with NoTracing():
        symbolic_int = SymbolicInt(factory.varname + factory.space.uniq(), int)
        factory.space.add(0 <= symbolic_int.var)
        factory.space.add(symbolic_int.var < SymbolicInt._coerce_to_smt_sort(cap))
        return symbolic_int


def genfloat(factory: SymbolicFactory):
    with NoTracing():
        symbolic_float: Any = factory(float)
        factory.space.add(0.0 <= symbolic_float.var)
        factory.space.add(symbolic_float.var < 1.0)
        return symbolic_float


def make_registrations() -> None:
    register_type(
        random.Random,
        lambda p: ExplicitRandom([], partial(genint, p), partial(genfloat, p)),
    )
