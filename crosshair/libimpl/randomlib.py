import random
from functools import partial
from inspect import Parameter, Signature
from typing import Any, Callable, List, Optional, Union

from crosshair import NoTracing, SymbolicFactory, register_type
from crosshair.libimpl.builtinslib import SymbolicInt
from crosshair.register_contract import register_contract


class ExplicitRandom(random.Random):
    def __new__(cls, *a):
        return super().__new__(ExplicitRandom, 0)

    def __init__(
        self,
        future_values: Optional[List[Union[int, float]]] = None,
        idx: int = 0,
        intgen: Callable[[int], int] = lambda c: 0,
        floatgen: Callable[[], float] = lambda: 0.0,
    ):
        self._future_values = future_values if future_values else []
        self._idx = idx
        self._intgen = intgen
        self._floatgen = floatgen
        super().__init__()

    def __copy__(self):
        # Just a regular copy. (otherwise, we'd be deferring to getstate/setstate)
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __ch_deepcopy__(self, memo):
        # We pretend this is a deepcopy, but it isn't.
        # That way, the values lazily added to _future_values will be the same in each
        # instance, even though we don't know what they are at the time of the deep
        # copy.
        return self.__copy__()

    def __repr__(self) -> str:
        return f"crosshair.libimpl.randomlib.ExplicitRandom({self._future_values!r})"

    def __reduce__(self):
        return (ExplicitRandom, (self._future_values, self._idx))

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
        return self._randbelow(2**k)


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
        lambda p: ExplicitRandom([], 0, partial(genint, p), partial(genfloat, p)),
    )

    register_contract(
        random.Random.random,
        post=lambda __return__: 0.0 <= __return__ < 1.0,
        sig=Signature(
            parameters=[
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
            ],
            return_annotation=float,
        ),
    )
    register_contract(
        random.Random.randint,
        pre=lambda a, b: a <= b,
        post=lambda __return__, a, b: a <= __return__ <= b,
        sig=Signature(
            parameters=[
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                Parameter("b", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            ],
            return_annotation=int,
        ),
    )
    register_contract(
        random.Random.randrange,
        pre=lambda start, stop, step: (
            (step == 1 and start >= 1)
            if stop is None
            else (start != stop and step != 0 and (stop - start > 0) == (step > 0))
        ),
        post=lambda __return__, start, stop, step, _int=int: (
            (0 <= __return__ < start if stop is None else start <= __return__ < stop)
            and (step == 1 or (__return__ - start) % step == 0)
        ),
        sig=Signature(
            parameters=[
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("start", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                Parameter(
                    "stop",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=Optional[int],
                ),
                Parameter(
                    "step", Parameter.POSITIONAL_OR_KEYWORD, default=1, annotation=int
                ),
                Parameter(
                    "_int", Parameter.POSITIONAL_OR_KEYWORD, default=int, annotation=int
                ),
            ],
            return_annotation=int,
        ),
    )
    register_contract(
        random.Random.uniform,
        post=lambda __return__, a, b: min(a, b) <= __return__ <= max(a, b),
        sig=Signature(
            parameters=[
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, annotation=float),
                Parameter("b", Parameter.POSITIONAL_OR_KEYWORD, annotation=float),
            ],
            return_annotation=float,
        ),
    )
    register_contract(
        random.Random.getrandbits,
        pre=lambda k: k >= 0,
        post=lambda __return__, k: 0 <= __return__ < 2**k,
        sig=Signature(
            parameters=[
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("k", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            ],
            return_annotation=int,
        ),
    )
