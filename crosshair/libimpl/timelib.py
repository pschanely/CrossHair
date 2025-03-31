import time as real_time
from inspect import Signature
from math import isfinite
from typing import Any, Literal

from crosshair.core import register_patch
from crosshair.register_contract import register_contract
from crosshair.statespace import context_statespace
from crosshair.tracers import NoTracing


class EarliestPossibleTime:
    monotonic: float = 0.0
    process_time: float = 0.0

    def __init__(self, *a):
        pass


# Imprecision at high values becomes a sort of artificial problem
_UNREALISTICALLY_LARGE_TIME_FLOAT = float(60 * 60 * 24 * 365 * 100_000)


def _gte_last(kind: Literal["monotonic", "process_time"], value: Any) -> bool:
    with NoTracing():
        earliest_times = context_statespace().extra(EarliestPossibleTime)
        threshold = getattr(earliest_times, kind)
        setattr(earliest_times, kind, value)
    return all([threshold <= value, value < _UNREALISTICALLY_LARGE_TIME_FLOAT])


def _sleep(value: float) -> None:
    with NoTracing():
        earliest_times = context_statespace().extra(EarliestPossibleTime)
    earliest_times.monotonic += value
    return None


def make_registrations():
    register_contract(
        real_time.time,
        post=lambda __return__: __return__ > 0.0 and isfinite(__return__),
        sig=Signature(parameters=[], return_annotation=float),
    )
    register_contract(
        real_time.time_ns,
        post=lambda __return__: __return__ > 0,
        sig=Signature(parameters=[], return_annotation=int),
    )
    register_contract(
        real_time.monotonic,
        post=lambda __return__: _gte_last("monotonic", __return__)
        and isfinite(__return__),
        sig=Signature(parameters=[], return_annotation=float),
    )
    register_contract(
        real_time.monotonic_ns,
        post=lambda __return__: _gte_last("monotonic", __return__ / 1_000_000_000),
        sig=Signature(parameters=[], return_annotation=int),
    )
    register_contract(
        real_time.process_time,
        post=lambda __return__: _gte_last("process_time", __return__)
        and isfinite(__return__),
        sig=Signature(parameters=[], return_annotation=float),
    )
    register_contract(
        real_time.process_time_ns,
        post=lambda __return__: _gte_last("process_time", __return__ / 1_000_000_000),
        sig=Signature(parameters=[], return_annotation=int),
    )
    register_patch(real_time.sleep, _sleep)
