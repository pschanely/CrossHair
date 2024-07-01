import time as real_time
from inspect import Signature
from math import isfinite
from typing import Any, Callable

from crosshair.core import FunctionInterps
from crosshair.register_contract import register_contract
from crosshair.statespace import context_statespace
from crosshair.tracers import NoTracing


def _gte_last(fn: Callable, value: Any) -> bool:
    with NoTracing():
        interps = context_statespace().extra(FunctionInterps)
        previous = interps._interpretations[fn]
        if len(previous) < 2:
            return True
    return value >= previous[-2]


def make_registrations():
    register_contract(
        real_time.time,
        post=lambda __return__: __return__ > 0.0,
        sig=Signature(parameters=[], return_annotation=float),
    )
    register_contract(
        real_time.time_ns,
        post=lambda __return__: __return__ > 0,
        sig=Signature(parameters=[], return_annotation=int),
    )
    register_contract(
        real_time.monotonic,
        post=lambda __return__: isfinite(__return__)
        and _gte_last(real_time.monotonic, __return__),
        sig=Signature(parameters=[], return_annotation=float),
    )
    register_contract(
        real_time.monotonic_ns,
        post=lambda __return__: isfinite(__return__)
        and _gte_last(real_time.monotonic_ns, __return__),
        sig=Signature(parameters=[], return_annotation=int),
    )
    register_contract(
        real_time.process_time,
        post=lambda __return__: _gte_last(real_time.process_time, __return__),
        sig=Signature(parameters=[], return_annotation=float),
    )
    register_contract(
        real_time.process_time_ns,
        post=lambda __return__: _gte_last(real_time.process_time_ns, __return__),
        sig=Signature(parameters=[], return_annotation=int),
    )
