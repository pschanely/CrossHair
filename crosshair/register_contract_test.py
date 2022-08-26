import random
import sys
import time
from inspect import Parameter, Signature
from random import Random, randint
from typing import Union, overload

import numpy as np
import pytest

import crosshair.register_contract
from crosshair.register_contract import (
    ContractRegistrationError,
    get_contract,
    register_contract,
    register_modules,
)
from crosshair.statespace import CONFIRMED, POST_FAIL, MessageType
from crosshair.test_util import check_states


@pytest.fixture(autouse=True)
def clear_registrations():
    """Revert the registered contracts and modules after each test."""
    orig_contracts = crosshair.register_contract.REGISTERED_CONTRACTS.copy()
    orig_modules = crosshair.register_contract.REGISTERED_MODULES.copy()
    crosshair.register_contract.REGISTERED_CONTRACTS = {}
    crosshair.register_contract.REGISTERED_MODULES = set()
    yield None
    crosshair.register_contract.REGISTERED_CONTRACTS = orig_contracts
    crosshair.register_contract.REGISTERED_MODULES = orig_modules


def test_register_bound_method():
    with pytest.raises(ContractRegistrationError):
        register_contract(randint)


def test_register_malformed_contract():
    with pytest.raises(ContractRegistrationError):
        register_contract(
            Random.randint,
            pre=lambda a, wrong_name: a <= wrong_name,
        )


def test_register_malformed_signature():
    with pytest.raises(ContractRegistrationError):
        # The signature has wrong arg names (should be `a` and `b`).
        sig = Signature(
            [
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("low", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                Parameter("high", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            ],
            return_annotation=int,
        )
        register_contract(Random.randint, sig=sig)


def test_register_randint():
    def f(x: int) -> int:
        """
        pre: x <= 10
        post: _ >= x
        """
        return randint(x, 10)

    randint_sig = None
    # Stub parser is not available for python < 3.8.
    if sys.version_info < (3, 8):
        randint_sig = Signature(
            parameters=[
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD, annotation=Random),
                Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                Parameter("b", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            ],
            return_annotation=int,
        )
    register_contract(
        Random.randint,
        pre=lambda a, b: a <= b,
        post=lambda __return__, a, b: a <= __return__ <= b,
        sig=randint_sig,
    )
    check_states(f, CONFIRMED)


def test_register_numpy_randint():
    def f(x: int) -> int:
        """
        pre: x < 10
        post: _ >= x
        """
        return np.random.randint(x, 10)

    sig = Signature(
        parameters=[
            Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
            Parameter("low", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            Parameter("high", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
        ],
        return_annotation=int,
    )
    register_contract(
        np.random.RandomState.randint,
        pre=lambda low, high: low < high,
        post=lambda __return__, low, high: low <= __return__ < high,
        sig=sig,
    )
    check_states(f, CONFIRMED)


def test_register_overload():
    @overload
    def overld(a: int) -> int:
        ...

    @overload
    def overld(a: str) -> str:
        ...

    def overld(a: Union[int, str]) -> Union[int, str]:
        if isinstance(a, int):
            return -a if a < 0 else a
        else:
            return "hello: " + a

    def f1(x: int) -> int:
        """
        post: _ >= 0
        """
        return overld(x)

    def f2(x: str) -> str:
        """
        post: len(_) > 6
        """
        return overld(x)

    sig_1 = Signature(
        parameters=[Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, annotation=int)],
        return_annotation=int,
    )
    sig_2 = Signature(
        parameters=[Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, annotation=str)],
        return_annotation=str,
    )

    def post_cond(a, __return__):
        if isinstance(a, str):
            return len(__return__) > 6
        if isinstance(a, int):
            return __return__ > 0
        return False

    register_contract(
        overld,
        post=post_cond,
        sig=[sig_1, sig_2],
    )
    check_states(f1, CONFIRMED)
    check_states(f2, CONFIRMED)


def test_register_two_steps():
    def f():
        pass

    sig1 = Signature(parameters=[], return_annotation=int)
    sig2 = Signature(parameters=[], return_annotation=float)
    sig3 = Signature(parameters=[], return_annotation=str)
    register_contract(f, sig=sig1)
    register_contract(f, sig=[sig2, sig3])
    assert get_contract(f).sigs == [sig1, sig2, sig3]


def test_register_twice_with_different_post():
    def f() -> int:
        return 4

    register_contract(f)
    with pytest.raises(ContractRegistrationError):
        register_contract(f, post=lambda __return__: __return__ == 4)


if sys.version_info >= (3, 8):

    def test_register_modules():
        def f() -> int:
            """
            post: _ >= 0
            """
            return time.time_ns()

        register_modules(time)
        check_states(f, POST_FAIL)
        crosshair.register_contract

        def f() -> int:
            """
            post: _ > 0
            """
            return randint(5, 10)

        register_modules(random)
        check_states(f, POST_FAIL)
