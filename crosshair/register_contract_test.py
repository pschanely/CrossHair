from inspect import Parameter, Signature
from typing import Union, overload
import numpy as np
import pytest
from random import Random, randint

from crosshair.register_contract import (
    ContractRegistrationError,
    register_contract,
)
from crosshair.test_util import check_ok


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

    register_contract(
        Random.randint,
        pre=lambda a, b: a <= b,
        post=lambda __return__, a, b: a <= __return__ <= b,
    )
    actual, expected = check_ok(f)
    assert actual == expected


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
    actual, expected = check_ok(f)
    assert actual == expected


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
    actual, expected = check_ok(f1)
    assert actual == expected
    actual, expected = check_ok(f2)
    assert actual == expected
