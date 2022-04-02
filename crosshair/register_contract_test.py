from inspect import Parameter, Signature
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
        # The signature is missing the parameter `self`
        sig = Signature(
            [
                Parameter("a", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                Parameter("b", Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
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
        post=lambda __return__, a, b: a <= __return__ and __return__ <= b,
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
        post=lambda __return__, low, high: low <= __return__ and __return__ < high,
        sig=sig,
    )

    actual, expected = check_ok(f)
    assert actual == expected
