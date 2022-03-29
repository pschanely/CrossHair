from inspect import Parameter, Signature
import pytest
from crosshair.register_contract import (
    ContractRegistrationError,
    register_contract,
)
from random import Random, randint


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
