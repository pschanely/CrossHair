"""API for registering contracts for external libraries."""
from dataclasses import dataclass
from inspect import signature
from typing import Callable, Dict, Optional


class ContractRegistrationError(Exception):
    pass


@dataclass
class Contract:
    pre: Optional[Callable[..., bool]]
    post: Optional[Callable[..., bool]]


REGISTERED_CONTRACTS: Dict[str, Contract] = {}


def _verify_signatures(
    fn: Callable,
    pre: Optional[Callable[..., bool]],
    post: Optional[Callable[..., bool]],
) -> None:
    """Verify if pre- and post-condition signatures are valid."""
    fn_params = set(signature(fn).parameters.keys())
    if pre:
        pre_params = set(signature(pre).parameters.keys())
        if not pre_params <= fn_params:
            raise ContractRegistrationError(
                f"Malformated precondition for function {fn.__name__}. Unexpected arguments: {pre_params - fn_params}"
            )
    if post:
        post_params = set(signature(post).parameters.keys())
        fn_params.add("result")
        fn_params.add("OLD")
        if not post_params <= fn_params:
            raise ContractRegistrationError(
                f"Malformated postcondition for function {fn.__name__}. Unexpected parameters: {post_params - fn_params}"
            )


def register_contract(
    fn: Callable,
    *,
    pre: Optional[Callable[..., bool]] = None,
    post: Optional[Callable[..., bool]] = None,
) -> None:
    """
    Register a contract for the given function.

    :param fn: The function to add a contract for.
    :param pre: The preconditon which should hold when entering the function.
    :param post: The postcondition which should hold when returning from the function.
    :raise: `ContractRegistrationError` if the registered contract is malformed.
    """
    _verify_signatures(fn, pre, post)
    REGISTERED_CONTRACTS[fn.__module__ + fn.__name__] = Contract(pre, post)


def get_contract(fn: Callable) -> Optional[Contract]:
    """
    Get the contract associated to the given function, it the function was registered.

    :param fn: The function to retrieve the contract for.
    :return: The contract associated with the function or None if the function was not registered.
    """
    if (
        getattr(fn, "__module__", None) is not None
        and getattr(fn, "__name__", None) is not None
    ):
        return REGISTERED_CONTRACTS.get(fn.__module__ + fn.__name__)
    return None
