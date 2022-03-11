"""API for registering contracts for external libraries."""
from dataclasses import dataclass
from inspect import signature
import inspect
from typing import Callable, Dict, Optional


class ContractRegistrationError(Exception):
    pass


@dataclass
class Contract:
    pre: Optional[Callable[..., bool]]
    post: Optional[Callable[..., bool]]


REGISTERED_CONTRACTS: Dict[Callable, Contract] = {}


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
                f"Malformated postcondition for function {fn.__name__}. Unexpected parameters: {post_params - fn_params}."
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
    if not isinstance(fn, Callable):
        raise ContractRegistrationError(
            f"Cannot register {fn}, which is not a Callable."
        )
    if inspect.ismethod(fn):
        raise ContractRegistrationError(
            f"You registered the bound method {fn}. You should register the unbound function of the class {fn.__self__.__class__} instead."
        )
    _verify_signatures(fn, pre, post)
    REGISTERED_CONTRACTS[fn] = Contract(pre, post)


def get_contract(fn: Callable) -> Optional[Contract]:
    """
    Get the contract associated to the given function, it the function was registered.

    :param fn: The function to retrieve the contract for.
    :return: The contract associated with the function or None if the function was not registered.
    """
    return REGISTERED_CONTRACTS.get(fn)
