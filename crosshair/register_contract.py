"""API for registering contracts for external libraries."""
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class Contract:
    pre: Optional[Callable[..., bool]]
    post: Optional[Callable[..., bool]]


REGISTERED_CONTRACTS: Dict[str, Contract] = {}


def register_contract(
    fn: Callable,
    *,
    pre: Optional[Callable[..., bool]] = None,
    post: Optional[Callable[..., bool]] = None
) -> None:
    """
    Register a contract for the given function.

    :param fn: The function to add a contract for.
    :param pre: The preconditon which should hold when entering the function.
    :param post: The postcondition which should hold when returning from the function.
    """
    REGISTERED_CONTRACTS[fn.__module__ + fn.__name__] = Contract(pre, post)


def get_contract(fn: Callable) -> Optional[Contract]:
    """
    Get the contract associated to the given function, it the function was registered.

    :param fn: The function to retrieve the contract for.
    :return: The contract associated with the function or None if the function was not registered.
    """
    return REGISTERED_CONTRACTS.get(fn.__module__ + fn.__name__)
