"""API for registering contracts for external libraries."""
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class Contract:
    pre: Optional[Callable[..., bool]]
    post: Optional[Callable[..., bool]]


REGISTERED_CONTRACTS: Dict[Callable, Contract] = {}


def register_contract(
    fn: Callable,
    *,
    pre: Optional[Callable[..., bool]] = None,
    post: Optional[Callable[..., bool]] = None
) -> None:
    """
    Register a contract for the given function.

    :param fn: The function you wish to add a contract.
    :param pre: The preconditon which should hold when entering the function.
    :param post: The postcondition which should hold when returning from the function.
    """
    REGISTERED_CONTRACTS[fn] = Contract(pre, post)
