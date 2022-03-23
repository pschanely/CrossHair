"""API for registering contracts for external libraries."""
from dataclasses import dataclass
from inspect import Parameter, Signature, ismethod, signature
from typing import Callable, Dict, Optional

from crosshair.stubs_parser import signature_from_stubs
from crosshair.util import debug


class ContractRegistrationError(Exception):
    pass


@dataclass
class Contract:
    pre: Optional[Callable[..., bool]]
    post: Optional[Callable[..., bool]]
    sig: Optional[Signature]  # TODO: Keep optional or not?


REGISTERED_CONTRACTS: Dict[Callable, Contract] = {}


def _verify_signatures(fn: Callable, contract: Contract, ref_sig: Signature) -> None:
    """Verify the provided signatures (including signatures of `pre` and `post`)."""
    params = list(ref_sig.parameters.keys())
    if contract.sig:
        sig_params = list(contract.sig.parameters.keys())
        if sig_params != params:
            raise ContractRegistrationError(
                f"Malformed signature for function {fn.__name__}. "
                f"Expected parameters: {params}, found: {sig_params}"
            )
    fn_params = set(params)
    if contract.pre:
        pre_params = set(signature(contract.pre).parameters.keys())
        if not pre_params <= fn_params:
            raise ContractRegistrationError(
                f"Malformated precondition for function {fn.__name__}. "
                f"Unexpected arguments: {pre_params - fn_params}"
            )
    if contract.post:
        post_params = set(signature(contract.post).parameters.keys())
        fn_params.add("result")
        fn_params.add("OLD")
        if not post_params <= fn_params:
            raise ContractRegistrationError(
                f"Malformated postcondition for function {fn.__name__}. "
                f"Unexpected parameters: {post_params - fn_params}."
            )


def register_contract(
    fn: Callable,
    *,
    pre: Optional[Callable[..., bool]] = None,
    post: Optional[Callable[..., bool]] = None,
    sig: Optional[Signature] = None,
) -> None:
    """
    Register a contract for the given function.

    :param fn: The function to add a contract for.
    :param pre: The preconditon which should hold when entering the function.
    :param post: The postcondition which should hold when returning from the function.
    :param sig: If provided, CrossHair will use this signature for the function.\
        Usefull for manually providing type annotation.
    :raise: `ContractRegistrationError` if the registered contract is malformed.
    """
    if ismethod(fn):
        cls = getattr(getattr(fn, "__self__", None), "__class__", None)
        if not cls:
            cls = "<class name not found>"
        raise ContractRegistrationError(
            f"You registered the bound method {fn}. You should register the unbound "
            f"function of the class {cls} instead."
        )
    reference_sig = None
    try:
        reference_sig = signature(fn)
    except ValueError:
        pass
    if not sig and reference_sig:
        sig = reference_sig
    if not sig or sig.return_annotation == Parameter.empty:
        sig = signature_from_stubs(fn)
        # TODO: if the return type is generic, check that the same TypeVar is present in the args
        if sig:
            debug(f"Found signature for {fn.__name__} in stubs:", sig)
    contract = Contract(pre, post, sig)
    if reference_sig:
        _verify_signatures(fn, contract, reference_sig)
    REGISTERED_CONTRACTS[fn] = contract


def get_contract(fn: Callable) -> Optional[Contract]:
    """
    Get the contract associated to the given function, it the function was registered.

    :param fn: The function to retrieve the contract for.
    :return: The contract associated with the function or None if the function was not\
        registered.
    """
    return REGISTERED_CONTRACTS.get(fn)
