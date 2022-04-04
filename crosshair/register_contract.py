"""API for registering contracts for external libraries."""
from dataclasses import dataclass
from inspect import Parameter, Signature, ismethod, signature
from typing import Callable, Dict, List, Optional, Union

from crosshair.stubs_parser import signature_from_stubs
from crosshair.util import debug

# TODO: One might want to add more features to overloading. Currently contracts only
# support multiple signatures, but not pre- and postconditions depending on the
# overload. REGISTERED_CONTRACTS might become Dict[Callable, List[ContractOverride]].


class ContractRegistrationError(Exception):
    pass


@dataclass
class ContractOverride:
    pre: Optional[Callable[..., bool]]
    post: Optional[Callable[..., bool]]
    sigs: List[Signature]
    skip_body: bool
    # TODO: Once supported, we might want to register Exceptions ("raises") as well


REGISTERED_CONTRACTS: Dict[Callable, ContractOverride] = {}


def _verify_signatures(
    fn: Callable, contract: ContractOverride, ref_sig: Optional[Signature]
) -> None:
    """Verify the provided signatures (including signatures of `pre` and `post`)."""
    if ref_sig:
        all_sigs = contract.sigs.copy()
        all_sigs.append(ref_sig)
    else:
        all_sigs = contract.sigs
    for sig in all_sigs:
        params = set(sig.parameters.keys())
        # First verify the parameters against the reference sig.
        if ref_sig:
            ref_params = set(ref_sig.parameters.keys())
            # Cannot test for equality, because of overloads.
            if not params <= ref_params:
                raise ContractRegistrationError(
                    f"Malformed signature for function {fn.__name__}. "
                    f"Expected parameters: {ref_params}, found: {params}"
                )
        # Verify the signature of the precondition.
        if contract.pre:
            pre_params = set(signature(contract.pre).parameters.keys())
            if not pre_params <= params:
                raise ContractRegistrationError(
                    f"Malformated precondition for function {fn.__name__}. "
                    f"Unexpected arguments: {pre_params - params}"
                )
        # Verify the signature of the postcondition.
        if contract.post:
            post_params = set(signature(contract.post).parameters.keys())
            params.add("__return__")
            params.add("__old__")
            if not post_params <= params:
                raise ContractRegistrationError(
                    f"Malformated postcondition for function {fn.__name__}. "
                    f"Unexpected parameters: {post_params - params}."
                )


def _add_contract(
    fn: Callable, contract: ContractOverride, ref_sig: Optional[Signature]
) -> None:
    """Add a contract to the function and check for consistency."""
    _verify_signatures(fn, contract, ref_sig)
    old_contract = REGISTERED_CONTRACTS.get(fn)
    if old_contract:
        if (
            old_contract.pre == contract.pre
            and old_contract.post == contract.post
            and old_contract.skip_body == contract.skip_body
        ):
            old_contract.sigs.extend(contract.sigs)
        else:
            raise ContractRegistrationError(
                "Pre- and postconditons and skip_body should not differ when "
                f"registering multiple contracts for the same function: {fn.__name__}."
            )
    else:
        REGISTERED_CONTRACTS[fn] = contract


def register_contract(
    fn: Callable,
    *,
    pre: Optional[Callable[..., bool]] = None,
    post: Optional[Callable[..., bool]] = None,
    sig: Union[Signature, List[Signature], None] = None,
    skip_body: bool = True,
) -> None:
    """
    Register a contract for the given function.

    :param fn: The function to add a contract for.
    :param pre: The preconditon which should hold when entering the function.
    :param post: The postcondition which should hold when returning from the function.
    :param sig: If provided, CrossHair will use this signature for the function.\
        Usefull for manually providing type annotation. You can provide multiple\
        signatures for overloaded functions.
    :param skip_body: By default registered functions will be skipped executing,\
        assuming the postconditions hold. Set this to `False` to still execute the body.
    :raise: `ContractRegistrationError` if the registered contract is malformed or if\
        no signature is found for the contract.
    """
    # Don't allow registering bound methods.
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

    # In the case the signature is incomplete, look in the stubs.
    if not sig and (
        not reference_sig or reference_sig.return_annotation == Parameter.empty
    ):
        sigs = signature_from_stubs(fn)
        if sigs:
            # TODO: if the return type is generic, check that the same TypeVar is present in the args
            debug(f"Found {str(len(sigs))} signature(s) for {fn.__name__} in stubs")
            if any(sig.return_annotation == Parameter.empty for sig in sigs):
                raise ContractRegistrationError(
                    f"Incomplete signature for {fn.__name__} in stubs, consider "
                    f"registering the signature manually. Signatures found: {str(sigs)}"
                )
            contract = ContractOverride(pre, post, sigs, skip_body)
            _add_contract(fn, contract, reference_sig)
        else:
            raise ContractRegistrationError(
                f"No signature found for function {fn.__name__}, "
                "consider registering the signature manually."
            )
    else:
        # Verify the contract and register it.
        if sig is None:
            sig = []
        elif isinstance(sig, Signature):
            sig = [sig]
        contract = ContractOverride(pre, post, sig, skip_body)
        _add_contract(fn, contract, reference_sig)


def get_contract(fn: Callable) -> Optional[ContractOverride]:
    """
    Get the contract associated to the given function, it the function was registered.

    :param fn: The function to retrieve the contract for.
    :return: The contract associated with the function or None if the function was not\
        registered.
    """
    return REGISTERED_CONTRACTS.get(fn)
