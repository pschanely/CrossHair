"""API for registering contracts for external libraries."""
from dataclasses import dataclass
from inspect import Parameter, Signature, getmodule, ismethod, signature
from types import MethodDescriptorType, ModuleType, WrapperDescriptorType
from typing import Callable, Dict, List, Optional, Set, Union
from weakref import ReferenceType

from crosshair.fnutil import resolve_signature
from crosshair.stubs_parser import signature_from_stubs
from crosshair.util import debug, warn

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
REGISTERED_MODULES: Set[ModuleType] = set()

# Don't automatically register those functions.
_NO_AUTO_REGISTER: Set[str] = {
    "__init__",
    "__init_subclass__",
    "__new__",
}


def _raise_or_warn(message: str, no_raises: bool) -> None:
    if no_raises:
        warn(message)
    else:
        raise ContractRegistrationError(message)


def required_param_names(sig: Signature) -> Set[str]:
    return {k for (k, v) in sig.parameters.items() if v.default is Parameter.empty}


def _verify_signatures(
    fn: Callable,
    contract: ContractOverride,
    ref_sig: Optional[Signature],
    no_raises: bool,
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
            # Cannot test for strict equality, because of overloads.
            if not required_param_names(sig) <= ref_params:
                _raise_or_warn(
                    f"Malformed signature for function {fn.__name__}. "
                    f"Expected parameters: {ref_params}, found: {params}",
                    no_raises,
                )
        # Verify the signature of the precondition.
        if contract.pre:
            pre_params = required_param_names(signature(contract.pre))
            if not pre_params <= params:
                _raise_or_warn(
                    f"Malformated precondition for function {fn.__name__}. "
                    f"Unexpected arguments: {pre_params - params}",
                    no_raises,
                )
        # Verify the signature of the postcondition.
        if contract.post:
            post_params = required_param_names(signature(contract.post))
            params.add("__return__")
            params.add("__old__")
            if not post_params <= params:
                _raise_or_warn(
                    f"Malformated postcondition for function {fn.__name__}. "
                    f"Unexpected parameters: {post_params - params}.",
                    no_raises,
                )


def _add_contract(
    fn: Callable,
    contract: ContractOverride,
    ref_sig: Optional[Signature],
    no_raises: bool,
) -> None:
    """Add a contract to the function and check for consistency."""
    _verify_signatures(fn, contract, ref_sig, no_raises)
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


def _internal_register_contract(
    fn: Callable,
    pre: Optional[Callable[..., bool]] = None,
    post: Optional[Callable[..., bool]] = None,
    sig: Union[Signature, List[Signature], None] = None,
    skip_body: bool = True,
    no_raises: bool = False,
) -> None:
    reference_sig = None

    sig_or_error = resolve_signature(fn)
    if isinstance(sig_or_error, Signature):
        reference_sig = sig_or_error

    # In the case the signature is incomplete, look in the stubs.
    if not sig and (
        not reference_sig or reference_sig.return_annotation == Parameter.empty
    ):
        sigs, is_valid = signature_from_stubs(fn)
        if sigs:
            debug(f"Found {str(len(sigs))} signature(s) for {fn.__name__} in stubs")
            if not is_valid or any(
                sig.return_annotation == Parameter.empty for sig in sigs
            ):
                _raise_or_warn(
                    f"Incomplete signature for {fn.__name__} in stubs, consider "
                    f"registering the signature manually. Signatures found: "
                    f"{str(sigs)}",
                    no_raises,
                )
            contract = ContractOverride(pre, post, sigs, skip_body)
            _add_contract(fn, contract, reference_sig, no_raises)
        else:
            if not no_raises or reference_sig:
                _raise_or_warn(
                    f"No valid signature found for function {fn.__name__}, "
                    "consider registering the signature manually.",
                    no_raises,
                )
            # We did not raise an error, and we have a reference signature available.
            if reference_sig:
                contract = ContractOverride(pre, post, [], skip_body)
                _add_contract(fn, contract, reference_sig, no_raises)
            # No signature available at all, we cannot register the function.
            else:
                _raise_or_warn(
                    f"Could not automatically register {fn.__name__}, reason: no "
                    "signature found.",
                    no_raises,
                )
    else:
        # Verify the contract and register it.
        if sig is None:
            sig = []
        elif isinstance(sig, Signature):
            sig = [sig]
        contract = ContractOverride(pre, post, sig, skip_body)
        _add_contract(fn, contract, reference_sig, no_raises)


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
        cls = getattr(getattr(fn, "__self__", None), "__class__", "<not found>")
        raise ContractRegistrationError(
            f"You registered the bound method {fn}. You should register the unbound "
            f"function of the class {cls} instead."
        )
    _internal_register_contract(fn, pre, post, sig, skip_body)


def get_contract(fn: Callable) -> Optional[ContractOverride]:
    """
    Get the contract associated to the given function, it the function was registered.

    :param fn: The function to retrieve the contract for.
    :return: The contract associated with the function or None if the function was not\
        registered.
    """
    # Weak references are not hashable: REGISTERED_CONTRACTS.get(fn) fails.
    if isinstance(fn, ReferenceType):
        return None
    # Return the registered contract for the function, if any.
    contract = REGISTERED_CONTRACTS.get(fn)
    if contract:
        return contract
    fn_name = getattr(fn, "__name__", None)
    # Some functions should not be automatically registered.
    if fn_name in _NO_AUTO_REGISTER:
        return None

    # If the function belongs to a registered module, register it.
    module = getmodule(fn)
    # If this is a classmethod, look for the module in the mro.
    fn_self = getattr(fn, "__self__", None)
    if fn_name and fn_self and isinstance(fn_self, type):
        for mro in fn_self.mro():
            if fn_name in mro.__dict__:
                module_name = mro.__module__
                if module_name in map(lambda x: x.__name__, REGISTERED_MODULES):
                    _internal_register_contract(fn, no_raises=True)
                    return REGISTERED_CONTRACTS.get(fn)
                return None
    if module and module in REGISTERED_MODULES:
        _internal_register_contract(fn, no_raises=True)
        return REGISTERED_CONTRACTS.get(fn)
    # Some builtins and some C functions are wrapped into Descriptors
    if isinstance(fn, (MethodDescriptorType, WrapperDescriptorType)):
        module_name = fn.__objclass__.__module__
        if module_name in map(lambda x: x.__name__, REGISTERED_MODULES):
            _internal_register_contract(fn, no_raises=True)
            return REGISTERED_CONTRACTS.get(fn)
    return None


def register_modules(*modules: ModuleType) -> None:
    """
    Specify a module whose functions should all be skipped at execution.

    THIS IS AN EXPERIMENTAL FEATURE!
    Registering a whole module might be too much in some cases and you might fallback to
    register individual functions instead.

    Note that functions `__init__`, `__init_subclass__` and `__new__` are never
    registered automatically.

    If you wish to register all functions, except function `foo`, register that function
    manually with the option `skip_body=False`.

    :param modules: one or multiple modules whose functions should be skipped.
    """
    REGISTERED_MODULES.update(modules)
