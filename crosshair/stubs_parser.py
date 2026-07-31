from __future__ import annotations

import ast
import functools
import re
import sys
from collections.abc import __all__ as abc_all
from importlib import import_module
from inspect import Parameter, Signature, signature
from types import ClassMethodDescriptorType, MethodDescriptorType, WrapperDescriptorType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from typing import __all__ as typing_all  # type: ignore

from crosshair.fnutil import resolve_signature
from crosshair.typeshed_lookup import FunctionDef as _FunctionDef
from crosshair.typeshed_lookup import qualname_funcdefs, stub_module_ast, stub_source
from crosshair.util import debug


def signature_from_stubs(fn: Callable) -> Tuple[List[Signature], bool]:
    """
    Try to find signature(s) for the given function in the stubs.

    For overloaded functions, all signatures found will be returned.

    :param fn: The function to lookup a signature for.
    :return: A list containing the signature(s) found, if any and a validity boolean.\
        If the boolean is False, signatures returned might be incomplete (some error\
        occured while parsing).
    """
    if getattr(fn, "__module__", None) and getattr(fn, "__qualname__", None):
        module_name = fn.__module__
    else:
        # Some builtins and some C functions are wrapped into Descriptors.
        if isinstance(
            fn, (MethodDescriptorType, WrapperDescriptorType, ClassMethodDescriptorType)
        ) and getattr(fn, "__qualname__", None):
            module_name = fn.__objclass__.__module__
        else:
            # Builtins classmethods have their module available only via __self__.
            fn_self = getattr(fn, "__self__", None)
            if isinstance(fn_self, type):
                module_name = fn_self.__module__
            else:
                return [], True

    # Use the `qualname` to find the function inside its module.
    path_in_module: List[str] = fn.__qualname__.split(".")
    fn_defs, defining_module = qualname_funcdefs(module_name, path_in_module)
    if not fn_defs:
        debug("No stub found for", module_name, fn.__qualname__)
        return [], True
    stub_text = stub_source(defining_module)
    if stub_text is None:
        debug("No stub source for module", defining_module)
        return [], True

    sigs: List[Signature] = []
    is_valid = True
    glo = dict(_stub_namespace(defining_module))
    for fn_def in fn_defs:
        sig, valid = _sig_from_functiondef(fn_def, stub_text, glo)
        if sig:
            sigs.append(sig)
        is_valid = is_valid and valid
    return sigs, is_valid


def _get_source_segment(source: str, node: ast.AST) -> Optional[str]:
    """Get source code segment of the *source* that generated *node*."""
    return ast.get_source_segment(source, node)


@functools.lru_cache(maxsize=None)
def _stub_namespace(module: str) -> Dict[str, Any]:
    """The namespace a module's stub annotations resolve against: this module's
    globals plus the stub's own imports and TypeVar definitions."""
    glo = globals().copy()
    stub_text = stub_source(module)
    module_ast = stub_module_ast(module)
    if stub_text is not None and module_ast is not None:
        _exec_stub_definitions(module_ast.body, stub_text, glo)
    return glo


def _exec_stub_definitions(
    stmts: List[ast.stmt], stub_text: str, glo: Dict[str, Any]
) -> None:
    """Execute a stub's imports and TypeVar assignments into ``glo``, descending into
    the branch of each ``sys``-dependent ``if`` that this interpreter takes."""
    for node in stmts:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _exec_import(node, glo)

        elif isinstance(node, ast.Assign):
            value_text = _get_source_segment(stub_text, node.value)
            if value_text and "TypeVar" in value_text:
                assign_text = _get_source_segment(stub_text, node)
                if assign_text:
                    try:
                        exec(assign_text, glo)
                    except Exception:
                        debug("Not able to evaluate TypeVar assignment:", assign_text)

        elif isinstance(node, ast.If):
            test_text = _get_source_segment(stub_text, node.test)
            if not test_text:
                continue
            try:
                condition = eval(test_text, glo)
            except Exception:
                debug("Not able to evaluate condition:", test_text)
                continue
            _exec_stub_definitions(
                node.body if condition else node.orelse, stub_text, glo
            )


def _exec_import(imp: Union[ast.Import, ast.ImportFrom], glo: Dict[str, Any]):
    """Try to execute the import statement and add it to the `glo` namespace."""
    if isinstance(imp, ast.Import):
        for n in imp.names:
            name = n.name
            asname = n.asname or name
            if name != "_typeshed":
                try:
                    glo[asname] = import_module(name)
                except Exception:
                    debug("Not able to import", name)

    elif isinstance(imp, ast.ImportFrom):
        # Replace imports from `_typeshed` by their equivalent
        if imp.module == "_typeshed":
            for n in imp.names:
                name = n.name
                asname = n.asname or name
                if name in _REPLACE_TYPESHED:
                    new_module, replace = _REPLACE_TYPESHED[name]
                    glo[asname] = getattr(import_module(new_module), replace)
                elif name == "Self":
                    Self = TypeVar("Self")
                    glo["Self"] = Self
        elif imp.module:
            try:
                module = import_module(imp.module)
            except Exception:
                debug("Not able to import", imp.module)
                return
            for n in imp.names:
                name = n.name
                asname = n.asname or name
                try:
                    glo[asname] = getattr(module, name)
                except Exception:
                    debug("Not able to import", name, "from", imp.module)


# Replace _typeshed imports by their closest equivalent
_collection_module = "typing" if sys.version_info < (3, 9) else "collections.abc"
_REPLACE_TYPESHED: Dict[str, Tuple[str, str]] = {
    "SupportsLenAndGetItem": (_collection_module, "Collection"),
    "SupportsNext": (_collection_module, "Iterator"),
    "SupportsAnext": (_collection_module, "AsyncIterator"),
    # One might wish to add more if needed, but exact equivalents do not exist.
}


def _sig_from_functiondef(
    fn_def: _FunctionDef, stub_text: str, glo: Dict[str, Any]
) -> Tuple[Optional[Signature], bool]:
    """Given an ast FunctionDef, return the corresponding signature."""
    # Get the source text for the function stub and parse the signature from it.
    function_text = _get_source_segment(stub_text, fn_def)
    if function_text:
        exec(function_text, glo)
        sig_or_error = resolve_signature(glo[fn_def.name])
        if isinstance(sig_or_error, str):
            try:
                sig_or_error = signature(glo[fn_def.name])
            except Exception:
                debug("Not able to perform function evaluation:", function_text)
                return None, False
        parsed_sig, valid = _parse_sig(sig_or_error, glo)
        # If the function is @classmethod, remove cls from the signature.
        for decorator in fn_def.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "classmethod":
                oldparams = list(parsed_sig.parameters.values())
                newparams = oldparams[1:]
                slf = "Self"
                if (
                    slf in glo
                    and oldparams[0].annotation == Type[glo[slf]]
                    and parsed_sig.return_annotation == glo[slf]
                ):
                    # We don't support return type "Self" in classmethods.
                    return (
                        parsed_sig.replace(
                            parameters=newparams,
                            return_annotation=Parameter.empty,
                        ),
                        False,
                    )
                return parsed_sig.replace(parameters=newparams), valid
        return parsed_sig, valid
    return None, False


def _parse_sig(sig: Signature, glo: Dict[str, Any]) -> Tuple[Signature, bool]:
    """
    Signature annotations are escaped into strings.

    This is due to `from __future__ import annotations`.
    """
    is_valid = True
    ret_type, valid = _parse_annotation(sig.return_annotation, glo)
    is_valid = is_valid and valid
    params: List[Parameter] = []
    for param in sig.parameters.values():
        annot, valid = _parse_annotation(param.annotation, glo)
        params.append(param.replace(annotation=annot))
        is_valid = is_valid and valid
    return sig.replace(parameters=params, return_annotation=ret_type), is_valid


def _parse_annotation(annotation: Any, glo: Dict[str, Any]) -> Tuple[Any, bool]:
    if isinstance(annotation, str):
        if sys.version_info < (3, 10):
            annotation = _rewrite_with_union(annotation)
        try:
            return eval(annotation, glo), True
        except Exception as e:
            debug("Not able to parse annotation:", annotation, "Error:", e)
            return Parameter.empty, False
    return annotation, True


def _rewrite_with_union(s: str) -> str:
    """
    Undo PEP 604 to be compliant with Python < 3.10.

    For example `Dict[str | int]` will become `Dict[Union[str, int]]`

    Main idea of the algorithm:
    - Walk through the string and remember each opening parenthesis or bracket (push the
      current state to the saved states).
    - Uppon closing a parenthesis or bracket, if a `|` was found since the opening
      parenthesis, surround with `Union[]` and replace `|` by `,`. Then pop the state
      from the saved states.
    Note: the given string is assumed to have a valid syntax.
    """
    s_new = s  # The new string being built
    saved_states: List[Tuple[int, bool]] = []  # Stack of saved states
    start: int = 0  # Index (in s_new) where Union would begin
    found: bool = False  # True if a `|` was found since `start`
    idx: int = 0  # Current index in `s_new`

    for char in s:
        if char == "|":
            found = True

        # Closing the current scope. Surround with `Union[]` if a `|` was found.
        if char == ")" or char == "]" or char == ",":
            if found:
                s_new = (
                    s_new[: start + 1]
                    + "Union["
                    + s_new[start + 1 : idx].replace("|", ",")
                    + "]"
                    + s_new[idx:]
                )
                idx += len("Union[]")
            if char != ",":
                start, found = saved_states.pop()  # Restore previous scope.

        # Opening a new scope.
        if char == "(" or char == "[" or char == ",":
            if char != ",":
                saved_states.append((start, found))  # Save the current scope.
            start = idx
            found = False
        idx += 1

    if found:
        s_new = "Union[" + s_new.replace("|", ",") + "]"
    return s_new
