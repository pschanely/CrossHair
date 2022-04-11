from __future__ import annotations
import ast
from collections.abc import __all__ as abc_all
from inspect import Parameter, Signature, signature
from pathlib import Path
import re
import sys
from types import MethodDescriptorType, WrapperDescriptorType
from typeshed_client import get_stub_ast, get_stub_file, get_search_context  # type: ignore
from typing import (  # type: ignore
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    __all__ as typing_all,
)

from crosshair.util import debug


def signature_from_stubs(fn: Callable) -> Tuple[List[Signature], bool]:
    """
    Try to find signature(s) for the given function in the stubs.

    Note: this feature is only available for Python >= 3.8.

    For overloaded functions, all signatures found will be returned.

    :param fn: The function to lookup a signature for.
    :return: A list containing the signature(s) found, if any and a validity boolean.\
        If the boolean is False, signatures returned might be incomplete (some error\
        occured while parsing).
    """
    # ast.get_source_segment requires Python 3.8
    if sys.version_info < (3, 8):
        return [], True
    if getattr(fn, "__module__", None) and getattr(fn, "__qualname__", None):
        module_name = fn.__module__
    else:
        # Some builtins and some C functions are wrapped into Descriptors
        if isinstance(fn, (MethodDescriptorType, WrapperDescriptorType)) and getattr(
            fn, "__qualname__", None
        ):
            module_name = fn.__objclass__.__module__
        else:
            return [], True
    # Use the `qualname` to find the function inside its module.
    path_in_module: List[str] = fn.__qualname__.split(".")
    # Find the stub_file and corresponding AST using `typeshed_client`.
    search_path = [Path(path) for path in sys.path if path]
    search_context = get_search_context(search_path=search_path)
    stub_file = get_stub_file(module_name, search_context=search_context)
    module = get_stub_ast(module_name, search_context=search_context)
    if not stub_file or not module or not isinstance(module, ast.Module):
        debug("No stub found for module", module_name)
        return [], True
    glo = globals().copy()
    return _sig_from_ast(module.body, path_in_module, stub_file.read_text(), glo)


def _sig_from_ast(
    stmts: List[ast.stmt],
    next_steps: List[str],
    stub_text: str,
    glo: Dict[str, Any],
) -> Tuple[List[Signature], bool]:
    """Lookup in the given ast for a function signature, following `next_steps` path."""
    if len(next_steps) == 0:
        return [], True

    # First walk through the nodes to execute imports and assignments
    for node in stmts:
        # If we encounter an import statement, add it to the namespace
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _exec_import(node, stub_text, glo)

        # If we encounter the definition of a `TypeVar`, add it to the namespace
        elif isinstance(node, ast.Assign):
            value_text = ast.get_source_segment(stub_text, node.value)
            if value_text and "TypeVar" in value_text:
                assign_text = ast.get_source_segment(stub_text, node)
                if assign_text:
                    try:
                        exec(assign_text, glo)
                    except Exception:
                        debug("Not able to evaluate TypeVar assignment:", assign_text)

    # Walk through the nodes to find the next node
    next_node_name = next_steps[0]
    sigs = []
    is_valid = True
    for node in stmts:
        # Only one step remaining => look for the function itself
        if (
            len(next_steps) == 1
            and isinstance(node, ast.FunctionDef)
            and node.name == next_node_name
        ):
            sig, valid = _sig_from_functiondef(node, stub_text, glo)
            if sig:
                sigs.append(sig)
            is_valid = is_valid and valid

        # More than one step remaining => look for the next step
        elif (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and node.name == next_node_name
        ):
            new_sigs, valid = _sig_from_ast(node.body, next_steps[1:], stub_text, glo)
            sigs.extend(new_sigs)
            is_valid = is_valid and valid

    # Additionally, we might need to look for the next node into if statements
    for node in stmts:
        if isinstance(node, (ast.If)):
            assign_text = ast.get_source_segment(stub_text, node.test)
            # Some function depends on the execution environment
            if assign_text and "sys." in assign_text:
                condition = None
                try:
                    condition = eval(assign_text, glo)
                except Exception:
                    debug("Not able to evaluate condition:", assign_text)
                if condition is not None:
                    new_sigs, valid = _sig_from_ast(
                        node.body if condition else node.orelse,
                        next_steps,
                        stub_text,
                        glo,
                    )
                    sigs.extend(new_sigs)
                    is_valid = is_valid and valid

    return sigs, is_valid


def _exec_import(
    imp: Union[ast.Import, ast.ImportFrom], stub_text: str, glo: Dict[str, Any]
):
    """Try to execute the import statement and add it to the `glo` namespace."""
    for n in imp.names:
        if n.name == "_typeshed":
            return
    # Replace imports from `_typeshed` by their equivalent
    module = getattr(imp, "module", None)
    if module == "_typeshed":
        for n in imp.names:
            name = n.name
            if name in _REPLACE_TYPESHED:
                new_module, replace = _REPLACE_TYPESHED[name]
                exec("from " + new_module + " import " + replace + " as " + name, glo)
            if name == "Self":
                Self = TypeVar("Self")
                glo["Self"] = Self
        return
    import_text = ast.get_source_segment(stub_text, imp)
    if import_text:
        try:
            exec(import_text, glo)
        except Exception:
            debug("Not able to perform import:", import_text)


# Replace _typeshed imports by their closest equivalent
_collection_module = "typing" if sys.version_info < (3, 9) else "collections.abc"
_REPLACE_TYPESHED: Dict[str, Tuple[str, str]] = {
    "SupportsLenAndGetItem": (_collection_module, "Collection"),
    "SupportsNext": (_collection_module, "Iterator"),
    "SupportsAnext": (_collection_module, "AsyncIterator"),
    # TODO: more to come
}


def _sig_from_functiondef(
    fn_def: ast.FunctionDef, stub_text: str, glo: Dict[str, Any]
) -> Tuple[Optional[Signature], bool]:
    """Given an ast FunctionDef, return the corresponding signature."""
    # Get the source text for the function stub and parse the signature from it.
    function_text = ast.get_source_segment(stub_text, fn_def)
    if function_text:
        try:
            exec(function_text, glo)
            sig = signature(glo[fn_def.name])
        except Exception:
            debug("Not able to perform function evaluation:", function_text)
            return None, False
        return _parse_sign(sig, glo)
    return None, False


def _parse_sign(sig: Signature, glo: Dict[str, Any]) -> Tuple[Signature, bool]:
    """
    In Python < 3.11, all signature annotations are escaped into strings.

    This is due to `from __future__ import annotations`.
    """
    is_valid = True
    if sys.version_info >= annotations.getMandatoryRelease():
        return sig, is_valid
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
        if sys.version_info < (3, 9):
            annotation = _rewrite_with_typing_types(annotation, glo)
        try:
            return eval(annotation, glo), True
        except Exception as e:
            debug("Not able to parse annotation:", annotation, "Error:", e)
            return Parameter.empty, False
    return Parameter.empty, True


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


_REPLACEMENTS_PEP_585: Dict[re.Pattern[str], str] = {}
"""Dictionnary of regexes and replacement strings to revert PEP 585."""

if sys.version_info < (3, 9):
    # 1. Replace type subscription by types from typing
    base = r"(?<![\.\w])"
    for t in typing_all:
        replacement = "typing." + t + "["
        _REPLACEMENTS_PEP_585[re.compile(base + t.lower() + r"\[")] = replacement

    # 2. Replace collections.abc by typing
    # (?<![\.\w]) is to avoid match if the char before is alphanumerical or a dot
    bases = [r"(?<![\.\w])collections\.abc\.", r"(?<![\.\w])"]
    for t in set(typing_all).intersection(abc_all):
        replacement = "typing." + t + "["
        for base in bases:
            _REPLACEMENTS_PEP_585[re.compile(base + t + r"\[")] = replacement
    # Special case for `from collections.abc import Set as AbstractSet`
    _REPLACEMENTS_PEP_585[re.compile(r"(?<![\.\w])AbstractSet\[")] = "typing.Set["


def _rewrite_with_typing_types(s: str, glo: Dict[str, Any]) -> str:
    """
    Undo PEP 585 to be compliant with Python < 3.9.

    For example `list[int]` will become `typing.List[int]` and types from
    collections.abc will be replaced by those of typing.
    """
    for regx, replace in _REPLACEMENTS_PEP_585.items():
        s_new = regx.sub(replace, s)
        if s != s_new and replace.startswith("typing.") and "typing" not in glo:
            exec("import typing", glo)
        s = s_new
    return s
