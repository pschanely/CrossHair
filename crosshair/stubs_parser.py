import ast
from inspect import _ParameterKind, Parameter, Signature
from typing import Callable, List, Optional, Union
from typeshed_client import get_stub_ast  # type: ignore

# TODO: add to the pip requirements

from crosshair.util import debug


def signature_from_stubs(fn: Callable) -> Optional[Signature]:
    """
    Try to find a signature for the given function in the stubs.

    :param fn: The function to lookup a signature for.
    :return: The signature found, if any.
    """
    # TODO: also try to execute the .pyi file directly
    if not hasattr(fn, "__module__") or not hasattr(fn, "__qualname__"):
        return None
    path_in_module: List[str] = fn.__qualname__.split(".")
    module = get_stub_ast(fn.__module__)
    if not module or not isinstance(module, ast.Module):
        debug("No ast found for module", fn.__module__)
        return None
    return _sig_from_ast(module, path_in_module)


def _sig_from_ast(
    _ast: Union[ast.Module, ast.ClassDef, ast.FunctionDef], next_steps: List[str]
) -> Optional[Signature]:
    """Lookup in the given ast for a function signature, following `next_steps` path."""
    # TODO: also look for if statements on the python version
    if len(next_steps) == 0:
        return None
    next_node_name = next_steps.pop(0)
    if len(next_steps) == 0:
        # Only one step remaining => Find the corresponding FunctionDef
        fn_defs = [
            node
            for node in _ast.body
            if isinstance(node, ast.FunctionDef) and node.name == next_node_name
        ]
        if len(fn_defs) != 1:
            debug(f"Multiple or no functions with name {next_node_name} in the AST.")
            return None
        return _sig_from_functiondef(fn_defs[0])
    # More than one step remaining => find the next node in the AST
    ast_next = [
        node
        for node in _ast.body
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.name == next_node_name
    ]
    if len(ast_next) != 1:
        debug(f"Multiple or no objects with name {next_node_name} in the AST.")
        return None
    return _sig_from_ast(ast_next[0], next_steps)


def _sig_from_functiondef(fn_def: ast.FunctionDef) -> Signature:
    """Given an ast FunctionDef, return the corresponding signature."""
    parameters: List[Parameter] = []
    # Positional args
    parameters.extend(
        _param_from_arg(arg, Parameter.POSITIONAL_OR_KEYWORD)
        for arg in fn_def.args.args
    )
    # Var-positional arg
    if fn_def.args.vararg:
        parameters.append(_param_from_arg(fn_def.args.vararg, Parameter.VAR_POSITIONAL))
    # Keyword args
    parameters.extend(
        _param_from_arg(arg, Parameter.KEYWORD_ONLY) for arg in fn_def.args.kwonlyargs
    )
    # Var-keyword arg
    if fn_def.args.kwarg:
        parameters.append(_param_from_arg(fn_def.args.kwarg, Parameter.VAR_KEYWORD))
    # Return annotation
    ret_annotation = _type_from_annotation(fn_def.returns)
    return Signature(parameters, return_annotation=ret_annotation)


def _param_from_arg(arg: ast.arg, param_type: _ParameterKind) -> Parameter:
    """Given an ast arg, return a signature parameter with the given parameter type."""
    annotation = _type_from_annotation(arg.annotation)
    return Parameter(arg.arg, param_type, annotation=annotation)


def _type_from_annotation(annotation: Optional[ast.expr]) -> Any:
    """Given an ast annotation, return the correponding type."""
    if isinstance(annotation, ast.Name):
        try:
            return eval(annotation.id)
            # TODO: problem with eval if word not imported --> should ensure the correct Namespace
        except Exception:
            pass
    return Parameter.empty
