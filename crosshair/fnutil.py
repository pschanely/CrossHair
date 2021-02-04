import builtins
import importlib
import traceback
from dataclasses import dataclass
from inspect import getclosurevars
from inspect import isfunction
from inspect import signature
from inspect import Signature
from types import FunctionType, BuiltinFunctionType
from types import ModuleType
import sys
from typing import *

from crosshair.util import debug
from crosshair.util import import_module
from crosshair.util import source_position
from crosshair.util import ErrorDuringImport

if sys.version_info >= (3, 8):

    class Descriptor(Protocol):
        def __get__(self, instance: object, cls: type) -> Any:
            ...


else:
    Descriptor = Any


def fn_globals(fn: Callable) -> Dict[str, object]:
    if hasattr(fn, "__wrapped__"):
        return fn_globals(fn.__wrapped__)  # type: ignore
    if isfunction(fn):  # excludes built-ins, which don't have closurevars
        closure_vars = getclosurevars(fn)
        if closure_vars.nonlocals:
            return {**closure_vars.nonlocals, **getattr(fn, "__globals__", {})}
    if hasattr(fn, "__globals__"):
        return fn.__globals__  # type:ignore
    return builtins.__dict__


def resolve_signature(fn: Callable) -> Union[Signature, str]:
    """
    Get signature and resolve type annotations with get_type_hints.
    Returns a pair of Nones if no signature is available for the function.
    (e.g. it's implemented in C)
    Returns an unresolved signature and an error message if the type resultion errors.
    (e.g. the annotation references a type name that isn't dfined)
    """
    # TODO: Test resolution with members at multiple places in the hierarchy.
    # e.g. https://bugs.python.org/issue29966
    try:
        sig = signature(fn)
    except ValueError:
        # Happens, for example, on builtins
        return "No signature available"
    try:
        type_hints = get_type_hints(fn, fn_globals(fn))
    except NameError as name_error:
        return str(name_error)
        # filename, lineno = source_position(fn)
        # return (sig, ConditionSyntaxMessage(filename, lineno, str(name_error)))
    params = sig.parameters.values()
    newparams = []
    for name, param in sig.parameters.items():
        if name in type_hints:
            param = param.replace(annotation=type_hints[name])
        newparams.append(param)
    newreturn = type_hints.get("return", sig.return_annotation)
    return Signature(newparams, return_annotation=newreturn)


def set_first_arg_type(sig: Signature, first_arg_type: type) -> Signature:
    newparams = list(sig.parameters.values())
    newparams[0] = newparams[0].replace(annotation=first_arg_type)
    return Signature(newparams, return_annotation=sig.return_annotation)


@dataclass
class FunctionInfo:
    """
    Abstractions around code.

    Allows you to access, inspect the signatures of, and patch
    code in a module or class, even when that code is wrapped in
    decorators like @staticmethod, @classmethod, and @property.
    """

    context: Union[type, ModuleType, None]
    name: str
    descriptor: Descriptor

    @staticmethod
    def from_module(context: ModuleType, name: str) -> "FunctionInfo":
        return FunctionInfo(context, name, context.__dict__[name])

    @staticmethod
    def from_class(context: type, name: str) -> "FunctionInfo":
        return FunctionInfo(context, name, context.__dict__[name])

    @staticmethod
    def from_fn(fn: Callable) -> "FunctionInfo":
        return FunctionInfo(None, fn.__name__, fn)  # type: ignore

    def callable(self) -> Tuple[Callable, Signature]:
        maybe = self.get_callable()
        assert maybe is not None
        return maybe

    def get_callable(self) -> Optional[Tuple[Callable, Signature]]:
        ctx, desc = self.context, self.descriptor
        if isinstance(ctx, ModuleType) or ctx is None:
            fn = cast(Callable, desc)
            sig = resolve_signature(fn)
            if isinstance(sig, Signature):
                return (fn, sig)
        else:
            if isinstance(desc, FunctionType):
                sig = resolve_signature(desc)
                if isinstance(sig, Signature):
                    return (desc, set_first_arg_type(sig, ctx))
            elif isinstance(desc, staticmethod):
                sig = resolve_signature(desc.__func__)
                if isinstance(sig, Signature):
                    return (desc.__func__, sig)
            elif isinstance(desc, classmethod):
                sig = resolve_signature(desc.__func__)
                if isinstance(sig, Signature):
                    return (desc.__func__, set_first_arg_type(sig, Type[ctx]))
            elif isinstance(desc, property):
                if desc.fget and not desc.fset and not desc.fdel:
                    sig = resolve_signature(desc.fget)
                    if isinstance(sig, Signature):
                        return (desc.fget, set_first_arg_type(sig, ctx))
        # Cannot get a signature:
        return None

    def patch_logic(self, patched: Callable) -> Union[None, Callable, Descriptor]:
        desc = self.descriptor
        if isinstance(desc, FunctionType):
            return patched
        elif isinstance(desc, staticmethod):
            return staticmethod(patched)
        elif isinstance(desc, classmethod):
            return classmethod(patched)
        elif isinstance(desc, property):
            return property(fget=patched, fset=desc.fset, fdel=desc.fdel)  # type: ignore
        return None


class NotFound(ValueError):
    pass  # TODO this seems unecessary


def walk_qualname(obj: Union[type, ModuleType], name: str) -> FunctionInfo:
    """
    >>> walk_qualname(builtins, 'sum') == FunctionInfo.from_module(builtins, 'sum')
    True
    >>> walk_qualname(list, 'append') == FunctionInfo.from_class(list, 'append')
    True
    >>> class Foo:
    ...   class Bar:
    ...     def doit():
    ...       pass
    >>> walk_qualname(Foo, 'Bar.doit') == FunctionInfo.from_class(Foo.Bar, 'doit')
    True
    """
    parts = name.split(".")
    for part in parts[:-1]:
        if part == "<locals>":
            raise ValueError("object defined inline are non-addressable(" + name + ")")
        if not hasattr(obj, part):
            raise NotFound(f'Name "{part}" not found on object "{obj}"')
        obj = getattr(obj, part)
    lastpart = parts[-1]
    if lastpart not in obj.__dict__:
        raise NotFound(f'Name "{lastpart}" not found on object "{obj}"')
    assert isinstance(obj, (type, ModuleType))
    return FunctionInfo(obj, lastpart, obj.__dict__[lastpart])


def load_by_qualname(name: str) -> FunctionInfo:
    """
    >>> type(load_by_qualname('os'))
    <class 'module'>
    >>> type(load_by_qualname('os.path'))
    <class 'module'>
    >>> type(load_by_qualname('os.path.join')).__name__
    'FunctionInfo'
    >>> type(load_by_qualname('pathlib.Path')).__name__
    'FunctionInfo'
    >>> type(load_by_qualname('pathlib.Path.is_dir')).__name__
    'FunctionInfo'
    """
    parts = name.split(".")
    # try progressively shorter prefixes until we can load a module:
    for i in reversed(range(1, len(parts) + 1)):
        cur_module_name = ".".join(parts[:i])
        try:
            try:
                spec_exists = importlib.util.find_spec(cur_module_name) is not None
                if not spec_exists:
                    raise ModuleNotFoundError(f"No module named '{cur_module_name}'")
            except ModuleNotFoundError:
                if i == 1:
                    raise
                else:
                    continue
            module = import_module(cur_module_name)
        except Exception as e:
            raise ErrorDuringImport(e, traceback.extract_tb(sys.exc_info()[2])[-1])
        remaining = ".".join(parts[i:])
        if remaining:
            return walk_qualname(module, remaining)
        else:
            return module
    assert False
