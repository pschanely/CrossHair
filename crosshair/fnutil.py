import builtins
import importlib
import os
import re
import sys
from dataclasses import dataclass, field
from inspect import (
    Signature,
    getclosurevars,
    getmembers,
    isclass,
    isfunction,
    signature,
)
from os.path import samefile
from pathlib import Path
from types import FunctionType, ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_type_hints,
)

from crosshair.util import (
    ErrorDuringImport,
    debug,
    import_module,
    load_file,
    sourcelines,
)

if sys.version_info >= (3, 8):
    from typing import Protocol

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

    :param fn: a function whose signature we are interested in

    :return:
        An annotated signature object, or an error message if the type resultion errors.
        (e.g. the annotation references a type name that isn't dfined)
    """
    # TODO: Test resolution with members at multiple places in the hierarchy.
    # e.g. https://bugs.python.org/issue29966
    try:
        sig = signature(fn)
    except ValueError:
        # Happens, for example, on builtins
        return "No signature available"
    except Exception as exc:
        # Catchall for other ill-behaved functions. z3 functions, for instance,
        # can raise "z3.z3types.Z3Exception: Z3 AST expected"
        return f"No signature ({type(exc)})"
    try:
        type_hints = get_type_hints(fn, fn_globals(fn))
    except (
        # SymbolicObject has __annotations__ as a property, which the inspect modules
        # rejects with AttributeError:
        AttributeError,
        # type name not resolvable:
        NameError,
        # TODO: why does this one happen, again?:
        TypeError,
    ) as hints_error:
        return str(hints_error)
    newparams = []
    for name, param in sig.parameters.items():
        if name in type_hints:
            param = param.replace(annotation=type_hints[name])
        newparams.append(param)
    newreturn = type_hints.get("return", sig.return_annotation)
    return Signature(newparams, return_annotation=newreturn)


def set_first_arg_type(sig: Signature, first_arg_type: object) -> Signature:
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
    descriptor: Descriptor = field(compare=False)
    _sig: Union[None, Signature, str] = field(init=False, compare=False, default=None)

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

    def get_sig(self, fn: Callable) -> Optional[Signature]:
        sig = self._sig
        if sig is None:
            sig = resolve_signature(fn)
            self._sig = sig
        return sig if isinstance(sig, Signature) else None

    def get_callable(self) -> Optional[Tuple[Callable, Signature]]:
        ctx, desc = self.context, self.descriptor
        if isinstance(ctx, ModuleType) or ctx is None:
            fn = cast(Callable, desc)
            sig = self.get_sig(fn)
            if sig:
                return (fn, sig)
        else:
            if isinstance(desc, FunctionType):
                sig = self.get_sig(desc)
                if sig:
                    return (desc, set_first_arg_type(sig, ctx))
            elif isinstance(desc, staticmethod):
                sig = self.get_sig(desc.__func__)
                if sig:
                    return (desc.__func__, sig)
            elif isinstance(desc, classmethod):
                sig = self.get_sig(desc.__func__)
                if sig:
                    try:
                        ctx_type = Type.__getitem__(ctx)
                    except TypeError:  # Raised by "Type[Generic]" etc
                        return None
                    return (desc.__func__, set_first_arg_type(sig, ctx_type))
            elif isinstance(desc, property):
                if desc.fget and not desc.fset and not desc.fdel:
                    sig = self.get_sig(desc.fget)
                    if sig:
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
    pass


def walk_qualname(obj: Union[type, ModuleType], name: str) -> Union[type, FunctionInfo]:
    """
    Resolve the function info by walking through the ``obj``.

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
    >>> walk_qualname(Foo, 'Bar') == Foo.Bar
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
    target = obj.__dict__[lastpart]
    if isclass(target):
        return target
    return FunctionInfo(obj, lastpart, target)


def load_by_qualname(name: str) -> Union[type, FunctionInfo]:
    """
    Load the function info by the fully qualified name.

    raises: NotFound

    >>> type(load_by_qualname('os'))
    <class 'module'>
    >>> type(load_by_qualname('os.path'))
    <class 'module'>
    >>> type(load_by_qualname('pathlib.Path'))
    <class 'type'>
    >>> type(load_by_qualname('os.path.join')).__name__
    'FunctionInfo'
    >>> type(load_by_qualname('pathlib.Path.is_dir')).__name__
    'FunctionInfo'
    """
    parts = name.split(".")
    original_modules = set(sys.modules.keys())
    # try progressively shorter prefixes until we can load a module:
    for i in reversed(range(1, len(parts) + 1)):
        cur_module_name = ".".join(parts[:i])
        try:
            try:
                spec_exists = importlib.util.find_spec(cur_module_name) is not None
                if not spec_exists:
                    raise ModuleNotFoundError(f"No module named '{cur_module_name}'")
            except ModuleNotFoundError as exc:
                if i == 1:
                    raise NotFound(f"Module '{cur_module_name}' was not found") from exc
                else:
                    continue
            module = import_module(cur_module_name)
        except Exception as e:
            raise ErrorDuringImport from e
        remaining = ".".join(parts[i:])
        if remaining:
            return walk_qualname(module, remaining)
        else:
            return module
    assert False


def _contains_line(entity: object, filename: str, linenum: int):
    (cur_filename, start, lines) = sourcelines(entity)
    end = start + len(lines)
    try:
        return samefile(filename, cur_filename) and start <= linenum <= end
    except IOError:
        return False


def load_function_at_line(
    entity: Union[ModuleType, type], filename: str, linenum: int
) -> Optional[FunctionInfo]:
    """Load a function or method at a line number."""
    modulename = (
        entity.__name__ if isinstance(entity, ModuleType) else entity.__module__
    )
    for name, member in getmembers(entity):
        if getattr(member, "__module__", None) != modulename:
            # member was likely imported, but not defined here.
            continue
        if isfunction(member) and _contains_line(member, filename, linenum):
            return FunctionInfo(entity, name, entity.__dict__[name])
        if isclass(member):
            ctxfn = load_function_at_line(member, filename, linenum)
            if ctxfn:
                return ctxfn
    return None


def analyzable_filename(filename: str) -> bool:
    """
    Check whether the file can be analyzed purely based on the ``filename``.

    >>> analyzable_filename('foo23.py')
    True
    >>> analyzable_filename('#foo.py')
    False
    >>> analyzable_filename('23foo.py')
    False
    >>> analyzable_filename('setup.py')
    False
    """
    if not filename.endswith(".py"):
        return False
    lead_char = filename[0]
    if (not lead_char.isalpha()) and (not lead_char.isidentifier()):
        # (skip temporary editor files, backups, etc)
        debug(f"Skipping {filename} because it begins with a special character.")
        return False
    if filename in ("setup.py",):
        debug(
            f"Skipping {filename} because files with this name are not usually import-able."
        )
        return False
    return True


def walk_paths(paths: Iterable[Path], ignore_missing=False) -> Iterable[Path]:
    for path in paths:
        if not path.exists():
            if ignore_missing:
                continue
            else:
                raise FileNotFoundError(str(path))
        if path.is_dir():
            for (dirpath, _dirs, files) in os.walk(str(path)):
                for curfile in files:
                    if analyzable_filename(curfile):
                        yield Path(dirpath) / curfile
        else:
            yield path


_FILE_WITH_LINE_RE = re.compile(r"^(.*\.py)\:(\d+)$")


def load_files_or_qualnames(
    specifiers: Iterable[str],
) -> Iterable[Union[ModuleType, type, FunctionInfo]]:
    fspaths = []
    for specifier in specifiers:
        file_line_match = _FILE_WITH_LINE_RE.match(specifier)
        if file_line_match:
            filename, linestr = file_line_match.groups()
            linenum = int(linestr)
            fn = load_function_at_line(load_file(filename), filename, linenum)
            if fn is None:
                raise ErrorDuringImport(
                    f"Cannot find a function or method on line {linenum}."
                )
            yield fn
        elif specifier.endswith(".py") or os.path.isdir(specifier):
            fspaths.append(Path(specifier))
        else:
            yield load_by_qualname(specifier)
    for path in walk_paths(fspaths):
        yield load_file(str(path))
