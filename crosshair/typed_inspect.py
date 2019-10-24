import os.path
import inspect
import importlib
from typing import *
from crosshair.util import debug, walk_qualname


def _has_annotations(sig: inspect.Signature):
    if sig.return_annotation != inspect.Signature.empty:
        return True
    for p in sig.parameters.values():
        if p.annotation != inspect.Parameter.empty:
            return True
    return False


def get_resolved_sig(fn: Callable, env=None) -> inspect.Signature:
    sig = inspect.signature(fn)
    #debug('get_resolved_seg input:', sig, next(iter(sig.parameters.keys())), inspect.ismethod(fn))
    type_hints = get_type_hints(fn, env, env.copy() if env else None)
    params = sig.parameters.values()
    if len(params) > 0 and next(iter(params)).name == 'self' and 'self' not in type_hints:
        fn_module = inspect.getmodule(fn)
        if fn_module:
            defining_thing = walk_qualname(
                fn_module, fn.__qualname__.rsplit('.', 1)[0])
            if inspect.isclass(defining_thing):
                type_hints['self'] = defining_thing
    #debug('TO HINTS ', type_hints)
    newparams = []
    for name, param in sig.parameters.items():
        if name in type_hints:
            param = param.replace(annotation=type_hints[name])
        newparams.append(param)
    newreturn = type_hints.get('return', sig.return_annotation)
    sig = inspect.Signature(newparams, return_annotation=newreturn)
    #debug('get_resolved_sig output: ', sig)
    return sig


def signature(fn: Callable, _stub_path: Optional[List[str]] = None) -> inspect.Signature:
    sig = get_resolved_sig(fn)
    debug('START ', fn.__name__, sig)
    if _has_annotations(sig):
        debug('has annotations already')
        return sig

    if _stub_path is None and os.environ.get('PYTHONPATH'):
        _stub_path = os.environ['PYTHONPATH'].split(':')

    if _stub_path is None:
        _stub_path = []

    try:
        src_file = inspect.getsourcefile(fn)
    except TypeError:  # raises this for builtins
        return sig
    if not src_file.endswith('.py'):
        debug(src_file, ' not ending in py')
        return sig
    pyi_file = src_file + 'i'
    if not os.path.exists(pyi_file):
        #debug('no pyi at ', pyi_file)
        filename = os.path.split(pyi_file)[1]
        for path in _stub_path:
            candidate_file = os.path.join(path, filename)
            if os.path.exists(candidate_file):
                pyi_file = candidate_file
                break
            #debug('no pyi at ', candidate_file)
    if not os.path.exists(pyi_file):
        debug('no pyi found on PYTHONPATH')
        return sig
    #debug('pyi found at ', pyi_file)

    loader = importlib.machinery.SourceFileLoader(fn.__module__, pyi_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    #debug('signature spec ', spec)
    ptr: Any = importlib.util.module_from_spec(spec)
    import sys
    old_module = sys.modules[spec.name]
    try:
        loader.exec_module(ptr)
    except BaseException as e:
        debug('Failed to load ' + pyi_file + '(' + str(e) + '); ignoring')
        return sig
    if old_module is not sys.modules[spec.name]:
        raise Exception('sys modules changed')

    ptr = walk_qualname(ptr, fn.__qualname__)
    return get_resolved_sig(ptr, inspect.getmodule(fn).__dict__)


def _assert_signature_works():
    '''
    post: 'return_annotation' in return
    post: 'self' in return
    post: 'parameters' in return
    post: return['return_annotation'].annotation  == Any
    post: return['parameters'].annotation == Optional[Sequence[inspect.Parameter]]
    '''
    return signature(inspect.Signature.__init__).parameters
