import hashlib
import sys
from typing import Callable, Dict

from crosshair.core import register_patch, with_realized_args


def make_registrations():
    # These C functions reject a symbolic name or parameter.
    # (scrypt requires OpenSSL, so it is absent on some builds, e.g. pyodide)
    for name in ("new", "scrypt"):
        fn = getattr(hashlib, name, None)
        if fn is not None:
            register_patch(fn, with_realized_args(fn))
    if sys.version_info < (3, 12):
        # As of Python 3.12, SymbolicBytes can implement __buffer__() to be compatible
        # with hash functions. Prior to that, we patch them manually:

        # A dict (not a set) keeps the patch order deterministic.
        to_patch: Dict[Callable, None] = {}
        for algo_string in sorted(hashlib.algorithms_available):
            hash_constructor = getattr(hashlib, algo_string, None)
            if hash_constructor is not None:
                to_patch[hash_constructor] = None
            example_instance = hashlib.new(algo_string)
            update_method = getattr((type(example_instance)), "update")
            to_patch[update_method] = None
        for fn in to_patch:
            register_patch(fn, with_realized_args(fn))
