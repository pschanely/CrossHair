import hashlib
import sys

from crosshair.core import register_patch, with_realized_args


def make_registrations():
    if sys.version_info < (3, 12):
        # As of Python 3.12, SymbolicBytes can implement __buffer__() to be compatible
        # with hash functions. Prior to that, we patch them manually:

        to_patch = {hashlib.new: None}  # we don't use a set so that the patch order
        # is deterministic, which matters for the patch_equivalence_test when
        # run under pytest -n
        for algo_string in sorted(hashlib.algorithms_available):
            hash_constructor = getattr(hashlib, algo_string, None)
            if hash_constructor is not None:
                to_patch[hash_constructor] = None
            try:
                example_instance = hashlib.new(algo_string)
            except ValueError:
                if sys.version_info < (3, 9):
                    # in 3.8, some "available" algorithms aren't available
                    continue
                else:
                    raise
            update_method = getattr((type(example_instance)), "update")
            to_patch[update_method] = None
        for fn in to_patch:
            register_patch(fn, with_realized_args(fn))
