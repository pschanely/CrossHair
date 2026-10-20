import os

from crosshair.core import register_patch, with_realized_args


def make_registrations():
    register_patch(os.fspath, os._fspath)
    # Device-number helpers are C functions that reject a symbolic proxy ("an
    # integer is required"); realize the arguments and defer to the real call.
    for name in ("major", "minor", "makedev"):
        fn = getattr(os, name, None)
        if fn is not None:
            register_patch(fn, with_realized_args(fn))
