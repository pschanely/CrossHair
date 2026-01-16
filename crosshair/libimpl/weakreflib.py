from gc import collect
from weakref import ref

from crosshair.core import register_patch


def _ref_call(r):
    # Attempt to make weak references deterministic by aggressively clearing them
    if not isinstance(r, ref):
        raise TypeError
    collect()
    return r()


def make_registrations():
    register_patch(ref.__call__, _ref_call)
