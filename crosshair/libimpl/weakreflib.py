from weakref import ref

from crosshair.core import register_patch


def _ref_call(r):
    if not isinstance(r, ref):
        raise TypeError
    return None


def make_registrations():
    register_patch(ref.__call__, _ref_call)
