import functools as orig_functools

from crosshair.core import register_patch


def _reduce(function, *a, **kw):
    return orig_functools.reduce(lambda x, y: function(x, y), *a, **kw)


def make_registrations():
    register_patch(orig_functools.reduce, _reduce)
