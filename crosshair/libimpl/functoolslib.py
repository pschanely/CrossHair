import functools as orig_functools

from crosshair.core import register_patch

# TODO: deal with lru_cache (note it needs to be intercepted at import-time)


def _partial(func, *a1, **kw1):
    if callable(func):
        return orig_functools.partial(lambda *a2, **kw2: func(*a2, **kw2), *a1, **kw1)
    else:
        raise TypeError


def _reduce(function, *a, **kw):
    return orig_functools.reduce(lambda x, y: function(x, y), *a, **kw)


def make_registrations():
    register_patch(orig_functools.partial, _partial)
    register_patch(orig_functools.reduce, _reduce)
