from functools import _lru_cache_wrapper, partial, reduce

from crosshair.core import register_patch

# TODO: deal with lru_cache (note it needs to be intercepted at import-time)


def _partial(func, *a1, **kw1):
    if callable(func):
        return partial(lambda *a2, **kw2: func(*a2, **kw2), *a1, **kw1)
    else:
        raise TypeError


def _reduce(function, *a, **kw):
    return reduce(lambda x, y: function(x, y), *a, **kw)


def make_registrations():
    register_patch(partial, _partial)
    register_patch(reduce, _reduce)

    def call_with_skipped_cache(self, *a, **kw):
        if not isinstance(self, _lru_cache_wrapper):
            raise TypeError
        return self.__wrapped__(*a, **kw)

    register_patch(_lru_cache_wrapper.__call__, call_with_skipped_cache)
