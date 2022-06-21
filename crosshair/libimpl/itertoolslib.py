import itertools as real_itertools
import operator

from crosshair import register_patch


def identity(x):
    return x


def _accumulate(iterable, func=operator.add, **kw):
    return real_itertools.accumulate(iterable, lambda a, b: func(a, b), **kw)


def _dropwhile(predicate, iterable):
    return real_itertools.dropwhile(lambda x: predicate(x), iterable)


def _filterfalse(predicate, iterable):
    return real_itertools.filterfalse(lambda x: predicate(x), iterable)


def _groupby(iterable, key=identity):
    if key is identity:
        return real_itertools.groupby(iterable)
    else:
        return real_itertools.groupby(iterable, lambda x: key(x))


def _starmap(function, iterable):
    return real_itertools.starmap(lambda *a: function(*a), iterable)


def _takewhile(predicate, iterable):
    return real_itertools.takewhile(lambda x: predicate(x), iterable)


def make_registrations():
    register_patch(real_itertools.accumulate, _accumulate)
    register_patch(real_itertools.dropwhile, _dropwhile)
    register_patch(real_itertools.filterfalse, _filterfalse)
    register_patch(real_itertools.groupby, _groupby)
    register_patch(real_itertools.starmap, _starmap)
    register_patch(real_itertools.takewhile, _takewhile)
