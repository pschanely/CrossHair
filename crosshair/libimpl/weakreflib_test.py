from weakref import WeakKeyDictionary, WeakSet, WeakValueDictionary, proxy, ref

import pytest

from crosshair.tracers import ResumedTracing


class Thing:
    x: int = 0


def test_weakref(space):
    thing1 = Thing()
    assert ref(thing1)() is thing1
    with ResumedTracing():
        assert ref(thing1)() is None


def test_weakref_WeakKeyDictionary(space):
    d = WeakKeyDictionary()
    thing1 = Thing()
    thing2 = Thing()
    d[thing1] = 1
    d[thing2] = 2
    assert len(d) == 2
    assert thing1 in d
    assert set(d.keys()) == {thing1, thing2}
    with ResumedTracing():
        assert set(d.keys()) == set()
        # You would expect the following assertions to work too
        # However, they don't require getting the referred object, so they appear to still exist:
        # assert thing1 not in d
        # assert len(d) == 0


def test_weakref_WeakValueDictionary(space):
    d = WeakValueDictionary()
    thing1 = Thing()
    thing2 = Thing()
    d[1] = thing1
    d[2] = thing2
    assert len(d) == 2
    assert 1 in d
    assert set(d.keys()) == {1, 2}
    with ResumedTracing():
        assert set(d.keys()) == set()
        assert 1 not in d
        # You would expect the length to update too.
        # However, it doesn't require getting the referred object, so it appears to still exist:
        # assert len(d) == 0


@pytest.mark.xfail(reason="weakref.WeakSet is not yet supported")
def test_weakref_WeakSet(space):
    s = WeakSet()
    thing1 = Thing()
    s.add(thing1)
    assert thing1 in s
    with ResumedTracing():
        assert thing1 not in s


@pytest.mark.xfail(reason="weakref.proxy is not yet supported")
def test_weakref_proxy(space):
    thing1 = Thing()
    thing1.x
    p = proxy(thing1)
    with pytest.raises(ReferenceError), ResumedTracing():
        p.x
