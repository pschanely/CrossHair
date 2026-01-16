from weakref import WeakKeyDictionary, WeakSet, WeakValueDictionary, proxy, ref

import pytest

from crosshair.tracers import ResumedTracing


class Thing:
    x: int = 0


def test_weakref(space):
    thing1 = Thing()
    thingref = ref(thing1)
    assert thingref() is thing1
    with ResumedTracing():
        assert thingref() is not None
        del thing1
        assert thingref() is None


def test_weakref_WeakKeyDictionary(space):
    d = WeakKeyDictionary()
    thing1 = Thing()
    thing2 = Thing()
    d[thing1] = 1
    d[thing2] = 2
    assert len(d) == 2
    assert thing1 in d
    with ResumedTracing():
        assert set(d.keys()) == {thing1, thing2}
        del thing1
        assert set(d.keys()) == {thing2}


def test_weakref_WeakValueDictionary(space):
    d = WeakValueDictionary()
    thing1 = Thing()
    thing2 = Thing()
    d[1] = thing1
    d[2] = thing2
    assert len(d) == 2
    assert 1 in d
    with ResumedTracing():
        assert set(d.keys()) == {1, 2}
        del thing1
        assert 2 in d
        assert 1 not in d
        # You would expect the length to update too.
        # However, it doesn't require getting the referred object, so it appears to still exist:
        # assert len(d) == 1


def test_weakref_WeakSet(space):
    s = WeakSet()
    thing1 = Thing()
    s.add(thing1)
    with ResumedTracing():
        assert len(s) == 1
        assert thing1 in s
        del thing1
        assert len(s) == 0


def test_weakref_proxy(space):
    thing1 = Thing()
    thing1.x
    p = proxy(thing1)
    with ResumedTracing():
        p.x
        del thing1
        with pytest.raises(ReferenceError):
            p.x
