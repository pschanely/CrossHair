import pytest

from crosshair.objectproxy import ObjectProxy


class ObjectWrap(ObjectProxy):
    def __init__(self, obj):
        object.__setattr__(self, "_o", obj)

    def _wrapped(self):
        return object.__getattribute__(self, "_o")


class TestObjectProxy:
    def test_object_proxy(self) -> None:
        i = [1, 2, 3]
        proxy = ObjectWrap(i)
        assert i == proxy
        proxy.append(4)
        assert [1, 2, 3, 4] == proxy
        assert [1, 2, 3, 4, 5] == proxy + [5]
        assert [2, 3] == proxy[1:3]
        assert [1, 2, 3, 4] == proxy
