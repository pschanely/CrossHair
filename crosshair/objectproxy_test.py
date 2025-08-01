import pytest

from crosshair.objectproxy import ObjectProxy


class ObjectWrap(ObjectProxy):
    def __init__(self, obj):
        object.__setattr__(self, "_o", obj)

    def _realize(self):
        return object.__getattribute__(self, "_o")


class TestObjectProxy:
    def test_object_proxy_over_list(self) -> None:
        i = [1, 2, 3]
        proxy = ObjectWrap(i)
        assert i == proxy
        proxy.append(4)
        assert [1, 2, 3, 4] == proxy
        assert [1, 2, 3, 4, 5] == proxy + [5]
        assert [2, 3] == proxy[1:3]
        assert [1, 2, 3, 4] == proxy

    def test_inplace_identities(self) -> None:
        proxy = ObjectWrap(3.0)
        orig_proxy = proxy
        proxy += 1.0
        assert proxy is not orig_proxy
        proxy = ObjectWrap([1, 2])
        orig_proxy = proxy
        proxy += [3, 4]
        assert proxy is orig_proxy

    def test_object_proxy_over_float(self) -> None:
        proxy = ObjectWrap(4.5)
        proxy //= 2.0
        assert 2.0 == proxy
        proxy = ObjectWrap(5.0)
        proxy /= 2.0
        assert 2.5 == proxy
