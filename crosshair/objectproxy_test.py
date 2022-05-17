import unittest

from crosshair.objectproxy import ObjectProxy


class ObjectWrap(ObjectProxy):
    def __init__(self, obj):
        object.__setattr__(self, "_o", obj)

    def _wrapped(self):
        return object.__getattribute__(self, "_o")


class ObjectProxyTest(unittest.TestCase):
    def test_object_proxy(self) -> None:
        i = [1, 2, 3]
        proxy = ObjectWrap(i)
        self.assertEqual(i, proxy)
        proxy.append(4)
        self.assertEqual([1, 2, 3, 4], proxy)
        self.assertEqual([1, 2, 3, 4, 5], proxy + [5])
        self.assertEqual([2, 3], proxy[1:3])
        self.assertEqual([1, 2, 3, 4], proxy)


if __name__ == "__main__":
    unittest.main()
