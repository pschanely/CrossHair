import hashlib

from crosshair.core import proxy_for_type, standalone_statespace


def test_sha384():
    with standalone_statespace:
        x = proxy_for_type(bytes, "x")
        hashlib.new("sha384", x)
        hashlib.sha384(x)


def test_blake_via_update():
    with standalone_statespace:
        x = proxy_for_type(bytes, "x")
        h = hashlib.blake2b()
        h.update(x)
        h.hexdigest()
