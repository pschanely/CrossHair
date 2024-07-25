import zlib

from crosshair.core import proxy_for_type, realize
from crosshair.tracers import ResumedTracing


def test_compress_on_symbolic(space):
    buffer = proxy_for_type(bytes, "buffer")
    with ResumedTracing():
        space.add(len(buffer) == 1)
        result = zlib.compress(buffer)
    realized_buffer = realize(buffer)
    assert zlib.decompress(result) == realized_buffer
