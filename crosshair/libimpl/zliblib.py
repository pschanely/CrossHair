import zlib

from crosshair.core import register_patch, with_realized_args


def make_registrations():
    for fn in (
        zlib.adler32,
        zlib.compress,
        zlib.crc32,
        zlib.decompress,
        type(zlib.compressobj()).compress,
        type(zlib.decompressobj()).decompress,
    ):
        register_patch(fn, with_realized_args(fn))
