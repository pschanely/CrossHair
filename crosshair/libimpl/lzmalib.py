import lzma

from crosshair.core import register_patch, with_realized_args


def make_registrations():
    # The compressor/decompressor objects reject symbolic integer options.
    for fn in (lzma.compress, lzma.decompress):
        register_patch(fn, with_realized_args(fn))
