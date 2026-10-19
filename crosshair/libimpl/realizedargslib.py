import importlib
import zlib
from typing import Callable, Dict, Iterator, Tuple

from crosshair.core import register_patch, with_realized_args

# Functions that reject symbolic arguments (typically C functions that parse
# their arguments strictly), keyed by module.  Each is patched to realize its
# arguments and defer to the real implementation.
_REALIZE_ARGS: Dict[str, Tuple[str, ...]] = {
    "hashlib": ("new", "scrypt"),
    "ipaddress": ("ip_network", "ip_interface"),
    "lzma": ("compress", "decompress"),
    "os": ("major", "minor", "makedev"),
    "socket": ("htonl", "htons", "ntohl", "ntohs", "if_indextoname"),
    "stat": (
        "S_IMODE",
        "S_IFMT",
        "S_ISDIR",
        "S_ISCHR",
        "S_ISBLK",
        "S_ISREG",
        "S_ISFIFO",
        "S_ISLNK",
        "S_ISSOCK",
        "S_ISDOOR",
        "S_ISPORT",
        "S_ISWHT",
        "filemode",
    ),
    "zlib": ("adler32", "compress", "crc32", "decompress"),
}

# Like _REALIZE_ARGS, but for functions that need their arguments deeply realized.
_DEEP_REALIZE_ARGS: Dict[str, Tuple[str, ...]] = {
    "pickle": ("dumps",),
}


def _functions(names_by_module: Dict[str, Tuple[str, ...]]) -> Iterator[Callable]:
    for module_name, names in names_by_module.items():
        module = importlib.import_module(module_name)
        for name in names:
            fn = getattr(module, name, None)
            if fn is not None:
                yield fn


def make_registrations() -> None:
    for fn in _functions(_REALIZE_ARGS):
        register_patch(fn, with_realized_args(fn))
    for method in (
        type(zlib.compressobj()).compress,
        type(zlib.decompressobj()).decompress,
    ):
        register_patch(method, with_realized_args(method))
    for fn in _functions(_DEEP_REALIZE_ARGS):
        register_patch(fn, with_realized_args(fn, deep=True))
