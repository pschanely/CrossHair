import stat

from crosshair.core import register_patch, with_realized_args

# The stat helpers are C functions that unpack their argument with the "i"
# format code, so they reject a symbolic proxy outright ("an integer is
# required").  Realize the mode and defer to the real implementation.
_MODE_FUNCS = (
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
)


def make_registrations():
    for name in _MODE_FUNCS:
        fn = getattr(stat, name, None)
        if fn is not None:
            register_patch(fn, with_realized_args(fn))
