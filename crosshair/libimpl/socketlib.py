import socket

from crosshair.core import register_patch, with_realized_args

# Byte-order and interface-index helpers are C functions that unpack their
# integer argument with a strict format code, so they reject a symbolic proxy
# outright ("an integer is required" / "expected int").  Each is a deterministic
# function of its (realized) argument, so realize and defer to the real call.
_INT_ARG_FUNCS = (
    "htonl",
    "htons",
    "ntohl",
    "ntohs",
    "if_indextoname",
)


def make_registrations():
    for name in _INT_ARG_FUNCS:
        fn = getattr(socket, name, None)
        if fn is not None:
            register_patch(fn, with_realized_args(fn))
