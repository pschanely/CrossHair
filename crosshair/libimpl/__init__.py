from crosshair.libimpl.stdlib import make_stdlib_registrations
from crosshair.libimpl.builtinslib import make_builtin_registrations

def make_registrations():
    make_builtin_registrations()
    make_stdlib_registrations()
