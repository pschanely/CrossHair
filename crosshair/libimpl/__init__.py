from crosshair.libimpl import collectionslib
from crosshair.libimpl import stdlib
from crosshair.libimpl import builtinslib

def make_registrations():
    builtinslib.make_registrations()
    collectionslib.make_registrations()
    stdlib.make_registrations()

