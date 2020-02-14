from crosshair.libimpl import builtinslib
from crosshair.libimpl import collectionslib
from crosshair.libimpl import datetimelib

def make_registrations():
    builtinslib.make_registrations()
    collectionslib.make_registrations()
    datetimelib.make_registrations()

