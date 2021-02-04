from crosshair.libimpl import builtinslib
from crosshair.libimpl import collectionslib
from crosshair.libimpl import datetimelib
from crosshair.libimpl import mathlib
from crosshair.libimpl import randomlib
from crosshair.libimpl import relib


def make_registrations():
    builtinslib.make_registrations()
    collectionslib.make_registrations()
    datetimelib.make_registrations()
    mathlib.make_registrations()
    randomlib.make_registrations()
    relib.make_registrations()
