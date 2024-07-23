import os

from crosshair.core import register_patch


def make_registrations():
    register_patch(os.fspath, os._fspath)
