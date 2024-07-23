from pathlib import PurePath

from crosshair.core import proxy_for_type
from crosshair.tracers import ResumedTracing


def test_PurePath___init__(space):
    pathstr = proxy_for_type(str, "pathstr")
    with ResumedTracing():
        PurePath(pathstr)
