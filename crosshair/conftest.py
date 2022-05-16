from crosshair.util import set_debug
from sys import argv

import pytest

from crosshair.core_and_libs import standalone_statespace
from crosshair.core_and_libs import NoTracing


def pytest_configure(config):
    if "-v" in argv or "-vv" in argv:
        set_debug(True)


@pytest.fixture()
def space():
    with standalone_statespace as spc, NoTracing():
        yield spc
