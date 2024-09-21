from sys import argv

import pytest

from crosshair.core_and_libs import NoTracing, standalone_statespace
from crosshair.util import mem_usage_kb, set_debug


def pytest_configure(config):
    if "-v" in argv or "-vv" in argv:
        set_debug(True)


LEAK_LIMIT_KB = 400 * 1024


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    last_ram = mem_usage_kb()
    outcome = yield
    growth = mem_usage_kb() - last_ram
    assert (
        growth < LEAK_LIMIT_KB
    ), f"Leaking memory (grew {growth // 1024}M while running)"


@pytest.fixture()
def space():
    with standalone_statespace as spc, NoTracing():
        yield spc
