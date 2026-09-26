import os
from pathlib import Path
from sys import argv

import pytest

from crosshair.core_and_libs import NoTracing, standalone_statespace
from crosshair.util import mem_usage_kb, set_debug


def pytest_configure(config):
    if "-v" in argv or "-vv" in argv:
        set_debug(True)
    xdist_worker = os.environ.get("PYTEST_XDIST_WORKER")
    if xdist_worker:
        try:
            from hypothesis.configuration import set_hypothesis_home_dir
        except ImportError:
            pass
        else:
            # Hypothesis draws integers from constants in local source files, via
            # a cache that is not safe for concurrent writers; a partially written
            # cache file changes the inputs that parametrize tests at collection.
            set_hypothesis_home_dir(Path(".hypothesis") / f"xdist-{xdist_worker}")


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
