import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from crosshair.statespace import MessageType
from crosshair.test_util import simplefs
from crosshair.watcher import Watcher

# TODO: DRY a bit with main_test.py


@pytest.fixture(autouse=True)
def rewind_modules():
    defined_modules = list(sys.modules.keys())
    yield None
    for name, module in list(sys.modules.items()):
        # Some standard library modules aren't happy with getting reloaded.
        if name.startswith("multiprocessing"):
            continue
        if name not in defined_modules:
            del sys.modules[name]


BUGGY_FOO = {
    "foo.py": """
def foofn(x: int) -> int:
  ''' post: _ == x '''
  return x + 1
"""
}

CORRECT_FOO = {
    "foo.py": """
def foofn(x: int) -> int:
  ''' post: _ == 1 + x '''
  return x + 1
"""
}

BAD_SYNTAX_FOO = {
    "foo.py": """
def foofn(x: int) -> int:
  ''' post: _ == x '''
  return $ x + 1
"""
}

EMPTY_BAR = {
    "bar.py": """
# Nothing here
"""
}


def test_added_file(tmp_path: Path):
    simplefs(tmp_path, CORRECT_FOO)
    watcher = Watcher([tmp_path])
    assert watcher.check_changed()
    assert not watcher.check_changed()
    simplefs(tmp_path, EMPTY_BAR)
    assert watcher.check_changed()


def test_added_file_to_empty_dir(tmp_path: Path):
    watcher = Watcher([tmp_path])
    assert list(watcher.run_iteration()) == []
    simplefs(tmp_path, BUGGY_FOO)  # Add new file.
    watcher._next_file_check = time.time() - 1
    # Detect file (yields empty result to wake up the loop)
    assert list(watcher.run_iteration()) == [(Counter(), [])]
    # Find bug in new file after restart.
    (result,) = list(watcher.run_iteration())
    assert [m.state for m in result[1]] == [MessageType.POST_FAIL]


def test_modified_file(tmp_path: Path):
    simplefs(tmp_path, CORRECT_FOO)
    watcher = Watcher([tmp_path])
    assert watcher.check_changed()
    assert not watcher.check_changed()
    time.sleep(0.01)  # Ensure mtime is actually different!
    simplefs(tmp_path, BUGGY_FOO)
    assert watcher.check_changed()
    assert not watcher.check_changed()


def test_removed_file(tmp_path: Path):
    simplefs(tmp_path, CORRECT_FOO)
    simplefs(tmp_path, EMPTY_BAR)
    watcher = Watcher([tmp_path])
    assert watcher.check_changed()
    assert not watcher.check_changed()
    (tmp_path / "bar.py").unlink()
    assert watcher.check_changed()


def test_removed_file_given_as_argument(tmp_path: Path):
    simplefs(tmp_path, CORRECT_FOO)
    watcher = Watcher([tmp_path / "foo.py"])
    assert watcher.check_changed()
    (tmp_path / "foo.py").unlink()
    assert watcher.check_changed()
