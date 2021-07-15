from pathlib import Path

from crosshair.options import DEFAULT_OPTIONS
from crosshair.watcher import Watcher


def simplefs(path: Path, files: dict) -> None:
    for name, contents in files.items():
        subpath = path / name
        if isinstance(contents, str):
            with open(subpath, "w") as fh:
                fh.write(contents)
        elif isinstance(contents, dict):
            subpath.mkdir()
            simplefs(subpath, contents)
        else:
            raise Exception("bad input to simplefs")


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


def test_modified_file(tmp_path: Path):
    simplefs(tmp_path, CORRECT_FOO)
    watcher = Watcher([tmp_path])
    assert watcher.check_changed()
    assert not watcher.check_changed()
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
