import os
import platform
import sys
import urllib.request
from subprocess import call

import pytest

from crosshair.auditwall import SideEffectDetected, engage_auditwall

# audit hooks cannot be uninstalled, and we don't want to wall off the
# testing process. Spawn subprcoesses instead.

pyexec = sys.executable


def test_fs_read_allowed():
    assert call([pyexec, __file__, "read_open", "withwall"]) != 10


def test_scandir_allowed():
    assert call([pyexec, __file__, "scandir", "withwall"]) == 0


def test_import_allowed():
    assert call([pyexec, __file__, "import", "withwall"]) == 0


def test_fs_write_disallowed():
    assert call([pyexec, __file__, "write_open", "withwall"]) == 10


def test_fs_write_allowed_if_prefix_allowed():
    try:
        assert (
            call(
                [
                    pyexec,
                    __file__,
                    "write_open",
                    "withwall",
                    "open:./auditwall.testwrite.txt:w",
                ]
            )
            == 0
        )
        # Confirm the new handler doesn't interfere with the prior handler:
        assert (
            call(
                [
                    pyexec,
                    __file__,
                    "read_open",
                    "withwall",
                    "open:./auditwall.testwrite.txt:w",
                ]
            )
            == 0
        )
    finally:
        try:
            os.unlink("./auditwall.testwrite.txt")
        except FileNotFoundError:
            pass


def test_http_disallowed():
    assert call([pyexec, __file__, "http", "withwall"]) == 10


def test_unlink_disallowed():
    assert call([pyexec, __file__, "unlink", "withwall"]) == 10


def test_popen_disallowed():
    assert call([pyexec, __file__, "popen", "withwall"]) == 10


def test_popen_allowed_if_prefix_allowed():
    assert call([pyexec, __file__, "popen", "withwall", "subprocess.Popen"]) == 0
    assert call([pyexec, __file__, "popen", "withwall", "subprocess.Popen:echo"]) == 0
    assert (
        call(
            [
                pyexec,
                __file__,
                "popen",
                "withwall",
                "subprocess.Popen:echo:['echo', 'hello']",
            ]
        )
        == 0
    )
    assert (
        call(
            [
                pyexec,
                __file__,
                "popen",
                "withwall",
                "os:chdir",
                "subprocess.Popen:ech",
                "subprocess.Popen:echo:['echo', 'bye']",
                "subprocess.Popen:echo:['echo', 'hello']",
            ]
        )
        == 0
    )


def test_popen_disallowed_with_unrelated_prefix_allowences():
    assert call([pyexec, __file__, "popen", "withwall", "os:chdir"]) == 10
    assert call([pyexec, __file__, "popen", "withwall", "subprocess.Popen:date"]) == 10
    assert (
        call(
            [
                pyexec,
                __file__,
                "popen",
                "withwall",
                "subprocess.Popen:echo:['echo', 'bye']",
            ]
        )
        == 10
    )


def test_chdir_allowed():
    assert call([pyexec, __file__, "chdir", "withwall"]) == 0


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Python 3.9+ required")
def test_popen_via_platform_allowed():
    assert call([pyexec, __file__, "popen_via_platform", "withwall"]) == 0


_ACTIONS = {
    "read_open": lambda: open("/dev/null", "rb"),
    "scandir": lambda: os.scandir("."),
    "import": lambda: __import__("shutil"),
    "write_open": lambda: open("./auditwall.testwrite.txt", "w"),
    "http": lambda: urllib.request.urlopen("http://localhost/foo"),
    "unlink": lambda: os.unlink("./delme.txt"),
    "popen": lambda: call(["echo", "hello"]),
    "popen_via_platform": lambda: platform._syscmd_ver(  # type: ignore
        supported_platforms=(sys.platform,)
    ),
    "chdir": lambda: os.chdir("."),
}

if __name__ == "__main__":
    action, wall, *allow_prefixes = sys.argv[1:]
    if wall == "withwall":
        engage_auditwall(allow_prefixes)

    try:
        _ACTIONS[action]()
    except SideEffectDetected as e:
        print(e)
        sys.exit(10)
