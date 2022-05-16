import os
import sys
import urllib.request
from subprocess import call

from crosshair.auditwall import SideEffectDetected, engage_auditwall

# audit hooks cannot be uninstalled, and we don't want to wall off the
# testing process. Spawn subprcoesses instead.

if sys.version_info >= (3, 8):  # audithook is new in 3.8

    def test_fs_read_allowed():
        assert call(["python", __file__, "read_open", "withwall"]) != 10

    def test_scandir_allowed():
        assert call(["python", __file__, "scandir", "withwall"]) == 0

    def test_import_allowed():
        assert call(["python", __file__, "import", "withwall"]) == 0

    def test_fs_write_disallowed():
        assert call(["python", __file__, "write_open", "withwall"]) == 10

    def test_http_disallowed():
        assert call(["python", __file__, "http", "withwall"]) == 10

    def test_unlink_disallowed():
        assert call(["python", __file__, "unlink", "withwall"]) == 10


_ACTIONS = {
    "read_open": lambda: open("/dev/null", "rb"),
    "scandir": lambda: os.scandir("."),
    "import": lambda: __import__("shutil"),
    "write_open": lambda: open("/.auditwall.testwrite.txt", "w"),
    "http": lambda: urllib.request.urlopen("http://localhost/foo"),
    "unlink": lambda: os.unlink("./delme.txt"),
}

if __name__ == "__main__":
    action, wall = sys.argv[1:]
    if wall == "withwall":
        engage_auditwall()

    try:
        _ACTIONS[action]()
    except SideEffectDetected as e:
        print(e)
        sys.exit(10)
