import os
import platform
import sys
import urllib.request
from subprocess import call, run

import pytest

from crosshair.auditwall import (
    SideEffectDetected,
    _split_windows_command,
    engage_auditwall,
)

# audit hooks cannot be uninstalled, and we don't want to wall off the
# testing process. Spawn subprcoesses instead.

pyexec = sys.executable


@pytest.mark.parametrize(
    "command,arguments",
    [
        ('"abc" d e', ["abc", "d", "e"]),
        (r'a\\\b d"e f"g h', [r"a\\\b", "de fg", "h"]),
        (r"a\\\"b c d", [r"a\"b", "c", "d"]),
        (r'a\\\\"b c" d e', [r"a\\b c", "d", "e"]),
        ('a"b"" c d', ['ab" c d']),
    ],
)
def test_windows_command_quoting_rules(command, arguments):
    assert _split_windows_command("program " + command) == ["program", *arguments]


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
    # --unblock arg-prefixes match against the canonicalized argv tokens
    # (program, *arguments) -- see crosshair.auditwall.match_tokens -- so these
    # match identically on POSIX and Windows despite the differing raw audit args.
    assert call([pyexec, __file__, "popen", "withwall", "subprocess.Popen"]) == 0
    assert call([pyexec, __file__, "popen", "withwall", "subprocess.Popen:echo"]) == 0
    assert (
        call([pyexec, __file__, "popen", "withwall", "subprocess.Popen:echo:hello"])
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
                "subprocess.Popen:echo:bye",
                "subprocess.Popen:echo:hello",
            ]
        )
        == 0
    )


def test_popen_disallowed_with_unrelated_prefix_allowences():
    assert call([pyexec, __file__, "popen", "withwall", "os:chdir"]) == 10
    assert call([pyexec, __file__, "popen", "withwall", "subprocess.Popen:date"]) == 10
    assert (
        call([pyexec, __file__, "popen", "withwall", "subprocess.Popen:echo:bye"]) == 10
    )


@pytest.mark.parametrize(
    "argument",
    [
        "plain",
        "two words",
        "",
        'a"b',
        'a "quoted" value',
        "two words\\",
        "folder\\file",
    ],
)
@pytest.mark.parametrize("allowed", [False, True])
def test_popen_argument_prefixes(argument, allowed):
    prefix_argument = argument if allowed else argument + "different"
    program = """
import subprocess
import sys
from crosshair.auditwall import SideEffectDetected, engage_auditwall

engage_auditwall([sys.argv[1]])
try:
    subprocess.run(
        ["python", "-c", "pass", sys.argv[2]],
        executable=sys.executable,
        check=True,
    )
except SideEffectDetected:
    sys.exit(10)
"""
    result = run(
        [
            pyexec,
            "-c",
            program,
            "subprocess.Popen:python:-c:pass:" + prefix_argument,
            argument,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == (0 if allowed else 10), result.stderr


def test_cli_unblock_quoted_argument(tmp_path):
    target = tmp_path / "quoted_argument.py"
    target.write_text(
        """
import subprocess
import sys

def launch() -> bool:
    '''post: _'''
    subprocess.run(
        ["python", "-c", "pass", 'a"b'],
        executable=sys.executable,
        check=True,
    )
    return True
""",
        encoding="utf-8",
    )
    result = run(
        [
            pyexec,
            "-m",
            "crosshair",
            "check",
            "--report_all",
            "--per_condition_timeout=5",
            "--analysis_kind=PEP316",
            '--unblock=subprocess.Popen:python:-c:pass:a"b',
            "--",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Confirmed over all paths" in result.stdout


def test_chdir_allowed():
    assert call([pyexec, __file__, "chdir", "withwall"]) == 0


def test_popen_via_platform_allowed():
    assert call([pyexec, __file__, "popen_via_platform", "withwall"]) == 0


_ACTIONS = {
    "read_open": lambda: open(os.devnull, "rb"),
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
