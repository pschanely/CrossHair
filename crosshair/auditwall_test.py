import itertools
import os
import platform
import shlex
import subprocess
import sys
import urllib.request
from subprocess import call, run

import pytest

from crosshair.auditwall import (
    _SPECIAL_HANDLERS,
    SideEffectDetected,
    _decode_unblock_prefix,
    _encode_unblock_prefix,
    _make_prefix_based_handler,
    _quote_shell_argument,
    _split_windows_command,
    _update_special_handlers,
    accept,
    engage_auditwall,
    explain,
)

# audit hooks cannot be uninstalled, and we don't want to wall off the
# testing process. Spawn subprcoesses instead.

pyexec = sys.executable


def test_windows_command_roundtrip():
    # list2cmdline is pure Python, so this runs on every platform.
    pieces = ["a", " ", "\t", '"', "\\"]
    texts = [
        "".join(chars)
        for n in range(6)
        for chars in itertools.product(pieces, repeat=n)
    ]
    short_texts = [t for t in texts if len(t) <= 2]
    for text in texts:
        argv = ["program", text]
        assert _split_windows_command(subprocess.list2cmdline(argv)) == argv
    for first, second in itertools.product(short_texts, repeat=2):
        argv = ["program", first, second]
        assert _split_windows_command(subprocess.list2cmdline(argv)) == argv


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


@pytest.mark.parametrize("posix", [True, False])
@pytest.mark.parametrize(
    "argument",
    [
        "a:b",
        "C:\\x\\",
        "\\\\srv\\share",
        "a\\\\b",
        "path\\:name",
        "",
        "a'b $HOME",
        'print("hi")',
        'C:\\x\\"',
        "a&b",
    ],
)
def test_unblock_option_round_trip(argument, posix):
    event = "subprocess.Popen"
    tokens = ("python", "-c", "pass", argument)
    event_args = (None, list(tokens), None, None)
    option = _quote_shell_argument(
        f"--unblock={_encode_unblock_prefix(event, tokens)}", posix
    )
    split_option = shlex.split(option) if posix else _split_windows_command(option)
    assert len(split_option) == 1
    unblock_prefix = split_option[0].split("=", 1)[1]
    decoded_event, decoded_tokens = _decode_unblock_prefix(unblock_prefix)
    assert decoded_event == event

    handler = _make_prefix_based_handler((decoded_tokens,), on_match=accept)

    handler(event, event_args)
    wrong_args = (None, list(tokens[:-1] + (argument + "different",)), None, None)
    with pytest.raises(SideEffectDetected):
        handler(event, wrong_args)


def test_explain_includes_quoted_unblock_option():
    event = "subprocess.Popen"
    tokens = ("python", "-c", 'print("hi")')
    option = _quote_shell_argument(
        f"--unblock={_encode_unblock_prefix(event, tokens)}", os.name == "posix"
    )

    message = explain(event, (None, list(tokens), None, None))

    assert f"use {option}. (or some colon-delimited prefix)" in message


def test_explain_without_argument_prefix():
    message = explain("custom.event", ())

    assert "--unblock=custom.event" in message
    assert "colon-delimited prefix" not in message


def test_update_special_handlers_for_event_wide_prefix(monkeypatch):
    event = "custom.event"
    monkeypatch.setitem(_SPECIAL_HANDLERS, event, None)

    _update_special_handlers((event,), accept)

    assert _SPECIAL_HANDLERS[event] is accept


def test_update_special_handlers_for_argument_prefix(monkeypatch):
    event = "custom.event"
    args = ("a:b",)
    prefix = _encode_unblock_prefix(event, args)
    monkeypatch.setitem(_SPECIAL_HANDLERS, event, None)

    _update_special_handlers((prefix,), accept)

    handler = _SPECIAL_HANDLERS[event]
    handler(event, args)
    with pytest.raises(SideEffectDetected):
        handler(event, ("a", "b"))


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

engage_auditwall([
    event + ":" + sys.argv[1]
    for event in ("subprocess.Popen", "os.posix_spawn")
])
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
            "python:-c:pass:" + prefix_argument,
            argument,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == (0 if allowed else 10), result.stderr


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
