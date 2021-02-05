from collections import defaultdict
from contextlib import contextmanager
import os
import sys
from typing import Callable, Tuple


class SideEffectDetected(Exception):
    pass


_BLOCKED_OPEN_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_TRUNC
)


def check_open(event, args) -> None:
    (filename_or_descriptor, mode, flags) = args
    if flags & _BLOCKED_OPEN_FLAGS:
        raise SideEffectDetected(
            f'We\'ve blocked a file writing operation on "{filename_or_descriptor}". '
            f"CrossaHair should not be run on code with side effects"
        )


def accept(event, args) -> None:
    pass


def reject(event, args) -> None:
    raise SideEffectDetected(
        f'A "{event}" operation was detected. '
        f"CrossaHair should not be run on code with side effects"
    )


def make_handler(event: str) -> Callable[[str, Tuple], None]:
    # Allow file opening, for reads only.
    if event == "open":
        return check_open
    # Explicitly allow certain events.
    if event in (
        # These seem important for the operation of Python:
        "os.listdir",
        "os.scandir",
        # These seem not terribly dangerous to allow:
        "os.chdir",
        "os.fwalk",
        "os.getxattr",
        "os.listxattr",
        "os.putenv",
        "os.unsetenv",
    ):
        return accept
    # Blocklist groups of events.
    event_prefix = event.split(".", 1)[0]
    if event_prefix in (
        "os",
        "fcntl",
        "ftplib",
        "glob",
        "imaplib",
        "msvcrt",
        "nntplib",
        "os",
        "pathlib",
        "poplib",
        "shutil",
        "smtplib",
        "socket",
        "sqlite3",
        "subprocess",
        "telnetlib",
        "urllib",
        "webbrowser",
        "winreg",
    ):
        return reject
    # Allow other events.
    return accept


_HANDLERS = {}
_ENABLED = True


def audithook(event, args):
    if not _ENABLED:
        return
    handler = _HANDLERS.get(event)
    if handler is None:
        handler = make_handler(event)
        _HANDLERS[event] = handler
    handler(event, args)


@contextmanager
def opened_auditwall():
    global _ENABLED
    assert _ENABLED
    _ENABLED = False
    yield
    _ENABLED = True


def engage_auditwall():
    sys.dont_write_bytecode = True  # disable .pyc file writing
    if sys.version_info >= (3, 8):  # audithook is new in 3.8
        sys.addaudithook(audithook)
