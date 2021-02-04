from collections import defaultdict
import os
import sys
from typing import Callable, Tuple


class SideEffectDetected(Exception):
    pass


_BLOCKED_OPEN_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_TRUNC
)


def check_open(event, args) -> None:
    (filename, mode, flags) = args
    if flags & _BLOCKED_OPEN_FLAGS:
        raise SideEffectDetected(
            "A file open operation was detected. "
            "CrossaHair should not be run on code with side effects"
        )


def ignore(event, args) -> None:
    pass


def reject(event, args) -> None:
    raise SideEffectDetected(
        f'A "{event}" operation was detected. '
        f"CrossaHair should not be run on code with side effects"
    )


def make_handler(event: str) -> Callable[[str, Tuple], None]:
    lead = event.split(".", 1)[0]
    if event == "open":
        return check_open
    if lead in (
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
    return ignore


_HANDLERS = {}


def audithook(event, args):
    handler = _HANDLERS.get(event)
    if handler is None:
        handler = make_handler(event)
        _HANDLERS[event] = handler
    handler(event, args)


def engage_auditwall():
    sys.addaudithook(audithook)
