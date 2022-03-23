from contextlib import contextmanager
import os
import sys
from typing import Callable, Dict, Generator, Tuple


class SideEffectDetected(Exception):
    pass


_BLOCKED_OPEN_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_TRUNC
)


def check_open(event: str, args: Tuple) -> None:
    (filename_or_descriptor, mode, flags) = args
    if filename_or_descriptor == "/dev/null":
        return
    if flags & _BLOCKED_OPEN_FLAGS:
        raise SideEffectDetected(
            f'We\'ve blocked a file writing operation on "{filename_or_descriptor}". '
            f"CrossHair should not be run on code with side effects"
        )


def accept(event: str, args: Tuple) -> None:
    pass


def reject(event: str, args: Tuple) -> None:
    raise SideEffectDetected(
        f'A "{event}" operation was detected. '
        f"CrossHair should not be run on code with side effects"
    )


def make_handler(event: str) -> Callable[[str, Tuple], None]:
    # Allow file opening, for reads only.
    if event == "open":
        return check_open
    # Block certain events
    if event in (
        "winreg.CreateKey",
        "winreg.DeleteKey",
        "winreg.DeleteValue",
        "winreg.SaveKey",
        "winreg.SetValue",
        "winreg.DisableReflectionKey",
        "winreg.EnableReflectionKey",
    ):
        return reject
    # Allow certain events.
    if event in (
        # These seem not terribly dangerous to allow:
        "os.putenv",
        "os.unsetenv",
        # These involve I/O, but are hopefully non-destructive:
        "os.listdir",  # (important for Python's importer)
        "os.scandir",  # (important for Python's importer)
        "os.chdir",
        "os.fwalk",
        "os.getxattr",
        "glob.glob",
        "os.listxattr",
        "os.walk",
        "pathlib.Path.glob",
        "socket.gethostbyname",  # (FastAPI TestClient uses this)
        "socket.__new__",  # (FastAPI TestClient uses this)
    ):
        return accept
    # Block groups of events.
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
    ):
        return reject
    # Allow other events.
    return accept


_HANDLERS: Dict[str, Callable[[str, Tuple], None]] = {}
_ENABLED = True


def audithook(event: str, args: Tuple) -> None:
    if not _ENABLED:
        return
    handler = _HANDLERS.get(event)
    if handler is None:
        handler = make_handler(event)
        _HANDLERS[event] = handler
    handler(event, args)


@contextmanager
def opened_auditwall() -> Generator:
    global _ENABLED
    assert _ENABLED
    _ENABLED = False
    try:
        yield
    finally:
        _ENABLED = True


def engage_auditwall() -> None:
    sys.dont_write_bytecode = True  # disable .pyc file writing
    if sys.version_info >= (3, 8):  # audithook is new in 3.8
        sys.addaudithook(audithook)
