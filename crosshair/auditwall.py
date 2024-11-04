import importlib
import os
import sys
import traceback
from contextlib import contextmanager
from types import ModuleType
from typing import Callable, Dict, Generator, Iterable, Optional, Set, Tuple


class SideEffectDetected(Exception):
    pass


_BLOCKED_OPEN_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_TRUNC
)


def accept(event: str, args: Tuple) -> None:
    pass


def reject(event: str, args: Tuple) -> None:
    raise SideEffectDetected(
        f'A "{event}{args}" operation was detected. '
        f"CrossHair should not be run on code with side effects"
    )


def inside_module(modules: Iterable[ModuleType]) -> bool:
    files = {m.__file__ for m in modules}
    for frame, lineno in traceback.walk_stack(None):
        if frame.f_code.co_filename in files:
            return True
    return False


def check_open(event: str, args: Tuple) -> None:
    (filename_or_descriptor, mode, flags) = args
    if filename_or_descriptor in ("/dev/null", "nul"):
        # (no-op writes on unix/windows)
        return
    if flags & _BLOCKED_OPEN_FLAGS:
        raise SideEffectDetected(
            f'We\'ve blocked a file writing operation on "{filename_or_descriptor}". '
            f"CrossHair should not be run on code with side effects"
        )


def check_msvcrt_open(event: str, args: Tuple) -> None:
    print(args)
    (handle, flags) = args
    if flags & _BLOCKED_OPEN_FLAGS:
        raise SideEffectDetected(
            f'We\'ve blocked a file writing operation on "{handle}". '
            f"CrossHair should not be run on code with side effects"
        )


_MODULES_THAT_CAN_POPEN: Optional[Set[ModuleType]] = None


def modules_with_allowed_popen():
    global _MODULES_THAT_CAN_POPEN
    if _MODULES_THAT_CAN_POPEN is None:
        allowed_module_names = ("_aix_support", "ctypes", "platform", "uuid")
        _MODULES_THAT_CAN_POPEN = set()
        for module_name in allowed_module_names:
            try:
                _MODULES_THAT_CAN_POPEN.add(importlib.import_module(module_name))
            except ImportError:
                pass
    return _MODULES_THAT_CAN_POPEN


def check_subprocess(event: str, args: Tuple) -> None:
    if not inside_module(modules_with_allowed_popen()):
        reject(event, args)


_SPECIAL_HANDLERS = {
    "open": check_open,
    "subprocess.Popen": check_subprocess,
    "msvcrt.open_osfhandle": check_msvcrt_open,
}


def make_handler(event: str) -> Callable[[str, Tuple], None]:
    special_handler = _SPECIAL_HANDLERS.get(event, None)
    if special_handler:
        return special_handler
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
        "msvcrt.heapmin",
        "msvcrt.kbhit",
        # These involve I/O, but are hopefully non-destructive:
        "glob.glob",
        "msvcrt.get_osfhandle",
        "msvcrt.setmode",
        "os.listdir",  # (important for Python's importer)
        "os.scandir",  # (important for Python's importer)
        "os.chdir",
        "os.fwalk",
        "os.getxattr",
        "os.listxattr",
        "os.walk",
        "pathlib.Path.glob",
        "socket.gethostbyname",  # (FastAPI TestClient uses this)
        "socket.__new__",  # (FastAPI TestClient uses this)
        "socket.bind",  # pygls's asyncio needs this on windows
        "socket.connect",  # pygls's asyncio needs this on windows
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
    sys.addaudithook(audithook)


def disable_auditwall() -> None:
    global _ENABLED
    assert _ENABLED
    _ENABLED = False
