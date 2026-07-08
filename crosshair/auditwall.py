import importlib
import inspect
import itertools
import os
import sys
import traceback
from contextlib import contextmanager
from types import ModuleType
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
)


class SideEffectDetected(Exception):
    pass


_BLOCKED_OPEN_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_TRUNC
)


def accept(event: str, args: Tuple) -> None:
    pass


def explain(event: str, args: Tuple) -> str:
    argstr = "".join(f":{arg}" for arg in args)
    parts = [
        f"It's dangerous to run CrossHair on code with side effects.",
        f'To allow this operation anyway, use "--unblock={event}{argstr}".',
    ]
    if args:
        parts.append("(or some colon-delimited prefix)")
    return " ".join(parts)


def reject(event: str, args: Tuple) -> NoReturn:
    raise SideEffectDetected(
        f'A "{event}" operation was detected. ' + explain(event, args)
    )


def inside_module(modules: Iterable[ModuleType]) -> bool:
    """Checks whether the current call stack is inside one of the given modules."""
    for frame, _lineno in traceback.walk_stack(None):
        frame_module = inspect.getmodule(frame)
        if frame_module and frame_module in modules:
            return True
    return False


def check_open(event: str, args: Tuple) -> None:
    filename_or_descriptor, mode, flags = args
    if filename_or_descriptor in ("/dev/null", "nul"):
        # (no-op writes on unix/windows)
        return
    if flags & _BLOCKED_OPEN_FLAGS:
        raise SideEffectDetected(
            f'We\'ve blocked a file writing operation on "{filename_or_descriptor}". '
            + explain(event, args)
        )


def check_msvcrt_open(event: str, args: Tuple) -> None:
    handle, flags = args
    if flags & _BLOCKED_OPEN_FLAGS:
        raise SideEffectDetected(
            f'We\'ve blocked a file writing operation on "{handle}". '
            + explain(event, args)
        )


_MODULES_THAT_CAN_POPEN: Optional[Set[ModuleType]] = None


def modules_with_allowed_subprocess():
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
    if not inside_module(modules_with_allowed_subprocess()):
        reject(event, args)


_SPECIAL_HANDLERS: Dict[str, Callable[[str, Tuple], None]] = {
    "open": check_open,
    "subprocess.Popen": check_subprocess,
    "os.posix_spawn": check_subprocess,
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
_HOOK_INSTALLED = False


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


@contextmanager
def enabled_auditwall(
    allow_prefixes: Sequence[str] = (),
    reject_prefixes: Sequence[str] = (),
) -> Generator:
    """Enable the auditwall just for the duration of this context.

    Installs the audit hook (with default, maximally-restrictive config) if it
    isn't engaged yet, and restores the prior active state on exit -- so a wall
    that was dormant (never engaged, or opened) stays dormant afterward, and one
    already engaged (possibly with custom ``--unblock`` prefixes) keeps its config
    untouched. Use this to police a short, self-contained region for side effects
    (e.g. probing whether an operation reaches for I/O), without leaving the wall
    active for surrounding work.

    ``allow_prefixes`` / ``reject_prefixes`` layer extra event rules on top of the
    default handlers just for this scope: an ``allow`` entry accepts an event (the
    ``--unblock`` form -- ``"event"`` or ``"event:arg:prefix"``), a ``reject`` entry
    blocks one.  This is how a caller tightens the wall temporarily -- e.g. the
    operation-classification probe rejects ``os.chdir`` / ``glob.glob`` / ``socket``
    events the default allows for analysis convenience.  We snapshot the special-
    handler table and the resolved-handler cache and restore them on exit, so any
    prior configuration (including startup ``--unblock`` prefixes) comes back
    exactly -- which is why this can safely take prefixes a temporary scope owns."""
    global _ENABLED, _SPECIAL_HANDLERS, _HANDLERS
    was_active = _HOOK_INSTALLED and _ENABLED
    saved_special = dict(_SPECIAL_HANDLERS)
    saved_cache = dict(_HANDLERS)
    if not _HOOK_INSTALLED:
        engage_auditwall()
    if allow_prefixes:
        _update_special_handlers(allow_prefixes, accept)
    if reject_prefixes:
        _update_special_handlers(reject_prefixes, reject)
    if allow_prefixes or reject_prefixes:
        _HANDLERS.clear()  # resolved-handler cache is now stale
    _ENABLED = True
    try:
        yield
    finally:
        _ENABLED = was_active
        _SPECIAL_HANDLERS.clear()
        _SPECIAL_HANDLERS.update(saved_special)
        _HANDLERS.clear()
        _HANDLERS.update(saved_cache)


def _make_prefix_based_handler(
    arg_prefixes: Sequence[Sequence[str]],
    previous_handler: Optional[Callable[[str, Tuple], None]] = None,
    on_match: Callable[[str, Tuple], None] = accept,
) -> Callable[[str, Tuple], None]:
    # On a matching arg-prefix, apply ``on_match`` (accept for an allow rule, reject
    # for a reject rule).  Otherwise chain to the prior handler, or -- with none --
    # fall back to the OPPOSITE of on_match (an allow rule rejects everything else;
    # a reject rule leaves everything else alone).
    on_miss = previous_handler or (reject if on_match is accept else accept)
    trie: Dict = {}
    for prefix in arg_prefixes:
        current = trie
        for part in prefix:
            current = current.setdefault(part, {})
        current[None] = True  # Mark the end of a valid prefix

    def handler(event: str, args: Tuple) -> None:
        current = trie
        for arg in map(str, args):
            if arg not in current:
                break
            current = current[arg]
            if None in current:
                return on_match(event, args)  # matched a listed prefix
        return on_miss(event, args)

    return handler


def _update_special_handlers(
    prefixes: Sequence[str], terminal: Callable[[str, Tuple], None]
) -> None:
    """Install per-event rules from ``--unblock``-style entries: a bare ``"event"``
    applies ``terminal`` (``accept`` / ``reject``) to the whole event, while
    ``"event:arg:prefix"`` applies it only to matching args."""
    for event, group_itr in itertools.groupby(
        sorted(prefixes), lambda p: p.split(":", 1)[0]
    ):
        group = tuple(group_itr)
        if any(event == g for g in group):
            _SPECIAL_HANDLERS[event] = terminal
        else:
            args = tuple(a.split(":")[1:] for a in group)
            _SPECIAL_HANDLERS[event] = _make_prefix_based_handler(
                args, _SPECIAL_HANDLERS.get(event), terminal
            )


def engage_auditwall(allow_prefixes: Sequence[str] = ()) -> None:
    global _HOOK_INSTALLED
    if "EVERYTHING" in allow_prefixes:
        return
    _update_special_handlers(allow_prefixes, accept)
    sys.dont_write_bytecode = True  # disable .pyc file writing
    sys.addaudithook(audithook)
    _HOOK_INSTALLED = True


def disable_auditwall() -> None:
    global _ENABLED
    assert _ENABLED
    _ENABLED = False
