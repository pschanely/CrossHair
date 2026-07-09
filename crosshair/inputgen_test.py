"""Tests for crosshair.inputgen -- the operation catalog and input generation."""

import multiprocessing
import subprocess
import sys
import textwrap

import pytest

from crosshair.inputgen import (
    CATALOG_FUNC_MODULES,
    CATALOG_METHOD_MODULES,
    documented_stdlib_modules,
)


def test_catalog_surface_is_documented_only():
    """The exclusion-model surface enumerates DOCUMENTED stdlib modules only.  Two
    guards on the derivation: every enumerated module's top-level name is documented
    (catches a denylist/extra-module slip), and the undocumented internals that used
    to leak onto hand-maintained inclusion lists stay OFF -- they have no stable
    public contract to differentially test against."""
    documented = documented_stdlib_modules()
    surface = set(CATALOG_FUNC_MODULES) | set(CATALOG_METHOD_MODULES)
    tops = {m.split(".")[0] for m in surface}
    undocumented = sorted(t for t in tops if t not in documented)
    assert not undocumented, f"undocumented modules on the surface: {undocumented!r}"
    for internal in (
        "sre_parse",
        "sre_compile",
        "opcode",
        "nturl2path",
        "genericpath",
        "posixpath",
        "ntpath",
        "antigravity",
        "this",
    ):
        assert internal not in tops, f"{internal} is undocumented; keep it off surface"
    for public in ("math", "json", "stat", "os"):  # sanity: documented ones present
        assert public in tops, f"expected documented module {public} on the surface"


# The sweep runs in a CLEAN subprocess, not in-process: the concrete probe forks
# per op, and forking pytest's multi-threaded process risks deadlocking the child
# (and pytest's own breakpoint/capture hooks perturb ops like builtins.breakpoint).
# A fresh single-threaded interpreter is the environment the probe is built for.
_SWEEP = textwrap.dedent("""
    import sys
    from crosshair.inputgen import CRASH, HANG, catalog, probe_side_effect_isolated
    bad = []
    for op in catalog(probe=False):
        # Skip what the sweep never runs forward: undrivable, no synthesizable
        # inputs, or already excluded as out-of-scope / a side effect / a probe
        # hazard.  not_value_function ops ARE run (measured, black-suppressed), so
        # they stay in.
        if (op.call is None or op.out_of_scope or op.no_inputs
                or op.side_effect or op.probe_hazard):
            continue
        reason = probe_side_effect_isolated(op.call, seedkey=None, timeout=2.0)
        if reason in (HANG, CRASH):
            # A HANG/CRASH can be a fork-latency flake under the big-surface load;
            # re-probe once with a generous timeout before believing it.
            reason = probe_side_effect_isolated(op.call, seedkey=None, timeout=8.0)
        if reason is not None:
            bad.append(f"{op.key}: {reason}")
    sys.stdout.write("\\n".join(bad))
    """)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="probe_side_effect_isolated relies on a fork context (absent on Windows). "
    "The Windows-only op surface (msvcrt/winreg/winsound) still needs this guard via "
    "a fork-free probe -- deferred with the Windows op triage.",
)
def test_uncategorized_ops_probe_cleanly():
    """One-sided guard on the classification exclusion lists.

    Every op the concrete support sweep runs forward -- i.e. every drivable op we
    DON'T already classify away -- must run under the side-effect probe cleanly:
    without blocking, crashing the interpreter, or reaching for I/O.  A non-clean
    result names an op the sweep would run for real (wedging a worker, or doing
    actual I/O), so it must be categorized:

    * a blocking / crashing op -> ``PROBE_HAZARD_OVERRIDES``
    * an I/O op                -> ``SIDE_EFFECT_OVERRIDES``
    * an OS-handle op          -> ``_OS_HANDLE_PARAMS`` (out_of_scope)

    Deliberately one-sided: already-excluded ops are NOT re-checked (an
    input-dependent hang would only flake, and they already read as grey TODO cells
    on the support grid).  We assert only that nothing NEW slips through
    uncategorized -- which is what a Python / typeshed bump can introduce.

    Per-platform: this sweep only sees the surface of the interpreter it runs on --
    a module enumerates its ops only where it imports (winreg / winsound / msvcrt on
    Windows; os.lchmod / chflags on macOS/BSD).  Run it on each major platform; a
    new offender goes into the SAME tables as everything else.  Today the known
    off-Linux entrypoints are HAND-classified (the Linux sweep can't run them), so
    on a fresh platform expect this to name any we mis-judged -- fix the table, same
    workflow as a version bump.

    Slow: the probe forks per op over the whole documented surface (~3400 drivable
    ops / ~9min today), and a HANG is re-probed once at a longer timeout to shed
    fork-latency flakes.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _SWEEP],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    offenders = [line for line in proc.stdout.splitlines() if line.strip()]
    detail = "\n  ".join(offenders)
    if proc.returncode != 0:
        detail += f"\n[sweep exited {proc.returncode}]\n{proc.stderr[-2000:]}"
    assert proc.returncode == 0 and not offenders, (
        "uncategorized ops don't probe cleanly (hang / crash / I/O); categorize "
        "them so the concrete sweep skips them:\n  " + detail
    )
