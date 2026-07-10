"""Tests for crosshair.inputgen -- the operation catalog and input generation."""

import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
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


# The sweep runs in ONE clean subprocess, not in pytest's process: it runs ops
# concretely with the auditwall engaged, and pytest's multithreading + its
# breakpoint/capture hooks would perturb that (and ops like builtins.breakpoint).
# Within that subprocess it probes each op IN-PROCESS (probe_side_effect, no per-op
# fork) -- ~40x faster than isolating every op.  The surface is vetted, so we don't
# expect a hang/crash; if one slips in it wedges or kills the sweep rather than
# being caught-and-named, and the caller falls back to the isolated probe to debug.
#
# Everything crosses back to the parent via FILES, not pipes: the heartbeat file
# (argv[1]) names the in-flight op, the offenders file (argv[2]) collects results.
# A pipe would let a stray grandchild hold its write-end open and wedge the parent's
# drain read past the kill -- turning a 5-min named timeout into a 6h CI ceiling.
# Some hangs (e.g. aifc.open) don't even yield to an in-process SIGALRM, so the
# parent's process-group kill is the only reliable backstop; files survive it.
_SWEEP = textwrap.dedent("""
    import sys, time
    from crosshair.inputgen import catalog, probe_side_effect
    heartbeat, offenders = sys.argv[1], sys.argv[2]
    # Skip what the sweep never runs forward: undrivable, no synthesizable inputs,
    # or already excluded as out-of-scope / a side effect / a probe hazard.
    # not_value_function ops ARE run (measured, black-suppressed), so they stay in.
    ops = [op for op in catalog(probe=False)
           if not (op.call is None or op.out_of_scope or op.no_inputs
                   or op.side_effect or op.probe_hazard)]
    total = len(ops)
    start = time.time()
    bad = []
    for i, op in enumerate(ops, 1):
        # Heartbeat to a file BEFORE probing, flushed: since ops run in-process, a
        # blocking/crashing op wedges or kills the whole sweep -- so the last line
        # names the culprit and survives a hard process-group kill of the child.
        with open(heartbeat, "w") as fh:
            fh.write(f"{i}/{total} {time.time()-start:.0f}s {op.key}\\n")
        reason = probe_side_effect(op.call, seedkey=None)
        if reason is not None:
            bad.append(f"{op.key}: {reason}")
    with open(offenders, "w") as fh:
        fh.write("\\n".join(bad))
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

    Fast path: ops are probed IN-PROCESS with the auditwall (no per-op fork), which
    detects an I/O offender directly (~20s for the whole surface).  A hang or crash
    is NOT expected on this vetted surface; if one occurs it wedges or kills the
    sweep instead of being caught-and-named, and the heartbeat file plus
    ``probe_side_effect_isolated`` (per-op fork isolation) pin it for debugging.
    """

    def _read(path: str) -> str:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                return fh.read().strip()
        except OSError:
            return ""

    with tempfile.TemporaryDirectory() as tmp:
        heartbeat = os.path.join(tmp, "heartbeat")
        offenders_path = os.path.join(tmp, "offenders")
        # start_new_session=True puts the child in its own process group so a hang can
        # be killed group-wide (child + any stray grandchild).  stdout/stderr go to
        # DEVNULL -- all diagnostics travel through files, so no inherited pipe can
        # hold the parent hostage past the kill (the bug that turned a hang into a 6h
        # CI ceiling).  A blocked op that ignores SIGALRM still dies to the group kill.
        proc = subprocess.Popen(
            [sys.executable, "-c", _SWEEP, heartbeat, offenders_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            # The in-process sweep takes ~20s; a 300s overrun means an op BLOCKED.
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            raise AssertionError(
                "op probe sweep timed out (an op blocked in-process); the heartbeat "
                "below names the in-flight op. Confirm with probe_side_effect_"
                "isolated, then add it to PROBE_HAZARD_OVERRIDES:\n  "
                + (_read(heartbeat) or "(no heartbeat written)")
            )

    offenders = [line for line in _read(offenders_path).splitlines() if line.strip()]
    detail = "\n  ".join(offenders)
    if proc.returncode != 0:
        # The sweep process died -- an op crashed the interpreter (the auditwall
        # can't isolate that in-process).  The heartbeat names the in-flight op.
        detail += (
            f"\n[sweep exited {proc.returncode}: an op crashed the interpreter; the "
            "heartbeat names the in-flight op -- confirm with "
            f"probe_side_effect_isolated]\n  {_read(heartbeat)}"
        )
    assert proc.returncode == 0 and not offenders, (
        "uncategorized ops don't probe cleanly (I/O side effect, or a hang/crash); "
        "categorize them so the concrete sweep skips them:\n  " + detail
    )
