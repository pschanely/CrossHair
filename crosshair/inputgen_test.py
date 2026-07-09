"""Tests for crosshair.inputgen -- the operation catalog and input generation."""

import subprocess
import sys
import textwrap

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
_SWEEP = textwrap.dedent("""
    import sys, time
    from crosshair.inputgen import catalog, probe_side_effect
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
        # Heartbeat to stderr BEFORE probing, flushed: since ops run in-process, a
        # blocking/crashing op wedges or kills the whole sweep -- so the last line
        # names the culprit.  stderr is only surfaced on failure, quiet normally.
        sys.stderr.write(f"{i}/{total} {time.time()-start:.0f}s {op.key}\\n")
        sys.stderr.flush()
        reason = probe_side_effect(op.call, seedkey=None)
        if reason is not None:
            bad.append(f"{op.key}: {reason}")
    sys.stdout.write("\\n".join(bad))
    """)


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
    sweep instead of being caught-and-named, and the heartbeat tail plus
    ``probe_side_effect_isolated`` (per-op fork isolation) pin it for debugging.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SWEEP],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        # The in-process sweep takes ~20s; a 300s overrun means an op BLOCKED.  The
        # heartbeat's last line names it; re-probe it under probe_side_effect_isolated
        # to confirm, then add it to PROBE_HAZARD_OVERRIDES.
        tail = exc.stderr or b""
        if isinstance(tail, bytes):
            tail = tail.decode("utf-8", "replace")
        raise AssertionError(
            "op probe sweep timed out (an op blocked in-process); the LAST "
            "heartbeat line below names it. Confirm with "
            "probe_side_effect_isolated, then add it to PROBE_HAZARD_OVERRIDES:\n"
            + tail[-2000:]
        )
    offenders = [line for line in proc.stdout.splitlines() if line.strip()]
    detail = "\n  ".join(offenders)
    if proc.returncode != 0:
        # The sweep process died -- an op crashed the interpreter (the auditwall
        # can't isolate that in-process).  The heartbeat's last line names it.
        detail += (
            f"\n[sweep exited {proc.returncode}: an op crashed the interpreter; "
            "the LAST heartbeat line names it -- confirm with "
            f"probe_side_effect_isolated]\n{proc.stderr[-2000:]}"
        )
    assert proc.returncode == 0 and not offenders, (
        "uncategorized ops don't probe cleanly (I/O side effect, or a hang/crash); "
        "categorize them so the concrete sweep skips them:\n  " + detail
    )
