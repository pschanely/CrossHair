"""Tests for crosshair.inputgen -- the operation catalog and input generation."""

import ast
import gettext
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
from fractions import Fraction
from statistics import NormalDist

import pytest

from crosshair.inputgen import (
    _candidate_sigs,
    _func_candidate_sigs,
    _literal_const,
    _literal_values,
    _method_owner,
    _sig_for,
    call_expr,
    can_synthesize_inputs,
    catalog,
    catalog_modules,
    documented_stdlib_modules,
    func_call,
    inputs_for,
    op_call,
    op_shapes,
    valid_inputs,
)


def test_catalog_surface_is_documented_only():
    """The exclusion-model surface enumerates DOCUMENTED stdlib modules only.  Two
    guards on the derivation: every enumerated module's top-level name is documented
    (catches a denylist/extra-module slip), and the undocumented internals that used
    to leak onto hand-maintained inclusion lists stay OFF -- they have no stable
    public contract to differentially test against."""
    documented = documented_stdlib_modules()
    surface = set(catalog_modules())
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


# Receivers typeshed doesn't name `self`: fractions.pyi writes its operators as
# `def __add__(a, b: int | Fraction)`, so a name-based receiver rule reads `a` as an
# unannotated argument and discards the whole operator surface.


@pytest.mark.parametrize(
    "typ,method,module,expr",
    [
        (Fraction, "__neg__", "fractions", "-a"),
        (Fraction, "__add__", "fractions", "a + b"),
        (Fraction, "__mod__", "fractions", "a % b"),
        (NormalDist, "__mul__", "statistics", "a * x2"),
        # A @staticmethod's leading parameter is a real argument, not a receiver.
        (str, "maketrans", "builtins", "a.maketrans(x)"),
        # A @classmethod's `cls` IS the receiver.
        (dict, "fromkeys", "builtins", "a.fromkeys(iterable)"),
    ],
)
def test_ops_with_unconventional_receivers_resolve(typ, method, module, expr):
    call = op_call(typ, method, module)
    assert call is not None, f"{typ.__name__}.{method} resolved no signature"
    assert call.expr == expr


def test_free_function_parameter_named_cls_is_not_dropped():
    """In builtins.issubclass(cls, class_or_tuple), `cls` is an argument."""
    call = func_call("builtins", "issubclass")
    assert call is not None
    assert call.arg_names == ("cls", "class_or_tuple")


def test_undrivable_zero_arg_call_is_not_synthesized():
    """functools.total_ordering(cls) takes a required argument, so a zero-arg
    candidate would only raise; it reports as undrivable instead."""
    assert func_call("functools", "total_ordering") is None


# Correlated CUSTOM_INPUTS: the transforming regime should dominate, while the
# blend still leaves some plain no-op draws.


def _fraction(tuples, predicate):
    ok = sum(1 for t in tuples if predicate(t))
    return ok / len(tuples) if tuples else 0.0


def test_replace_inputs_mostly_match():
    # `old` is usually an actual substring of the receiver, so replace transforms.
    tuples = valid_inputs(str.replace, k=40, seed=3, seedkey="str.replace")
    assert tuples
    assert _fraction(tuples, lambda t: len(t) >= 2 and t[1] in t[0]) >= 0.6


def test_pad_inputs_mostly_widen():
    # width is usually greater than len(receiver), so ljust actually pads.
    tuples = valid_inputs(str.ljust, k=40, seed=3, seedkey="str.ljust")
    assert tuples
    assert _fraction(tuples, lambda t: len(t) >= 2 and t[1] > len(t[0])) >= 0.6


def test_strip_inputs_mostly_trim():
    # the receiver usually carries surrounding whitespace, so strip() changes it.
    tuples = valid_inputs(str.strip, k=40, seed=3, seedkey="str.strip")
    assert tuples
    assert _fraction(tuples, lambda t: t[0].strip() != t[0]) >= 0.6


def test_blend_keeps_some_plain_draws():
    # the blend keeps some plain no-op draws (doesn't fully replace independent fuzz)
    tuples = valid_inputs(str.replace, k=60, seed=7, seedkey="str.replace")
    assert _fraction(tuples, lambda t: len(t) >= 2 and t[1] not in t[0]) > 0.0


def test_custom_inputs_ignored_without_seedkey():
    # A caller that omits the seedkey (patch_equivalence_test) gets plain fuzzing,
    # unchanged by the override -- so those clients are unaffected.
    with_key = valid_inputs(str.replace, k=40, seed=1, seedkey="str.replace")
    without_key = valid_inputs(str.replace, k=40, seed=1)
    match = lambda t: len(t) >= 2 and t[1] in t[0]  # noqa: E731
    assert _fraction(with_key, match) > _fraction(without_key, match)


@pytest.mark.parametrize(
    "fn,seedkey",
    [(bytes.replace, "bytes.replace"), (bytearray.replace, "bytearray.replace")],
)
def test_bytes_replace_inputs_mostly_match(fn, seedkey):
    # the byte-string variants correlate the needle the same way str does.
    tuples = valid_inputs(fn, k=40, seed=3, seedkey=seedkey)
    assert tuples
    assert _fraction(tuples, lambda t: len(t) >= 2 and t[1] in t[0]) >= 0.6


@pytest.mark.parametrize(
    "fn,seedkey", [(bytes.strip, "bytes.strip"), (bytearray.strip, "bytearray.strip")]
)
def test_bytes_strip_inputs_mostly_trim_and_keep_type(fn, seedkey):
    # the receiver carries surrounding whitespace AND keeps its type (a bytearray
    # receiver must not silently become bytes, or the method dispatches wrong).
    tuples = valid_inputs(fn, k=40, seed=3, seedkey=seedkey)
    assert tuples
    recv_type = type(tuples[0][0])
    assert all(isinstance(t[0], recv_type) for t in tuples)
    assert _fraction(tuples, lambda t: bytes(t[0]).strip() != bytes(t[0])) >= 0.6


# Call shapes: an op is driven once per shape, so behavior in a non-primary overload
# (pow's 2-arg form) or behind a defaulted argument (subn's ``count``) isn't hidden.


def test_shapes_list_primary_first():
    # pow's primary shape is the 3-argument (base, exp, mod) form; the 2-argument
    # shape carries the float/complex bases and the negative-exponent Literals.
    assert [len(s.key) for s in _func_candidate_sigs("builtins", "pow")] == [3, 2]


def test_optional_positional_arg_gets_a_maximal_shape():
    # str.center(width, fillchar=' '): the required shape passes only ``width``; the
    # maximal shape also fills the defaulted ``fillchar`` (positionally).
    shapes = _candidate_sigs(str, "center")
    assert [s.arg_names for s in shapes] == [("width",), ("width", "fillchar")]


def test_keyword_only_arg_is_rendered_by_keyword():
    # int.to_bytes(length, byteorder, *, signed=False): the optional keyword-only
    # ``signed`` is filled as ``signed=signed`` in the maximal shape.
    call = op_call(int, "to_bytes", "builtins", 1)
    assert call is not None and call.expr.endswith("signed=signed)")


@pytest.mark.parametrize(
    "shape,expect_expr",
    [(0, "_fn(base, exp, mod)"), (1, "_fn(base, exp)")],
)
def test_func_call_builds_the_requested_shape(shape, expect_expr):
    call = func_call("builtins", "pow", shape)
    assert call is not None and call.expr == expect_expr


def test_unknown_shape_is_not_drivable():
    assert func_call("builtins", "pow", 7) is None


@pytest.mark.parametrize("shape,width", [(1, 2), (0, 3)])
def test_valid_inputs_matches_the_requested_shape(shape, width):
    tuples = valid_inputs(pow, k=6, shape=shape)
    assert tuples and all(len(t) == width for t in tuples)


def test_two_argument_pow_reaches_negative_exponents():
    """The 2-arg shape carries Literal[-1..-20]; 3-arg pow rejects a negative
    exponent outright, so this coverage exists only off the primary shape."""
    exponents = [t[1] for t in valid_inputs(pow, k=60, seed=1, shape=1)]
    assert any(isinstance(e, int) and e < 0 for e in exponents)
    # and the complex/float bases that only the 2-arg shape declares
    bases = [t[0] for t in valid_inputs(pow, k=60, seed=1, shape=1)]
    assert any(isinstance(b, (float, complex)) for b in bases)


def test_catalog_drives_every_call_shape():
    ops = {op.key: op for op in catalog(probe=False)}
    assert [c.expr for c in ops["builtins.pow"].alt_calls] == ["_fn(base, exp)"]
    assert [c.shape for c in ops["builtins.pow"].drives()] == [0, 1]
    # getattr's optional `default` lives in a 3-argument overload.
    assert [c.expr for c in ops["builtins.getattr"].alt_calls] == [
        "_fn(o, name, default)"
    ]
    # re.Pattern.subn's ``count`` is a defaulted argument (one overload), reached
    # only via the maximal shape -- the coverage this refactor adds.
    assert [c.expr for c in ops["re.Pattern_subn_method"].alt_calls] == [
        "a.subn(repl, string, count)"
    ]


def test_every_drive_consumes_all_of_its_arguments():
    """A generated argument the expression can't place would be silently discarded,
    so every argument name must appear in the expression that drives it."""
    unplaced = []
    for op in catalog(probe=False):
        for call in op.drives():
            if any(name not in call.expr for name in call.arg_names):
                unplaced.append((op.key, call.expr, call.arg_names))
    assert not unplaced, f"arguments generated but never passed: {unplaced[:5]}"


def test_call_spec_arg_names_include_a_methods_receiver():
    """A method's arg_names lead with the synthesized receiver; a free function's
    don't -- so both line up with what valid_inputs generates for the shape."""
    method = op_call(dict, "get", "builtins")
    assert method.arg_names == ("a", "key") and method.shape == 0
    free = func_call("builtins", "pow")
    assert free.arg_names == ("base", "exp", "mod") and free.shape == 0


def test_call_spec_invokes_its_expression():
    assert func_call("builtins", "pow").invoke((2, 10, 1000)) == 24
    assert op_call(dict, "get", "builtins").invoke(({"k": 7}, "k")) == 7
    assert op_call(int, "__add__").invoke((2, 3)) == 5


@pytest.mark.parametrize("shape,width", [(1, 2), (0, 3)])
def test_call_spec_generates_inputs_its_expression_accepts(shape, width):
    """A spec supplies its own inputs, so the shape it drives and the shape it
    generates for can't drift apart."""
    call = func_call("builtins", "pow", shape)
    tuples = inputs_for(call, k=6)
    assert tuples and all(len(t) == width and call.accepts(t) for t in tuples)


def test_method_spec_generates_a_receiver_plus_its_arguments():
    call = op_call(dict, "get", "builtins")
    tuples = inputs_for(call, k=4)
    assert tuples
    assert all(len(t) == 2 and isinstance(t[0], dict) for t in tuples)


def test_method_does_not_borrow_a_same_named_module_function():
    """A module often exports a function sharing a method's name.  Resolving the
    method to that function would bind the RECEIVER to the function's first argument
    -- driving ``NullTranslations.install()`` on a plain ``str``.  An unconstructable
    receiver yields no signature at all instead."""
    assert _method_owner(gettext.NullTranslations.install) is gettext.NullTranslations
    assert _sig_for(gettext.NullTranslations.install) is None
    assert valid_inputs(gettext.NullTranslations.install, k=3) == []
    call = op_call(gettext.NullTranslations, "install", "gettext")
    assert call.expr == "a.install()" and call.arg_names == ("a",)
    assert not can_synthesize_inputs(call)
    # the module-level function of the same name is itself resolvable
    assert _sig_for(gettext.install)[0] == "func"
    assert valid_inputs(gettext.install, k=3)


def test_method_owner_found_through_a_closure_qualname():
    """fractions.Fraction's operators are built by a closure, so their qualname is
    ``Fraction._operator_fallbacks.<locals>.forward``.  The owner is the longest
    leading prefix that resolves to a type."""
    assert _method_owner(Fraction.__add__) is Fraction
    assert _method_owner(Fraction.__neg__) is Fraction
    receivers = [t[0] for t in valid_inputs(Fraction.__add__, k=3)]
    assert receivers and all(isinstance(r, Fraction) for r in receivers)


def test_local_function_has_no_method_owner():
    def outer():
        def inner():
            pass

        return inner

    assert _method_owner(outer()) is None


def test_can_synthesize_inputs_reports_unconstructable_arguments():
    assert can_synthesize_inputs(func_call("builtins", "pow"))
    # A receiver CrossHair has no construction strategy for yields no inputs.
    unconstructable = [
        op for op in catalog(probe=False) if op.no_inputs and op.call is not None
    ]
    assert unconstructable, "expected some ops with unconstructable arguments"
    assert not can_synthesize_inputs(unconstructable[0].call)


def test_operator_syntax_yields_to_the_dunder_form_when_it_cannot_place_args():
    # `a ** x` has room for one operand; int.__pow__'s (value, mod) overload needs
    # the explicit form rather than dropping `mod`.
    assert call_expr("__pow__", ["x"]) == "a ** x"
    assert call_expr("__pow__", ["value", "mod"]) == "a.__pow__(value, mod)"
    assert call_expr("__pow__", []) is None  # no operand to apply


# Literal[...] elements that aren't bare constants.


def test_literal_const_reads_negated_numerics():
    node = ast.parse("-3", mode="eval").body
    assert _literal_const(node) == -3


def test_literal_const_resolves_a_dotted_member():
    node = ast.parse("RegexFlag.IGNORECASE", mode="eval").body
    assert _literal_const(node, "re") is re.RegexFlag.IGNORECASE


def test_literal_values_reads_enum_members():
    src = "Literal[RegexFlag.IGNORECASE, RegexFlag.MULTILINE]"
    node = ast.parse(src, mode="eval").body
    assert _literal_values(node, "re") == (
        re.RegexFlag.IGNORECASE,
        re.RegexFlag.MULTILINE,
    )


def test_literal_values_drops_a_member_the_runtime_lacks():
    """typeshed models dataclasses._MISSING_TYPE as an Enum with a MISSING member;
    the runtime class has no such attribute, so no value can be produced."""
    node = ast.parse("Literal[_MISSING_TYPE.MISSING]", mode="eval").body
    assert _literal_values(node, "dataclasses") == ()


def test_purely_variadic_op_offers_only_its_expanded_form():
    """`s.difference_update()` passes nothing to a *args-only op, exercising none of
    it, so the expanded form replaces the bare one rather than joining it."""
    assert [len(s.key) for s in _candidate_sigs(set, "difference_update")] == [1]
    assert [len(s.key) for s in _func_candidate_sigs("math", "gcd")] == [1]


def test_declared_zero_arg_overload_is_still_driven():
    """re.Match.group()'s no-argument form is a real overload (not a *args artifact),
    and returns the whole match rather than a group -- worth driving."""
    arg_counts = [len(s.key) for s in _candidate_sigs(re.Match, "group", "re")]
    assert 0 in arg_counts
    zero_shape = arg_counts.index(0)
    assert op_call(re.Match, "group", "re", zero_shape).expr == "a.group()"
