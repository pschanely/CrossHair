"""
Differential correctness test for CrossHair's symbolic operations.

For every builtin operation we draw concrete, valid inputs (typeshed-driven, from
``crosshair.inputgen``), pin a fresh symbolic value to each, run the SINGLE
operation symbolically, and assert the outcome -- return value, exception type,
and any in-place mutation -- matches running the same operation concretely.  A
divergence is a soundness bug in CrossHair's model of that operation.

This is the thorough form of the support matrix's "black" check (see
``crosshair/tools/measure_support.py``): "black" detects the cheap proxy
(CrossHair falsely *confirms* that no input yields a known-reachable output);
this test catches *any* symbolic-vs-concrete divergence, including bogus
witnesses.

Deliberately NOT a wide fuzz: inputs are bounded and each is checked on a single
pinned execution path, so the suite is fast enough for CI.  Known soundness gaps
live in ``KNOWN_FAILURES`` and are xfail'd NON-strict (their reproduction varies
by Python version and solver timing -- see the note there).
"""

import sys

import pytest

import crosshair.core_and_libs  # noqa: F401  -- ensure patches/plugins load
from crosshair.behavior_compare import run_differential
from crosshair.inputgen import catalog, inputs_for

# Inputs checked per operation (each pinned symbolic-vs-concrete).  Small for CI.
INPUTS_PER_OP = 3

# Per-input pin budget for the differential.  Deliberately small: an input that
# pins at all does so on the first iteration in well under a second, so the only
# effect of a larger budget is to slow the inputs that never pin (an op CrossHair
# can't model, or a value it can't match) -- pure waste in a CI gate that just
# needs one pinned path per op.  The thorough support measurement keeps the larger
# default (crosshair.behavior_compare).
PIN_ITERS = 12
PIN_TIMEOUT = 4.0

# Operations whose symbolic model diverges from concrete execution -- real
# soundness bugs surfaced by this test (forward-computation divergences; distinct
# from the support matrix's "black", which is *inverse*-search unsoundness).
# xfail'd NON-strict: which bugs reproduce varies by Python version (e.g.
# int.to_bytes args are optional only on 3.11+) and by solver timing, so a strict
# xfail would flake; run `pytest -rX crosshair/fuzz_core_test.py` to spot fixes
# (XPASS) and prune this list.
KNOWN_FAILURES = {
    "str.__format__": "format(symbolic_str) diverges from concrete",
    "float.__floordiv__": "symbolic float // float returns an int instead of a float",
    "float.__divmod__": "symbolic divmod(float, float) returns an int quotient instead of a float",
    "float.__mod__": "symbolic float % float diverges from concrete on extreme values",
    "float.__pow__": "symbolic float ** float crashes realizing the result (ArithRef.as_fraction)",
    "float.__round__": "symbolic round(float, ndigits) overflows (int too large to convert to float)",
    # (bytes/bytearray.startswith + removeprefix used to be here -- they rejected a
    # SymbolicBytes argument on <3.12, where there's no buffer protocol.  Fixed by
    # realizing the affix in AbcString.startswith/endswith; now pass on all versions.)
    # (bytearray.append/extend/insert/__setitem__ used to be here -- they skipped
    # CPython's byte-must-be-in-range(0,256) check.  Fixed by validating stored
    # values in _as_byte_value/_validated_byte_values.)
    "bytearray.resize": "[3.14+] resize() is new in 3.14 and unmodeled on SymbolicByteArray -> AttributeError",
    "bytearray.take_bytes": "[3.15+] take_bytes() is new in 3.15 and unmodeled on SymbolicByteArray -> AttributeError",
    # Surfaced by the aliased `(x, x)` CUSTOM_INPUTS strategy: symbolic execution
    # treats the two arguments as distinct objects, so `x is x` reads False (and
    # `x is not x` True) where concrete Python says the opposite -- crosshair
    # doesn't model that two parameters can be aliased.  crosshair CAN alias values
    # in some nested cases, but not yet at the top level (two separate parameters);
    # when that lands, the differential harness must pin an aliased concrete input
    # to a SHARED symbolic proxy (today run_symbolic_pinned pins each arg name to
    # its own proxy) -- and these two xfails should then flip to passing.
    "operator.is_": "symbolic `x is x` returns False (argument aliasing unmodeled)",
    "operator.is_not": "symbolic `x is not x` returns True (argument aliasing unmodeled)",
    # --- stdlib soundness bugs surfaced by the exclusion-model surface (Phase 2
    # baseline).  Grouped by root cause; xfail NON-strict (reproduction varies by
    # version/solver).  Prune with `pytest -rX` as fixes land. ---
    # ROOT CAUSE 1: a C function parses its int arg with the "i"/"I"/"index" format,
    # which rejects a symbolic int ("an integer is required" / "expected int" /
    # __index__ TypeError) instead of realizing it.  Fixed by realize-patches in the
    # owning libimpl modules: stat.S_I* / stat.filemode (statlib), os/posix
    # major/minor/makedev (oslib), socket.htonl/ntohl/if_indextoname (socketlib), and
    # ipaddress.ip_network/ip_interface (ipaddresslib).
    # ROOT CAUSE 2: symbolic float arithmetic diverges (cf. the float.* entries above).
    "colorsys.hls_to_rgb": "symbolic float arithmetic diverges from concrete",
    "colorsys.hsv_to_rgb": "symbolic float arithmetic diverges from concrete",
    "colorsys.rgb_to_yiq": "symbolic float arithmetic diverges from concrete",
    "statistics.covariance": "symbolic float arithmetic diverges from concrete",
    "statistics.median_grouped": "symbolic float arithmetic diverges from concrete",
    # fmean's weighted path (the optional `weights` arg, driven by the maximal shape)
    # sums/divides floats and diverges in the last ULP; input-dependent, so it only
    # reproduces where the sample hits extreme magnitudes (weights added in 3.11).
    "statistics.fmean": "symbolic float arithmetic diverges from concrete (weighted mean)",
    # ROOT CAUSE 3: a serializer / parser / compiler rejects a symbolic value instead
    # of realizing it (marshal/pickle unmarshallable, compile() wants a real str/bytes).
    "marshal.dumps": "symbolic value reported unmarshallable (should realize first)",
    "pickle.dumps": "symbolic value not pickled (should realize first)",
    "pickle.decode_long": "diverges on invalid input error handling",
    "struct.unpack": "symbolic format/buffer diverges (UnicodeEncodeError)",
    "codecs.escape_encode": "symbolic bytes rejected (should realize first)",
    # ROOT CAUSE 4: symbolic str / regex operations diverge from concrete.
    "shlex.join": "symbolic str quoting diverges (regex match differs)",
    "shlex.quote": "symbolic str quoting diverges (regex match differs)",
    "urllib.parse.unquote": "symbolic str percent-decoding diverges",
    "inspect.getblock": "symbolic source tokenization diverges (TypeError)",
    # --- surfaced by the per-version CI gate (the Linux dev sweep runs one
    # interpreter; these reproduce on other versions).  Most are the SymbolicBytes
    # analogue of ROOT CAUSE 3 above: a C function that consumes a bytes-like arg
    # rejects SymbolicBytes ("a bytes-like object is required, not 'SymbolicBytes'")
    # on Python <3.12, where SymbolicBytes exposes no buffer protocol -- so they
    # xfail there and XPASS (harmlessly, non-strict) on 3.12+. ---
    "base64.a85encode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "base64.b16encode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "base64.b85encode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.a2b_hex": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.b2a_hex": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.b2a_hqx": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.b2a_qp": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.b2a_uu": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.crc32": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.crc_hqx": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.hexlify": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.rlecode_hqx": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.rledecode_hqx": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "binascii.unhexlify": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "bz2.compress": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "bz2.decompress": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.ascii_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.charmap_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.iterdecode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.latin_1_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.utf_16_be_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.utf_16_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.utf_16_ex_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.utf_16_le_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.utf_32_be_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.utf_32_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.utf_32_ex_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.utf_32_le_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.utf_7_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "codecs.utf_8_decode": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "gzip.decompress": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "hmac.compare_digest": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "hmac.new": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "lzma.compress": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "lzma.decompress": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "marshal.loads": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "pickle.loads": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "pickletools.dis": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "pickletools.genops": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "pickletools.optimize": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "plistlib.loads": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "quopri.encodestring": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "secrets.compare_digest": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "socket.inet_ntoa": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "socket.inet_ntop": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    "ssl.DER_cert_to_PEM_cert": "C helper rejects SymbolicBytes (no buffer protocol <3.12)",
    # SymbolicList analogue -- heappush_max/heappushpop_max (new in 3.14) reject a
    # SymbolicList ("argument 1 must be list, not SymbolicList").
    "heapq.heappush_max": "C helper rejects SymbolicList (should realize first)",
    "heapq.heappushpop_max": "C helper rejects SymbolicList (should realize first)",
    # symbolic int rejected by a C helper ("an integer is required"), cf. ROOT CAUSE 1.
    "decimal.IEEEContext": "symbolic int rejected by the C context helper",
    # (zlib.adler32_combine / crc32_combine are classified PROBE_HAZARD instead -- the
    # concrete sweep HANGS on them, so they're skipped before the differential runs.)
    # CrossHair-internal / modeling gaps the differential exposes:
    "urllib.parse.quote": "CrossHairInternal: numeric op on symbolic while not tracing",
    "urllib.parse.quote_from_bytes": "CrossHairInternal: numeric op on symbolic while not tracing",
    "urllib.parse.quote_plus": "CrossHairInternal: numeric op on symbolic while not tracing",
    "urllib.parse.unquote_plus": "CrossHairInternal: numeric op on symbolic while not tracing",
    "urllib.parse.urlencode": "CrossHairInternal: numeric op on symbolic while not tracing",
    "difflib.ndiff": "SymbolicBool leaks through __bool__ (TypeError in difflib.compare)",
    "pipes.quote": "symbolic str quoting diverges (regex match differs; <3.13 only)",
    # symbolic datetime methods diverge from concrete (surfaced by making the
    # datetime receiver types drivable).
    "datetime.date.__sub__": "symbolic date - datetime returns a timedelta instead of raising TypeError",
    "datetime.date.isocalendar": "symbolic date.isocalendar() diverges from concrete (IsoCalendarDate)",
    "datetime.datetime.isocalendar": "symbolic datetime.isocalendar() diverges from concrete (IsoCalendarDate)",
    # --- surfaced by driving optional / keyword-only arguments (the shape-list
    # refactor: an op is now driven once per call shape, including a MAXIMAL shape
    # that fills the defaulted tail).  Each is a pre-existing model gap that the
    # primary shape never reached because it never passed the argument. ---
    # (bytes/bytearray find-family used to be here -- they mishandled a large-negative
    # ``start`` by offsetting by it instead of clamping to 0.  Fixed in AbcString._find.)
    # (bytes/bytearray.translate used to be here -- AbcString.translate modeled only the
    # 1-arg form; BytesLike.translate now handles the optional ``delete`` argument.)
    # ROOT CAUSE 1 (C helper rejects a symbolic int instead of realizing it): the
    # optional/keyword-only int now reaches a C function that parses it strictly.
    # (os.eventfd/posix.eventfd used to be here -- realizing can't fix them: each call
    # allocates a fresh fd, so a symbolic run and a concrete run return different fd
    # numbers.  Reclassified as SIDE_EFFECT_OVERRIDES, so the sweep never runs them.)
    "hashlib.scrypt": "symbolic int (n) rejected by the C scrypt helper (should realize)",
    # strptime's format path operates on a symbolic outside a statespace context.
    "time.strptime": "CrossHairInternal: strptime(string, format) leaves the statespace context",
    # --- surfaced by structural container pinning (core.pin_to descends into
    # nested / heap-backed containers, so inputs that previously could not be pinned
    # -- and were skipped as "no drivable inputs" -- now drive the differential).
    # Each is a pre-existing model gap unrelated to pinning. ---
    # ListBasedDeque implements no rich comparisons, so `deque <=> deque` raises
    # TypeError where concrete deques order lexicographically.
    "collections.deque.__lt__": "symbolic deque comparison unsupported (ListBasedDeque has no __lt__)",
    "collections.deque.__le__": "symbolic deque comparison unsupported (ListBasedDeque has no __le__)",
    "collections.deque.__gt__": "symbolic deque comparison unsupported (ListBasedDeque has no __gt__)",
    "collections.deque.__ge__": "symbolic deque comparison unsupported (ListBasedDeque has no __ge__)",
    # symbolic array('B', ...) doesn't range-check stored values, so extend/fromlist
    # accept an out-of-range int instead of raising OverflowError (cf. the bytearray
    # byte-range fix); the resulting array is left unrealizable.
    "array.array.extend": "symbolic array.extend skips the element range check (should raise OverflowError)",
    "array.array.fromlist": "symbolic array.fromlist skips the element range check (should raise OverflowError)",
    # symbolic list slicing with a large negative step returns the whole list
    # instead of the correct (often empty) slice.
    "list.__getitem__": "symbolic list slicing diverges for a large negative step",
}

# Divergences that surface only on Windows (issue #467, the Windows op triage).
# Scoped to win32 -- not folded into KNOWN_FAILURES -- so they don't muddy the
# Linux signal, where these ops pass (the Linux differential draws different
# inputs and/or these are genuinely platform-specific). xfail NON-strict like
# KNOWN_FAILURES; prune with `pytest -rX` on Windows as models catch up. These
# reproduce with AND without the CI rlimit budget, so they're real model gaps,
# not solver-budget artifacts.
WINDOWS_KNOWN_FAILURES = {
    # Windows-only C surface: symbolic int/handle rejected or wrong value returned.
    "msvcrt.SetErrorMode": "[win32] symbolic msvcrt.SetErrorMode returns the wrong mode",
    "msvcrt.open_osfhandle": "[win32] symbolic open_osfhandle raises TypeError vs concrete OSError",
    "ctypes.set_last_error": "[win32] symbolic ctypes.set_last_error returns 0, not the prior error",
    # os.*_handle_inheritable are Windows-only (operate on handles, not fds); the
    # symbolic int isn't realized before the C helper, same as waitstatus below.
    "os.get_handle_inheritable": "[win32] symbolic int rejected by the C helper ('an integer is required')",
    "os.set_handle_inheritable": "[win32] symbolic int rejected by the C helper ('an integer is required')",
    # Platform-divergent ops (behave differently / only on Windows).
    "select.select": "[win32] select() rejects non-socket fds (WinError 10038); symbolic raises TypeError",
    "os.waitstatus_to_exitcode": "symbolic int rejected by the C helper ('an integer is required')",
    # Cross-platform ops that only DIVERGE on Windows here (pass on Linux CI).
    "operator.pow": "[win32] symbolic pow() of large ints returns None (unmodeled)",
    "operator.ipow": "[win32] symbolic ipow() of large ints returns None (unmodeled)",
    "statistics.linear_regression": "[win32] symbolic float arithmetic diverges (last-ULP)",
}

# Ops SKIPPED (not xfail'd) on Windows: these CRASH the interpreter/worker, so an
# xfail is unsafe -- they must not run at all. Keyed by module prefix so sibling
# ops can't flake in. NOT fuzzable value functions in any case.
#   - turtle.*: drives a live Tk canvas; raises CrossHairInternal / a Tcl error
#     or crashes the xdist worker (turtle.pencolor).
#   - msilib.*: Windows-only native MSI library (removed in 3.13, PEP 594); the
#     C functions access-violate on fuzzed args (msilib.CreateRecord). Only
#     reachable on the <3.13 Windows job; on 3.13 the module is gone.
WINDOWS_SKIP_PREFIXES = {
    "turtle.": "windows: turtle drives a live Tk canvas (crashes/diverges; not fuzzable)",
    "msilib.": "windows: msilib native calls access-violate on fuzzed args (crash)",
}


def _windows_skip_reason(seedkey):
    if sys.platform == "win32":
        for prefix, reason in WINDOWS_SKIP_PREFIXES.items():
            if seedkey.startswith(prefix):
                return reason
    return None


def _check(label, op):
    """Assert symbolic == concrete across this op's valid inputs, for EVERY overload
    shape it offers.  An op whose overloads differ in argument count needs one drive
    per shape (`pow(base, exp)` is a different code path from
    `pow(base, exp, mod)`), and the interesting behavior often lives outside the
    primary sig."""
    checked = 0
    for call in op.drives():
        inputs = inputs_for(call, k=INPUTS_PER_OP, seedkey=op.seedkey)
        result = run_differential(
            call, inputs, max_pin_iters=PIN_ITERS, pin_timeout=PIN_TIMEOUT
        )
        checked += result.checked
        assert (
            result.divergence is None
        ), f"{label} diverges on `{call.expr}` {result.divergence.describe()}"
    if checked == 0:
        pytest.skip(f"no drivable inputs for {label}")


def _op_marks(op):
    """Marks for one catalogued op.  Skip what we can't/shouldn't fuzz -- out of
    scope (OS handle), a probe hazard (blocks/crashes), a side effect (real I/O),
    or an op whose output isn't a comparable value function (order/identity/
    reflection) -- and xfail known soundness gaps.  The skip reasons come straight
    off the catalog's classification (crosshair.inputgen), the same fields the
    support map reads.  The whole surface -- builtin AND stdlib -- is a gate: a
    divergence on any catalogued op is a soundness bug, so it either fails the
    suite or is enumerated (with its root cause) in ``KNOWN_FAILURES``."""
    marks = []
    if op.out_of_scope:
        marks.append(pytest.mark.skip(reason=f"out of scope: {op.out_of_scope}"))
    elif op.no_inputs:
        marks.append(pytest.mark.skip(reason=f"no inputs: {op.no_inputs}"))
    elif op.probe_hazard:
        marks.append(pytest.mark.skip(reason=f"probe hazard: {op.probe_hazard}"))
    elif op.side_effect:
        marks.append(pytest.mark.skip(reason=f"side effect: {op.side_effect}"))
    elif op.not_value_function:
        marks.append(
            pytest.mark.skip(reason=f"not a value function: {op.not_value_function}")
        )
    elif _windows_skip_reason(op.seedkey) is not None:
        marks.append(pytest.mark.skip(reason=_windows_skip_reason(op.seedkey)))
    elif op.seedkey in KNOWN_FAILURES:
        marks.append(pytest.mark.xfail(reason=KNOWN_FAILURES[op.seedkey], strict=False))
    elif sys.platform == "win32" and op.seedkey in WINDOWS_KNOWN_FAILURES:
        marks.append(
            pytest.mark.xfail(reason=WINDOWS_KNOWN_FAILURES[op.seedkey], strict=False)
        )
    return marks


# Enumerate the ONE canonical surface (crosshair.inputgen.catalog) -- the same set
# the support map measures, so the test and the map can't drift.  Static
# classification only (probe=False): fast, and complete here since this pure
# surface reaches for no live-probed side effects.  Keyed by the rendered key; the
# test looks the Operation back up, so params stay picklable (xdist-safe).
_CATALOG = {
    op.key: op
    for op in catalog(probe=False)
    if op.call is not None and not op.no_inputs
}


def _catalog_params():
    # sorted() so collection order is deterministic across processes. catalog()'s
    # yield order isn't stable process-to-process (it iterates object-keyed
    # collections whose order depends on address/ASLR), and pytest-xdist aborts
    # the run if its workers collect tests in different orders. Parametrization
    # order has no bearing on outcomes (each op is checked independently).
    for key in sorted(_CATALOG):
        yield pytest.param(key, id=key, marks=_op_marks(_CATALOG[key]))


@pytest.mark.parametrize("key", list(_catalog_params()))
def test_op(key):
    """Symbolic-vs-concrete differential for one catalogued operation."""
    op = _CATALOG[key]
    _check(key, op)
