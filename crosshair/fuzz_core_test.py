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
from crosshair.inputgen import catalog

# Inputs checked per operation (each pinned symbolic-vs-concrete).  Small for CI.
INPUTS_PER_OP = 3

# Operations whose symbolic model diverges from concrete execution -- real
# soundness bugs surfaced by this test (forward-computation divergences; distinct
# from the support matrix's "black", which is *inverse*-search unsoundness).
# xfail'd NON-strict: which bugs reproduce varies by Python version (e.g.
# int.to_bytes args are optional only on 3.11+) and by solver timing, so a strict
# xfail would flake; run `pytest -rX crosshair/fuzz_core_test.py` to spot fixes
# (XPASS) and prune this list.
KNOWN_FAILURES = {
    # length/byteorder became optional in 3.11; only then do we synthesize the
    # no-arg call that the symbolic impl mishandles, so this reproduces on 3.11+.
    "int.to_bytes": "[3.11+] symbolic int.to_bytes ignores its now-optional args -> TypeError",
    "bool.to_bytes": "[3.11+] symbolic bool.to_bytes ignores its now-optional args -> TypeError",
    "str.__format__": "format(symbolic_str) diverges from concrete",
    "tuple.__getitem__": "symbolic tuple[i] wraps the index modulo len instead of raising IndexError",
    "float.__floordiv__": "symbolic float // float returns an int instead of a float",
    "float.__divmod__": "symbolic divmod(float, float) returns an int quotient instead of a float",
    "float.__mod__": "symbolic float % float diverges from concrete on extreme values",
    "float.__pow__": "symbolic float ** float crashes realizing the result (ArithRef.as_fraction)",
    "float.__round__": "symbolic round(float, ndigits) overflows (int too large to convert to float)",
    # (bytes/bytearray.startswith + removeprefix used to be here -- they rejected a
    # SymbolicBytes argument on <3.12, where there's no buffer protocol.  Fixed by
    # realizing the affix in AbcString.startswith/endswith; now pass on all versions.)
    # symbolic bytearray mutators skip CPython's byte-must-be-in-range(0,256)
    # check.  (Reproduces on all supported versions incl. 3.12 -- the earlier
    # "[3.9-3.11]" tag was a guess from when these ops couldn't be input-bound and
    # so were never actually evaluated; the bytes-unification fix made them run.)
    "bytearray.append": "symbolic bytearray.append skips the byte-range check (no ValueError)",
    "bytearray.extend": "symbolic bytearray.extend skips the byte-range check (no ValueError)",
    "bytearray.insert": "symbolic bytearray.insert skips the byte-range check (no ValueError)",
    "bytearray.__setitem__": "symbolic bytearray[i]=v raises IndexError vs concrete ValueError (no byte-range check)",
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
    # __index__ TypeError) instead of realizing it -- a whole family of bit/id ops.
    "stat.S_IFMT": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_IMODE": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_ISBLK": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_ISCHR": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_ISDIR": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_ISDOOR": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_ISFIFO": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_ISLNK": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_ISPORT": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_ISREG": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_ISSOCK": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.S_ISWHT": "symbolic int rejected by the C stat helper (should realize first)",
    "stat.filemode": "symbolic int rejected by the C stat helper (should realize first)",
    "socket.htonl": "symbolic int rejected by the C byteorder helper (should realize)",
    "socket.ntohl": "symbolic int rejected by the C byteorder helper (should realize)",
    "socket.if_indextoname": "symbolic int rejected / interface-index lookup diverges",
    "os.major": "symbolic int rejected by the C device helper (should realize first)",
    "os.minor": "symbolic int rejected by the C device helper (should realize first)",
    "os.makedev": "symbolic int rejected by the C device helper (should realize first)",
    "posix.major": "symbolic int rejected by the C device helper (should realize first)",
    "posix.minor": "symbolic int rejected by the C device helper (should realize first)",
    "posix.makedev": "symbolic int rejected by the C device helper (should realize first)",
    "ipaddress.ip_network": "symbolic int not accepted via __index__ (TypeError)",
    "ipaddress.ip_interface": "symbolic int not accepted via __index__ (TypeError)",
    # ROOT CAUSE 2: symbolic float arithmetic diverges (cf. the float.* entries above).
    "colorsys.hls_to_rgb": "symbolic float arithmetic diverges from concrete",
    "colorsys.hsv_to_rgb": "symbolic float arithmetic diverges from concrete",
    "colorsys.rgb_to_yiq": "symbolic float arithmetic diverges from concrete",
    "statistics.covariance": "symbolic float arithmetic diverges from concrete",
    "statistics.median_grouped": "symbolic float arithmetic diverges from concrete",
    # ROOT CAUSE 3: a serializer / parser / compiler rejects a symbolic value instead
    # of realizing it (marshal/pickle unmarshallable, compile() wants a real str/bytes).
    "marshal.dumps": "symbolic value reported unmarshallable (should realize first)",
    "pickle.dumps": "symbolic value not pickled (should realize first)",
    "pickle.decode_long": "diverges on invalid input error handling",
    "struct.unpack": "symbolic format/buffer diverges (UnicodeEncodeError)",
    "ast.literal_eval": "symbolic str rejected by compile() (should realize first)",
    "code.compile_command": "symbolic source diverges through compile()",
    "codeop.compile_command": "symbolic source diverges through compile()",
    "dis.code_info": "symbolic source rejected by compile() (should realize first)",
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
    "base64.b16decode": "AbcString.translate() doesn't accept the 'delete' kwarg",
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
    # Platform-divergent ops (behave differently / only on Windows).
    "select.select": "[win32] select() rejects non-socket fds (WinError 10038); symbolic raises TypeError",
    "os.waitstatus_to_exitcode": "symbolic int rejected by the C helper ('an integer is required')",
    # Cross-platform ops that only DIVERGE on Windows here (pass on Linux CI).
    "operator.pow": "[win32] symbolic pow() of large ints returns None (unmodeled)",
    "operator.ipow": "[win32] symbolic ipow() of large ints returns None (unmodeled)",
    "statistics.linear_regression": "[win32] symbolic float arithmetic diverges (last-ULP)",
    "ast.parse": "symbolic str rejected by compile() (should realize first); cf. ast.literal_eval",
}

# Ops SKIPPED (not xfail'd) on Windows: turtle drives a live Tk canvas, which on
# the Windows runner raises CrossHairInternal / a Tcl error or CRASHES the xdist
# worker (turtle.pencolor) -- so it must not run at all. Not value functions here
# in any case. Skipped by module prefix so sibling turtle ops can't flake in.
WINDOWS_SKIP_PREFIXES = ("turtle.",)


def _windows_skip_reason(seedkey):
    if sys.platform == "win32" and seedkey.startswith(WINDOWS_SKIP_PREFIXES):
        return (
            "windows: turtle drives a live Tk canvas (crashes/diverges; not fuzzable)"
        )
    return None


def _check(label, call, seedkey):
    """Assert symbolic == concrete across this op's valid inputs."""
    fn, expr, names, eval_globals = call
    result = run_differential(
        fn, expr, names, eval_globals, k=INPUTS_PER_OP, seedkey=seedkey
    )
    if result.checked == 0:
        pytest.skip(f"no drivable inputs for {label}")
    assert result.divergence is None, f"{label} diverges {result.divergence.describe()}"


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
    _check(key, op.call, op.seedkey)
