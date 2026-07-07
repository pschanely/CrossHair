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
}


def _check(label, call):
    """Assert symbolic == concrete across this op's valid inputs."""
    fn, expr, names, eval_globals = call
    result = run_differential(fn, expr, names, eval_globals, k=INPUTS_PER_OP)
    if result.checked == 0:
        pytest.skip(f"no drivable inputs for {label}")
    assert result.divergence is None, f"{label} diverges {result.divergence.describe()}"


def _op_marks(op):
    """Marks for one catalogued op.  Skip what we can't/shouldn't fuzz -- out of
    scope (OS handle), a probe hazard (blocks/crashes), a side effect (real I/O),
    or an op whose output isn't a comparable value function (order/identity/
    reflection) -- and xfail known soundness gaps.  The skip reasons come straight
    off the catalog's classification (crosshair.inputgen), the same fields the
    support map reads.  Every NON-builtin op is gated behind ``--run-slow``: the
    builtin surface is the fast CI gate; the stdlib surface is EXPLORATORY (failures
    there are candidate soundness bugs to triage, not a gate)."""
    marks = []
    if op.out_of_scope:
        marks.append(pytest.mark.skip(reason=f"out of scope: {op.out_of_scope}"))
    elif op.probe_hazard:
        marks.append(pytest.mark.skip(reason=f"probe hazard: {op.probe_hazard}"))
    elif op.side_effect:
        marks.append(pytest.mark.skip(reason=f"side effect: {op.side_effect}"))
    elif op.not_value_function:
        marks.append(
            pytest.mark.skip(reason=f"not a value function: {op.not_value_function}")
        )
    elif op.seedkey in KNOWN_FAILURES:
        marks.append(pytest.mark.xfail(reason=KNOWN_FAILURES[op.seedkey], strict=False))
    if op.module != "builtins":
        marks.append(pytest.mark.slow)
    return marks


# Enumerate the ONE canonical surface (crosshair.inputgen.catalog) -- the same set
# the support map measures, so the test and the map can't drift.  Static
# classification only (probe=False): fast, and complete here since this pure
# surface reaches for no live-probed side effects.  Keyed by the rendered key; the
# test looks the Operation back up, so params stay picklable (xdist-safe).
_CATALOG = {op.key: op for op in catalog(probe=False) if op.call is not None}


def _catalog_params():
    for key, op in _CATALOG.items():
        yield pytest.param(key, id=key, marks=_op_marks(op))


@pytest.mark.parametrize("key", list(_catalog_params()))
def test_op(key):
    """Symbolic-vs-concrete differential for one catalogued operation."""
    _check(key, _CATALOG[key].call)
