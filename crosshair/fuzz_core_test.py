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
live in ``KNOWN_FAILURES`` and are xfail'd ``strict`` -- so when one gets fixed,
its entry must be removed (the list doubles as the open-bug registry).
"""

from typing import Dict

import pytest

import crosshair.core_and_libs  # noqa: F401  -- ensure patches/plugins load
from crosshair.inputgen import TYPES, _module_funcs, func_call, op_call, surface
from crosshair.test_util import DIFFERENTIAL_SKIP, run_differential

# Inputs checked per operation (each pinned symbolic-vs-concrete).  Small for CI.
INPUTS_PER_OP = 3

# Operations whose symbolic model diverges from concrete execution -- real
# soundness bugs surfaced by this test (forward-computation divergences; distinct
# from the support matrix's "black", which is *inverse*-search unsoundness).
# xfail'd NON-strict: which bugs reproduce varies by Python version (e.g.
# int.to_bytes args are optional only on 3.11+) and by solver timing, so a strict
# xfail would flake; run `pytest -rX crosshair/fuzz_core_test.py` to spot fixes
# (XPASS) and prune this list.
KNOWN_FAILURES: Dict[str, str] = {
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
    # bytes/bytearray.startswith (and removeprefix, which calls it) reject a
    # SymbolicBytes argument -- only hit on versions whose fuzzed inputs include a
    # bytes prefix arg (3.9-3.11 here; harmless XPASS elsewhere, hence non-strict).
    "bytes.startswith": "[3.9-3.11] symbolic bytes.startswith rejects a SymbolicBytes arg -> TypeError",
    "bytes.removeprefix": "[3.9-3.11] bytes.removeprefix calls startswith, which rejects SymbolicBytes -> TypeError",
    "bytearray.startswith": "[3.9-3.11] symbolic bytearray.startswith rejects a SymbolicBytes arg -> TypeError",
    "bytearray.removeprefix": "[3.9-3.11] bytearray.removeprefix calls startswith, which rejects SymbolicBytes -> TypeError",
    # symbolic bytearray mutators skip CPython's byte-must-be-in-range(0,256) check
    "bytearray.append": "[3.9-3.11] symbolic bytearray.append skips the byte-range check (no ValueError)",
    "bytearray.extend": "[3.9-3.11] symbolic bytearray.extend skips the byte-range check (no ValueError)",
    "bytearray.insert": "[3.9-3.11] symbolic bytearray.insert skips the byte-range check (no ValueError)",
    "bytearray.__setitem__": "[3.9-3.11] symbolic bytearray[i]=v raises IndexError vs concrete ValueError (no byte-range check)",
}

# Ops the differential can't meaningfully check (order-dependent / incomparable /
# not a pure value) live in test_util.DIFFERENTIAL_SKIP, shared with the support
# measurement so neither flags them.
EXCLUDED = DIFFERENTIAL_SKIP


def _check(label, call) -> None:
    """Assert symbolic == concrete across this op's valid inputs."""
    fn, expr, names, eval_globals = call
    result = run_differential(fn, expr, names, eval_globals, k=INPUTS_PER_OP)
    if result.checked == 0:
        pytest.skip(f"no drivable inputs for {label}")
    assert result.divergence is None, f"{label} diverges {result.divergence.describe()}"


def _marks(label):
    """Skip excluded ops; xfail known soundness bugs (non-strict: see
    KNOWN_FAILURES -- reproduction varies by Python version / solver timing)."""
    if label in EXCLUDED:
        return [pytest.mark.skip(reason=EXCLUDED[label])]
    if label in KNOWN_FAILURES:
        return [pytest.mark.xfail(reason=KNOWN_FAILURES[label], strict=False)]
    return []


def _builtin_type_ops():
    for typ in TYPES:
        for method in surface(typ):
            label = f"{typ.__name__}.{method}"
            if op_call(typ, method):
                yield pytest.param(typ, method, id=label, marks=_marks(label))


def _builtin_func_ops():
    for name in sorted(_module_funcs("builtins")):
        label = f"builtins.{name}"
        if func_call("builtins", name):
            yield pytest.param(name, id=label, marks=_marks(label))


@pytest.mark.parametrize("typ,method", list(_builtin_type_ops()))
def test_builtin_type_op(typ, method):
    _check(f"{typ.__name__}.{method}", op_call(typ, method))


@pytest.mark.parametrize("name", list(_builtin_func_ops()))
def test_builtin_func_op(name):
    _check(f"builtins.{name}", func_call("builtins", name))


# Opt-in (``--run-slow``): the same differential over a broad set of pure stdlib
# free functions.  EXPLORATORY -- not a CI gate and not curated into
# KNOWN_FAILURES, so failures here are candidate soundness bugs to triage (the
# support matrix already colors these; this confirms forward divergences).
STDLIB_MODULES = (
    "math cmath statistics fractions string textwrap unicodedata difflib shlex "
    "html json base64 binascii quopri zlib hashlib hmac itertools functools "
    "operator heapq bisect collections calendar colorsys fnmatch posixpath "
    "urllib.parse ipaddress uuid struct ast graphlib codecs"
).split()


def _stdlib_func_ops():
    for module in STDLIB_MODULES:
        for name in sorted(_module_funcs(module)):
            if not name.startswith("_") and func_call(module, name):
                yield pytest.param(module, name, id=f"{module}.{name}")


@pytest.mark.slow
@pytest.mark.parametrize("module,name", list(_stdlib_func_ops()))
def test_stdlib_func_op(module, name):
    _check(f"{module}.{name}", func_call(module, name))
