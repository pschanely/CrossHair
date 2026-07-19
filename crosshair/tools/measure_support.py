"""
Measure how well CrossHair can reason about each Python operation.

For one operation we fuzz a real input, run the operation forward to a concrete
output, then ask CrossHair to work backwards -- find inputs that produce that
output (``post: _ != output``, looking for a counterexample).  We sweep the input
size and watch where CrossHair stops being able to do it:

    always succeeds          -> green   (CrossHair handles it well)
    succeeds until a cliff   -> yellow  (fine for small inputs, slows as they grow)
    only the trivial case    -> red     (CrossHair struggles to reason backwards)
    no drivable / opaque call -> "?"    (couldn't measure it)

The operation surface, the valid-input generation, AND the up-front operation
classification (drivable / out of scope / probe hazard / side effect) all come
from the ONE ``crosshair.inputgen.catalog`` -- the same surface the differential
fuzz test enumerates, so the map and the test can't drift.

CLI (emits the JSON that ``generate_treemap.py`` renders):
    python -m crosshair.tools.measure_support measure --json out.json
    # a subset (per-module, for the deadlock workaround; or a single tier):
    python -m crosshair.tools.measure_support measure --modules math,json --json out.json
    python -m crosshair.tools.measure_support measure --tiers builtin-methods,functions ...

``--tiers`` selects among {builtin-methods, functions, stdlib-methods}; the docs
map is ``--tiers builtin-methods,functions`` (no stdlib class methods).

The catalog is huge (~22k ops) but a weighted treemap only draws the few hundred
whose usage clears its ``--min-weight``.  Pass the same usage JSON to skip the
long tail up front and measure only what will be shown:
    python -m crosshair.tools.measure_support measure \\
        --weights usage.json --min-weight 1 --json out.json
"""

import argparse
import contextlib
import copy
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import queue as _queue
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from hypothesis import HealthCheck, assume, given
from hypothesis import seed as hyp_seed
from hypothesis import settings
from hypothesis import strategies as st

import crosshair.core_and_libs  # noqa: F401  -- loads opcode patches / registrations
from crosshair.behavior_compare import _is_opaque, run_differential
from crosshair.core import suspected_proxy_intolerance_exception
from crosshair.core_and_libs import analyze_function, run_checkables
from crosshair.inputgen import (  # shared surface + valid-input generation
    ANN_NS,
    NOT_VALUE_FUNCTION,
    RECV,
    SKIP_DUNDERS,
    _ann,
    _candidate_sigs,
    _func_candidate_sigs,
    _resolve_arg,
    call_expr,
    catalog,
    func_call,
    is_deterministic,
    op_call,
    receiver_name,
    surface,
    tuple_strategy,
)
from crosshair.options import AnalysisOptionSet
from crosshair.statespace import MessageType
from crosshair.tools.demo_overrides import demo_overrides, demo_sources

_DEVNULL = open(os.devnull, "w")


# ---------------------------------------------------------------------------
# measuring an op: fuzz an input, run forward, then ask CrossHair to invert it
# ---------------------------------------------------------------------------
PRE = "from typing import *\nimport builtins, collections, datetime, itertools, json, math, random, re, time\n\n"
HOLDOUT_OPTS = AnalysisOptionSet(
    per_condition_timeout=8, max_uninteresting_iterations=200_000
)
SIZES = range(1, 6)
_ctr = 0
# per-process scratch dir for synthesized modules (system temp)
_TMPDIR: Optional[Path] = None


def _tmpdir() -> Path:
    global _TMPDIR
    if _TMPDIR is None:
        _TMPDIR = Path(tempfile.mkdtemp(prefix="ch_measure_"))
    return _TMPDIR


def _load(src: str, lib: str, inject: Optional[Dict[str, Any]] = None) -> Callable:
    global _ctr
    _ctr += 1
    # Write to the system temp dir (not cwd) so we never dirty the repo or wake IDEs.
    p = _tmpdir() / f"m{os.getpid()}_{_ctr}.py"
    p.write_text(PRE + src)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[p.stem] = mod
    spec.loader.exec_module(mod)
    try:  # demo conditions may reference names from the lib's test module (bare deque, ...)
        tm = importlib.import_module(f"crosshair.libimpl.{lib}lib_test")
        for k, v in vars(tm).items():
            if not k.startswith("_") and k not in mod.__dict__:
                mod.__dict__[k] = v
    except ImportError:
        pass
    if inject:  # bind real objects (e.g. the holdout target) without a repr round-trip
        mod.__dict__.update(inject)
    return mod.f


def _noise(x: Any) -> int:
    """A penalty for how hard ``x`` reads in a generated demo, scored on its repr:
    literal non-ASCII codepoints cost the most, backslash escapes (``\\x``, ``\\u``,
    ``\\U``) next, with the repr length as a mild tiebreak.  A plain ``'abc'`` or a
    small int scores near zero; astral-plane and control-char noise scores high."""
    try:
        r = repr(x)
    except Exception:
        return 10**9
    return sum(100 for ch in r if ord(ch) > 0x7E) + r.count("\\") * 10 + len(r)


def _pair_noise(pair: Tuple[Any, Any]) -> int:
    """Total demo-readability penalty for a ``(input_tuple, output)`` sample -- the
    sum over every value that a pinned demo would render literally."""
    inp, out = pair
    return sum(_noise(x) for x in inp) + _noise(out)


def _is_echo(t: Sequence[Any], v: Any, i: int) -> bool:
    """True when output ``v`` equals input argument ``i`` -- an identity-in-that-arg
    op (``list.copy``, an already-stripped ``str.strip``, ...).  Inverting such an
    arg yields a paste-solvable demo (``post: _ != <the value you'd set it to>``),
    so we steer away from it and flag the cell for a curated override."""
    try:
        return bool(v == t[i])
    except Exception:
        return False


def _fuzz_valid(
    args: Sequence[Tuple[str, Any]],
    n: int,
    rngseed: int,
    record: Callable[[tuple], Optional[Tuple[Any, Any]]],
    seedkey: Optional[str] = None,
) -> Optional[Tuple[Any, Any]]:
    """Drive Hypothesis over a size-n input tuple and return a developed (not
    Hypothesis's first minimal) (input, target) pair ``record`` keeps, preferring
    the most readable one for the demo link.  ``record(t)`` returns the pair to
    store, or None to assume() the example away (an exception, or a
    degenerate/non-mutating call).  ``seedkey`` selects a CUSTOM_INPUTS override
    (shared with the differential path) when one is registered for this op.

    Readability is only a *preference among the valid inputs Hypothesis already
    drew* -- it never constrains the strategy (the full-Unicode alphabet still
    explores every codepoint), so the measured verdict stays as honest as picking
    an arbitrary sample; we just surface the legible one when there's a tie."""
    strat = tuple_strategy(seedkey, [spec for _, spec in args], n)
    found = []

    @hyp_seed(rngseed)
    @settings(
        max_examples=12,
        database=None,
        deadline=None,
        suppress_health_check=list(HealthCheck),
    )
    @given(strat)
    def run(t):
        pair = record(t)
        if pair is None:
            assume(False)
            return
        found.append(pair)

    try:
        run()
    except Exception:
        return None
    if not found:
        return None
    # Prefer the least-noisy sample; on ties keep the later (developed) one, matching
    # the previous "take the last valid example" behavior.  Readability is the ONLY
    # bias here: it swaps one arbitrary same-size input for another, so the graded
    # verdict stays as honest as before -- unlike an echo/transform bias, which would
    # systematically favor harder inputs and repaint the cell.
    return min(enumerate(found), key=lambda iv: (_pair_noise(iv[1]), -iv[0]))[1]


def fuzz_valid(
    args: Sequence[Tuple[str, Any]],
    n: int,
    forward: Callable,
    rngseed: int,
    seedkey: Optional[str] = None,
) -> Optional[Tuple[Any, Any]]:
    """A VALID, non-degenerate input at size n: skip exceptions and None results,
    and take a developed example."""
    import copy

    def record(t):
        pre = copy.deepcopy(t)  # value-returning mutators (list.pop) mutate t;
        try:
            v = forward(*t)
        except Exception:
            return None
        if v is None:
            return None
        return (pre, v)  # store the PRE-call input, not the mutated one

    return _fuzz_valid(args, n, rngseed, record, seedkey)


def fuzz_valid_mut(
    args: Sequence[Tuple[str, Any]],
    n: int,
    forward: Callable,
    rngseed: int,
    seedkey: Optional[str] = None,
) -> Optional[Tuple[Any, Any]]:
    """Like ``fuzz_valid`` but for an in-place mutator: a VALID, non-degenerate
    input at size n whose call returns None *and* actually changes the receiver
    (compared against its deepcopied original).  Captures the post-call state of
    the receiver as the holdout target."""
    import copy

    def record(t):
        orig = copy.deepcopy(t[0])
        try:
            r = forward(*t)
        except Exception:
            return None
        if r is not None or t[0] == orig:  # mutators return None and must mutate
            return None
        # store the PRE-call input (receiver before mutation) + the post-call
        # state as the target, so re-running it (determinism check / per-arg
        # pinning) starts from the original, not the already-mutated, receiver.
        return ((orig, *t[1:]), copy.deepcopy(t[0]))

    return _fuzz_valid(args, n, rngseed, record, seedkey)


def _invert_one(
    header: str,
    params: Sequence[Tuple[str, str, Any]],
    expr: str,
    free_i: int,
    t: Sequence[Any],
    V: Any,
    scope: Optional[str],
    lib: str,
) -> str:
    """Invert ONE argument: make param ``free_i`` symbolic, pin the rest to their
    values in ``t``, and ask CrossHair to find a ``free_i`` that makes the op
    produce ``V`` (``post: _ != V``).  This is the non-degenerate inversion -- the
    other arguments can't be set freely to absorb the output.  Pinned args are
    injected as concrete globals; a *mutated* receiver must stay a fresh per-call
    parameter, so it is pinned with a ``pre`` instead.  Returns
    'sat'/'unsat'/'unknown'/'unsupported'/'error'."""
    # ``_mark`` is appended to when the op rejects the symbolic argument (proxy
    # intolerance).  Without this the rejection is swallowed to None, the op looks
    # like it always returns None, ``post: _ != _V`` holds vacuously, CrossHair
    # CONFIRMs it, and we'd misread that as 'unsat' (a false soundness bug).  The
    # marker lets us report 'unsupported' instead -- CrossHair can't model a
    # symbolic here, so it only ever falls back to concrete values (graded red).
    marker: List[int] = []
    sig, pres = [], []
    inject: Dict[str, Any] = {
        "_V": V,
        "_proxy": suspected_proxy_intolerance_exception,
        "_mark": marker,
    }
    for j, (name, ann, _spec) in enumerate(params):
        if j == free_i:
            sig.append(f"{name}: {ann}")
        elif scope and j == 0:  # mutated receiver: fresh param, pinned via pre
            sig.append(f"{name}: {ann}")
            pres.append(f"pre: {name} == _pin_{name}")
            inject[f"_pin_{name}"] = copy.deepcopy(t[j])
        else:  # pinned argument: a concrete global referenced by the expression
            inject[name] = copy.deepcopy(t[j])
    target = f"post[{scope}]: {scope} != _V" if scope else "post: _ != _V"
    doc = "\n    ".join(pres + [target])
    body = (
        f"{header}def f({', '.join(sig)}):\n"
        f'    """\n    {doc}\n    """\n'
        f"    try:\n        return {expr}\n"
        f"    except Exception as _e:\n"
        f"        if _proxy(_e):\n            _mark.append(1)\n"
        f"        return None"
    )
    try:
        fn = _load(body, lib, inject=inject)
        states = {m.state for m in run_checkables(analyze_function(fn, HOLDOUT_OPTS))}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        # NOT just Exception: CrossHair internals (CrossHairInternal,
        # CrosshairUnsupported, ...) derive from ControlFlowException(BaseException)
        # and would otherwise escape and kill the whole run. Treat as "error".
        return "error"
    if marker:  # op refused the symbolic arg -> not a value function of it here
        return "unsupported"
    if MessageType.POST_FAIL in states:
        return "sat"
    if not states or states <= {MessageType.CONFIRMED}:
        return "unsat"
    return "unknown"


def _sweep(
    params: Sequence[Tuple[str, str, Any]],
    expr: str,
    header: str,
    lib: str,
    seedkey: str,
    defer_on_norun: bool = False,
    mut: bool = False,
) -> Tuple[str, str, Optional[str]]:
    """Grade how readily CrossHair can invert the operation, **per argument**.

    For each parameter in turn we pin the *others* to a fuzzed concrete input and
    ask CrossHair to solve for that one (``_invert_one``).  An op is only as
    invertible as its hardest argument -- this rules out the degenerate "set one
    arg to the whole output, no-op the rest" inversions that an all-symbolic
    search exploits.  Sweeping input size shows where each argument cliffs.

    Returns (color, verdict, demo_source):
      green   every argument inverts at every size
      yellow  the worst argument slows down past some size
      red     some argument inverts only trivially / not at all
      black   an inversion is UNSOUND -- CrossHair confirms no value of an
              argument yields an output that one demonstrably does
      "?"     not drivable / output not value-comparable / nondeterministic
    The demo is the pinned inversion of the *worst* (most interesting) argument.

    ``params`` is [(name, annotation, fuzz_spec), ...] (receiver first for
    methods); ``expr`` is the op expression; ``header`` any import lines.
    """
    sig = ", ".join(f"{n}: {ann}" for n, ann, _ in params)
    specs = [(n, spec) for n, _, spec in params]
    fwd_body = (
        f"{header}def f({sig}):\n"
        f'    """\n    post: _ != None\n    """\n'
        f"    try:\n        return {expr}\n    except Exception:\n        return None"
    )
    fwd = _load(fwd_body, lib)
    scope = params[0][0] if mut else None  # receiver name (always "a" for synth)

    # 1. forward-fuzz a valid concrete input (and its output) at each size
    samples = []  # (n, t, v)
    checked_determinism = False
    for n in SIZES:
        seed = hash(seedkey) % 1000 + n
        sample = (
            fuzz_valid_mut(specs, n, fwd, seed, seedkey)
            if mut
            else fuzz_valid(specs, n, fwd, seed, seedkey)
        )
        if sample is None:
            continue
        t, v = sample
        if defer_on_norun and _is_opaque(v):  # output can't be compared by value
            return ("?", "output not comparable", None)
        # determinism is "output is a function of the inputs"; vacuous with no
        # inputs (a zero-arg op like time.time() is invertible via its symbolic
        # return even though concrete re-runs differ), so only check when params.
        if params and not checked_determinism:  # op-level property; check once
            checked_determinism = True
            if not is_deterministic(fwd, t, v, mut):
                return ("?", "nondeterministic output", None)
        samples.append((n, t, v))
    if not samples:
        return ("?" if defer_on_norun else "red", "no valid input found", None)

    # 2. per argument, find the largest size at which CrossHair can invert it.
    # With no params (a zero-arg op) we still run one inversion of the return.
    per_arg = []  # (free_i, ok_max, errored, unsupported, ex_t, ex_v)
    for i in range(len(params) or 1):
        ok_max, misses, errored, unsupported, ex_t, ex_v = None, 0, 0, False, None, None
        for n, t, v in samples:
            out = _invert_one(header, params, expr, i, t, v, scope, lib)
            if out == "unsat":  # provably-reachable output confirmed unreachable
                return (
                    "black",
                    "unsound: confirms no input yields an output that one does",
                    _pinned_demo(header, params, expr, i, t, v, scope),
                )
            if out == "unsupported":
                # CrossHair can't accept a symbolic value here (proxy intolerance),
                # so it can only fall back to concrete inputs -- no real backward
                # reasoning.  A stable op property, so stop probing this arg; grade
                # red (never black -- the 'unsat' reading would have been bogus).
                unsupported = True
                break
            if out == "error":
                errored += 1
            elif out == "unknown":
                misses += 1
                if misses >= 2:
                    break
            elif out == "sat":
                ok_max, misses, ex_t, ex_v = n, 0, t, v
        per_arg.append((i, ok_max, errored, unsupported, ex_t, ex_v))

    # 3. the op is only as invertible as its worst argument.  Drop args whose
    # inversion never even ran (all errored) -- "couldn't check", not "not
    # invertible" -- but KEEP unsupported args: those did produce a verdict (red).
    measurable = [
        e for e in per_arg if e[3] or not (e[1] is None and e[2] >= len(samples))
    ]
    if not measurable:
        return ("?" if defer_on_norun else "red", "couldn't run", None)

    # worst = smallest ok_max; then avoid a paste-solvable echo arg; then highest
    # index (a non-receiver arg makes the more interesting demo than re-deriving the
    # receiver).  The echo term is a pure tiebreak *after* difficulty, so it never
    # changes the graded color -- only which equally-hard argument we demo.
    # Unsupported/never-inverted args (ok_max None) sort worst -> red.
    def _sort_key(e):
        i, ok_max, _err, _unsup, ex_t, ex_v = e
        echo = ex_t is not None and _is_echo(ex_t, ex_v, i)
        return (ok_max if ok_max is not None else -1, echo, -i)

    worst = min(measurable, key=_sort_key)
    wi, w_ok, _, w_unsup, w_t, w_v = worst
    w_echo = w_t is not None and _is_echo(w_t, w_v, wi)
    # A successful inversion (``w_t`` set) demos a case CrossHair SOLVES; with none,
    # fall back to a forward sample -- a runnable contract CrossHair can't satisfy
    # (its failure IS the "struggles here" demonstration).  These read very
    # differently to a user clicking through, so the demo docstring says which.
    witnessed = w_t is not None
    if w_t is None:
        _, w_t, w_v = samples[-1]

    def demo(note: str = "") -> Optional[str]:
        return _pinned_demo(header, params, expr, wi, w_t, w_v, scope, note)

    # ``[echo]`` marks a cell whose demo inverts an identity-in-that-arg op: it's
    # still a valid, solvable illustration, but paste-solvable, so it reads as a
    # worklist entry for a curated ``@pytest.mark.demo`` override.
    tag = " [echo]" if w_echo else ""
    if w_unsup:
        return (
            "red",
            "unsupported: can't accept a symbolic argument here" + tag,
            demo(
                "CrossHair can't reason about a symbolic value in this position, "
                "so it can only try fixed guesses here."
            ),
        )
    if w_ok is not None and w_ok >= SIZES[-1]:
        return ("green", "handled at every size" + tag, demo())
    if w_ok is None or w_ok <= SIZES[0]:
        note = (
            "CrossHair can solve this small case, but struggles as the inputs grow."
            if witnessed
            else "CrossHair is unlikely to find a solution to this within its time budget."
        )
        return ("red", "only the trivial case" + tag, demo(note))
    return ("yellow", f"slows down past size {w_ok}" + tag, demo())


def _repr_ok(x: Any) -> bool:
    """True if ``x`` survives an ``eval(repr(x))`` round-trip (so it can be
    pinned/asserted literally in a generated demo)."""
    try:
        return eval(repr(x), dict(ANN_NS)) == x
    except Exception:
        return False


def _pinned_demo(
    header: str,
    params: Sequence[Tuple[str, str, Any]],
    expr: str,
    free_i: int,
    t: Optional[Sequence[Any]],
    V: Any,
    scope: Optional[str],
    note: str = "",
) -> Optional[str]:
    """A runnable crosshair-web source inverting ONLY ``params[free_i]`` -- the
    other arguments are pinned by ``pre`` to their values in ``t`` (so the demo
    shows real backward reasoning, not a degenerate set-an-arg-to-the-output).
    ``note`` is an optional plain-English line prepended to the docstring to tell
    the reader what to expect (a red demo may be a small case CrossHair solves, or
    one it can't -- see :func:`_sweep`).  None if any pinned value or the target
    won't round-trip through repr."""
    if V is None or t is None:
        return None
    if not _repr_ok(V):
        return None
    sig, pres = [], []
    for j, (name, ann, _spec) in enumerate(params):
        sig.append(f"{name}: {ann}")
        if j != free_i:
            if not _repr_ok(t[j]):
                return None
            pres.append(f"pre: {name} == {t[j]!r}")
    target = f"post[{scope}]: {scope} != {V!r}" if scope else f"post: _ != {V!r}"
    lines = ([note, ""] if note else []) + pres + [target]
    doc = "\n    ".join(lines)
    return (
        f"{header}from typing import *\n\n"
        f"def f({', '.join(sig)}):\n"
        f'    """\n    {doc}\n    """\n'
        f"    return {expr}"
    )


# ---------------------------------------------------------------------------
# measure the full surface -- synthesize a call per (type, method)
#
# A demo hands us a typed signature + body for free.  For an arbitrary operation
# we synthesize the same thing: receiver `a: <type>` plus fuzzed argument(s)
# whose types come from typeshed (already a dependency).  typeshed annotations
# carry overloads / TypeVars / special forms, so we emit *candidate* signatures
# and let the fuzzer pick the one that actually produces valid inputs (a wrong
# candidate -- or a None-returning mutator -- yields no sample and is dropped).
# ---------------------------------------------------------------------------


# dunders rendered as the operator users actually write (faithful to the patched
# opcode path); anything not here falls back to a plain method call.
def _synth_candidates(
    typ: type, method: str, module: str = "builtins"
) -> List[Tuple[Any, str, str]]:
    """[(params, expr, header), ...] candidate calls for typ.method, best-effort
    first.  ``params`` is [(name, annotation, fuzz_spec), ...] (receiver first).
    ``module`` is the type's owning module; for a non-builtin class the receiver
    annotation is the qualified name and the body imports the module."""
    if method in SKIP_DUNDERS:
        return []
    recv_ann = RECV.get(typ, (f"{module}.{typ.__name__}", {}))[0]
    header = "" if module == "builtins" else f"import {module}\n"
    cands = []
    for sig in _candidate_sigs(typ, method, module):
        argnames = [n for n, _, _ in sig]
        recv = receiver_name(argnames)
        expr = call_expr(method, argnames, recv)
        if expr is None:  # operator form needs an arg the sig doesn't supply
            continue
        try:
            params = [(recv, recv_ann, _ann(recv_ann))] + [
                (n, ann, _resolve_arg(n, ann, lits, module, method))
                for n, ann, lits in sig
            ]
        except Exception:
            continue
        cands.append((params, expr, header))
    return cands


DIFF_K = 3  # valid inputs to try in the forward-soundness check
# Keep the soundness probe cheap: don't grind the full pin-search on hard-to-pin
# container shapes (an input that can't pin in budget is skipped; the holdout
# still runs).  Budget must stay high enough to pin extreme floats, which need a
# few branching iterations -- e.g. the float// bug pins at ~8 (green at 4).
DIFF_MAX_PIN_ITERS = 12


def _diff_demo(call: Any, div: Any) -> Optional[str]:
    """A runnable crosshair-web source that reproduces a forward divergence: pin
    the inputs to the divergent values and assert the CORRECT (concrete) result,
    so CrossHair visibly returns the wrong answer (or raises) on it.  Returns None
    when concrete itself raised (can't assert a value) or anything won't
    repr-round-trip."""
    if div.concrete.exc is not None:  # "should have raised" isn't a simple post
        return None
    fn, expr, names = call[0], call[1], call[2]
    ret = div.concrete.ret
    if not _repr_ok(ret):
        return None
    pins, params = [], []
    for name, val in zip(names, div.args):
        if not _repr_ok(val):
            return None
        pins.append(f"{name} == {val!r}")
        params.append(f"{name}: {type(val).__name__}")
    imp = ""
    if "_fn" in expr:  # a free function -> qualify it for a standalone module
        mod, nm = getattr(fn, "__module__", None), getattr(fn, "__name__", None)
        if not mod or not nm:
            return None
        expr = expr.replace("_fn", f"{mod}.{nm}")
        imp = f"import {mod}\n"
    return (
        f"{imp}from typing import *\n\n"
        f"def f({', '.join(params)}):\n"
        f'    """\n    pre: {" and ".join(pins)}\n    post: _ == {ret!r}\n    """\n'
        f"    return {expr}"
    )


def _diff_black(label: str, call: Any) -> Optional[Tuple[str, str, Optional[str]]]:
    """If the op's symbolic FORWARD execution disagrees with concrete Python on
    any valid input, it's unsound -> return a black ``(color, verdict, example)``.
    Else None (let the holdout decide green/yellow/red and inverse-soundness).

    This is the thorough soundness signal: the holdout only catches *inverse*
    unsoundness (falsely confirming no input yields an output), so forward bugs
    like ``float // float`` returning an int read GREEN without it.  Ops whose
    output legitimately varies (order/identity) are left to the holdout."""
    if call is None or label in NOT_VALUE_FUNCTION:
        return None
    fn, expr, names, eval_globals = call
    try:
        result = run_differential(
            fn,
            expr,
            names,
            eval_globals,
            k=DIFF_K,
            max_pin_iters=DIFF_MAX_PIN_ITERS,
            seedkey=label,
        )
    except BaseException:
        return None  # never let the soundness probe break measurement
    if result.divergence is None:
        return None
    return (
        "black",
        "unsound: forward result differs from Python",
        _diff_demo(call, result.divergence),
    )


def measure_op(
    typ: type, method: str, module: str = "builtins"
) -> Optional[Tuple[str, str, Optional[str]]]:
    """(color, verdict, example) for a (type, method) operation, via a synthesized
    call.  ``module`` is the type's owning module (``"builtins"`` for the builtin
    types, e.g. ``"decimal"`` for ``decimal.Decimal``).

    Tries the value path first (invert the return).  If every candidate defers
    (the op returns None -- likely an in-place mutator) and the receiver type is
    mutable, retry the mutation path: invert the post-call receiver state.
    ``example`` is a runnable crosshair-web source (or None).  Returns color "?"
    (with a reason) when nothing is drivable or synthesizable; only returns None
    when nothing at all could be attempted (no synthesizable call is now itself a
    labeled "?" so it renders as an explained grey cell rather than a bare gap).
    """
    cands = _synth_candidates(typ, method, module)
    if not cands:  # no drivable call -- label it so the grey cell explains itself
        reason = (
            "no signature: skipped dunder"
            if method in SKIP_DUNDERS
            else "no signature: no synthesizable call"
        )
        return ("?", reason, None)
    # Keep the builtin seedkey byte-identical (it feeds the fuzz seed); qualify it
    # with the module only for stdlib classes (where names can collide).
    seedkey = (
        f"{typ.__name__}.{method}"
        if module == "builtins"
        else f"{module}.{typ.__name__}.{method}"
    )
    black = _diff_black(seedkey, op_call(typ, method, module))  # forward-soundness
    if black:
        return black
    deferred = None
    for params, expr, header in cands:
        color, verdict, demo = _sweep(
            params, expr, header, module, seedkey, defer_on_norun=True
        )
        if color != "?":  # this candidate produced valid inputs -> trust it
            return (color, verdict, demo)
        deferred = (color, verdict, None)
    # The value path found nothing (the op returns None) -> measure it as an
    # in-place mutator, for EVERY type rather than a hardcoded mutable set.
    # fuzz_valid_mut only keeps an input whose call actually changes the receiver
    # (deepcopy pre/post compare), so an immutable type just yields no sample and
    # keeps deferring -- the mutation detection IS the gate, so we don't maintain a
    # separate list that would silently drop mutators of any unlisted type.
    for params, expr, header in cands:
        color, verdict, demo = _sweep(
            params, expr, header, module, seedkey, defer_on_norun=True, mut=True
        )
        if color != "?":
            return (color, (verdict + " [mut]").strip(), demo)
        deferred = (color, verdict, None)
    return deferred


def measure_func(module: str, func: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """(color, verdict, example) for a module-level function.  A function that is
    typeshed-known but absent from this runtime, or that has no synthesizable call,
    now returns a labeled "?" (an explained grey cell) rather than None -- so these
    show up in the JSON and reconcile with the on-screen grey count."""
    try:  # latest typeshed may stub functions absent from the running interpreter
        if not hasattr(importlib.import_module(module), func):
            return ("?", "not in runtime: absent from interpreter", None)
    except Exception:
        return ("?", "not in runtime: module import failed", None)
    cands = _func_candidate_sigs(module, func)
    if not cands:
        return ("?", "no signature: no synthesizable call", None)
    seedkey = f"{module}.{func}"
    black = _diff_black(seedkey, func_call(module, func))  # forward-soundness first
    if black:
        return black
    header = f"import {module}\n"
    drivable = []  # (params, expr) for each candidate whose arguments resolve
    for sig in cands:
        argnames = [n for n, _, _ in sig]
        expr = f"{module}.{func}({', '.join(argnames)})"
        try:
            params = [
                (n, ann, _resolve_arg(n, ann, lits, module, func))
                for n, ann, lits in sig
            ]
        except Exception:
            continue
        drivable.append((params, expr))
    if not drivable:  # every candidate raised while resolving arguments
        return ("?", "no signature: no resolvable arguments", None)
    # zero-arg funcs are still measurable: invert the RETURN (CrossHair models
    # some, e.g. time.time(), as a symbolic value), so we don't skip them.
    # (A decoder's real-encoded input, aliased pairs, etc. now come from the
    # catalog's CUSTOM_INPUTS, keyed by seedkey inside _sweep -- no per-arg
    # override here.)
    deferred = None
    for params, expr in drivable:
        color, verdict, demo = _sweep(
            params, expr, header, module, seedkey, defer_on_norun=True
        )
        if color != "?":
            return (color, verdict, demo)
        deferred = (color, verdict, None)
    # value path found nothing (the func returns None) -> measure it as an in-place
    # mutator, for EVERY func rather than a hardcoded list.  fuzz_valid_mut only
    # keeps an input whose call actually changes its first argument, so a non-mutator
    # just yields no sample and keeps deferring (mirrors measure_op).
    for params, expr in drivable:
        color, verdict, demo = _sweep(
            params, expr, header, module, seedkey, defer_on_norun=True, mut=True
        )
        if color != "?":
            return (color, (verdict + " [mut]").strip(), demo)
        deferred = (color, verdict, None)
    return deferred


# ---------------------------------------------------------------------------
# parallel execution -- ops are independent, but CrossHair's global state
# (statespace/tracing/patches) is NOT thread-safe, so we fan out over PROCESSES.
# Worker fns are module-level (picklable under spawn); scratch modules live in a
# per-process system temp dir.
# ---------------------------------------------------------------------------
def _cleanup_tmp() -> None:
    global _TMPDIR
    if _TMPDIR is not None:
        shutil.rmtree(_TMPDIR, ignore_errors=True)
        _TMPDIR = None


def _safe(call: Callable[[], Any]) -> Any:
    """Run a measurement, silencing the op's own stdout/stderr (some ops print --
    calendar.prcal, pprint, ...) and never letting any BaseException (incl.
    CrossHair's ControlFlowException internals) escape a worker."""
    try:
        with contextlib.redirect_stdout(_DEVNULL), contextlib.redirect_stderr(_DEVNULL):
            return call()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return None
    finally:
        _cleanup_tmp()


def _resolve_type(op: Any) -> Optional[type]:
    """The runtime type object for a catalogued method op (its ``owner`` name in
    the op's owning module).  None if the module/type is absent at runtime."""
    try:
        return getattr(importlib.import_module(op.module), op.owner, None)
    except Exception:
        return None


def _demo_solves(source: str, lib: str) -> bool:
    """Does CrossHair actually solve this curated demo here (find a POST_FAIL
    counterexample) under the measurement budget?  Any load/analysis failure ->
    False -> fall back to the generated demo.  Cheap: one analysis, run only for the
    handful of green/yellow/red ops that carry an override."""
    try:
        fn = _load(source, lib)
        states = {m.state for m in run_checkables(analyze_function(fn, HOLDOUT_OPTS))}
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return False
    return MessageType.POST_FAIL in states


def _with_override(op: Any, res: Any) -> Any:
    """Swap the auto-generated demo for a curated one (:mod:`demo_overrides`, best
    candidate first).  Changes ONLY the demo link, never the measured color/verdict.

    Curated demos are harvested from ``@pytest.mark.demo`` tests, which assert
    ``check_states(f, POST_FAIL)`` in CI -- so every one is *winnable* by
    construction.  We use them for green/yellow AND red cells: showing a case
    CrossHair CAN handle is a fine illustration even when the op is hard in general
    (a red cell's own generated demo often shows a solvable small case too), and we
    re-confirm the demo still solves under the measurement budget.

    BLACK is the exception -- it's unsound, and its whole point is to exhibit a
    wrong answer.  A winnable demo would HIDE that bug, so a black cell keeps its
    generated demo (the forward wrong-answer repro / false-confirmation), which a
    CI-passing test could never show.  "?" cells have no demo to improve."""
    if not res:
        return res
    color, verdict, _example = res
    if color not in ("green", "yellow", "red"):
        return res
    for source in demo_sources(op.seedkey):
        if _demo_solves(source, op.module):
            return (color, verdict, source)
    return res


def _measure_task(key: str) -> Tuple[str, str, Any]:
    """Measure ONE catalogued op, looked up by its key.  The catalog already
    settled the up-front classification, so we honor it here instead of
    re-deriving it: an op that is out of scope, a probe hazard, or reaches for I/O
    is rendered as an explained grey cell and is NEVER run concretely (that's what
    kept the concrete sweep from doing real I/O).  Only a cleanly drivable op is
    handed to the symbolic measurement.

    A cleanly measured cell then gets its demo link upgraded to a curated one when
    one is registered and stays consistent with the measured color (:func:`_with_override`).
    """
    op = _CATALOG[key]
    if op.out_of_scope:
        res: Any = ("?", f"out of scope: {op.out_of_scope}", None)
    elif op.no_inputs:
        res = ("?", f"no inputs: {op.no_inputs}", None)
    elif op.probe_hazard:
        res = ("?", f"probe hazard: {op.probe_hazard}", None)
    elif op.side_effect:
        res = ("?", f"side effect: {op.side_effect}", None)
    elif op.call is None:
        res = ("?", op.skip_reason or "no signature: no synthesizable call", None)
    elif op.kind == "method":
        typ = _resolve_type(op)
        res = (
            _safe(lambda: measure_op(typ, op.name, op.module))
            if typ is not None
            else ("?", "not in runtime: type absent from interpreter", None)
        )
    else:
        res = _safe(lambda: measure_func(op.module, op.name))
    return (op.key, op.seedkey, _safe(lambda: _with_override(op, res)) or res)


# A single op can wedge a worker in native code (CrossHair's own per-condition
# timeout can't interrupt C extensions).  We run each op in a worker subprocess
# under a hard wall-clock cap: a worker that blows past it is SIGKILLed, its op
# marked skipped, and the worker replaced -- so one bad op costs only itself, not
# the rest of the run.  (No ProcessPoolExecutor: its atexit join hangs the whole
# process forever behind a wedged native worker.)
_PER_TASK_TIMEOUT = 90
# Recycle a worker after this many ops: each measured op leaks ~10MB (z3/cache
# state), so recycle to keep a long run's per-worker RSS bounded (~150MB).
_TASKS_PER_CHILD = 8
_RECYCLE_GRACE = 5  # seconds to let a recycled worker exit before SIGKILL


def _label(t: Any) -> str:
    if isinstance(t, str):  # catalog tasks are op keys (already the identity)
        return t
    return ".".join(getattr(x, "__name__", str(x)) for x in t)


def _worker_loop(inq: Any, outq: Any, worker: Callable[[Any], Any]) -> None:
    """Pull (idx, task) items off inq, run the measurement, push (idx, result)."""
    while True:
        item = inq.get()
        if item is None:  # shutdown sentinel
            return
        idx, task = item
        try:
            outq.put((idx, worker(task)))
        except BaseException:  # worker-side crash -> skip this op
            outq.put((idx, (None, _label(task), None)))


class _Slot:
    """One worker process plus the op currently in flight on it."""

    def __init__(self) -> None:
        self.proc: Optional[Any] = None
        self.inq: Optional[Any] = None
        self.idx: Optional[int] = None  # index of in-flight task, None when idle
        self.started: float = 0.0
        self.served: int = 0  # ops completed since this process spawned


def _run_tasks(
    tasks: Sequence[Any],
    worker: Callable[[Any], Tuple[Optional[str], str, Any]],
    jobs: Optional[int],
) -> Iterator[Tuple[Optional[str], str, Any]]:
    """Yield (key, label, result) for each task.

    Every op runs in a subprocess under a hard per-op wall-clock cap, so an op
    wedged in a C extension can't stall or sink the run -- it costs one skip.
    """
    tasks = list(tasks)
    n = len(tasks)
    if n == 0:
        return
    jobs = min(jobs or 1, n) if (jobs and jobs > 0) else 1
    # forkserver (not fork): workers are forked from a clean, single-threaded
    # server, NOT from this parent -- which by now runs Queue feeder threads, and
    # forking a multi-threaded process inherits their held locks and deadlocks the
    # child.  That deadlock is exactly what wedged every op after the first worker
    # recycle.  spawn is the fallback where forkserver is unavailable (e.g. Win).
    try:
        ctx = mp.get_context("forkserver")
    except ValueError:
        ctx = mp.get_context("spawn")
    outq = ctx.Queue()

    def spawn(slot: _Slot) -> None:
        slot.inq = ctx.Queue()
        slot.proc = ctx.Process(
            target=_worker_loop, args=(slot.inq, outq, worker), daemon=True
        )
        slot.proc.start()
        slot.idx = None
        slot.served = 0

    def kill(slot: _Slot) -> None:
        if slot.proc is not None and slot.proc.is_alive():
            slot.proc.kill()  # SIGKILL reaches even a native-wedged worker
        if slot.proc is not None:
            slot.proc.join(timeout=1)
        if slot.inq is not None:
            slot.inq.cancel_join_thread()  # don't block on a queued-item feeder
            slot.inq.close()

    def retire(slot: _Slot) -> None:
        # Recycle gracefully: a healthy worker may still hold the shared outq
        # write lock (mid result-send), and SIGKILLing then orphans it -- wedging
        # every other worker's result delivery.  SIGKILL only if it won't exit.
        if slot.inq is not None:
            try:
                slot.inq.put(None)
            except Exception:
                pass
        if slot.proc is not None:
            slot.proc.join(timeout=_RECYCLE_GRACE)
            if slot.proc.is_alive():
                slot.proc.kill()
                slot.proc.join(timeout=1)
        if slot.inq is not None:
            slot.inq.cancel_join_thread()
            slot.inq.close()

    slots = [_Slot() for _ in range(jobs)]
    for s in slots:
        spawn(s)
    next_task = 0
    remaining = n
    try:
        while remaining > 0:
            # hand the next queued op to any idle worker
            for s in slots:
                if s.idx is None and next_task < n:
                    s.idx = next_task
                    s.started = time.monotonic()
                    assert s.inq is not None
                    s.inq.put((next_task, tasks[next_task]))
                    next_task += 1
            # collect finished ops (block briefly so we don't busy-spin)
            finished = []
            try:
                finished.append(outq.get(timeout=1))
                while True:
                    finished.append(outq.get_nowait())
            except _queue.Empty:
                pass
            for idx, res in finished:
                matched = False
                for s in slots:
                    if s.idx == idx:
                        matched = True
                        s.idx = None
                        s.served += 1
                        if s.served >= _TASKS_PER_CHILD:
                            retire(s)
                            spawn(s)
                        break
                if not matched:
                    # already reaped on the deadline path (finished at the wire)
                    continue
                remaining -= 1
                yield res
            # reap workers past their deadline or crashed mid-op (one skip each)
            now = time.monotonic()
            for s in slots:
                if s.idx is None:
                    continue
                assert s.proc is not None
                if now - s.started > _PER_TASK_TIMEOUT or not s.proc.is_alive():
                    lost = s.idx
                    kill(s)
                    spawn(s)
                    remaining -= 1
                    yield (None, _label(tasks[lost]), None)
    finally:
        for s in slots:
            kill(s)
        outq.cancel_join_thread()
        outq.close()


def _measure_cmd(
    args: argparse.Namespace,
    tasks: Sequence[Any],
    worker: Callable[[Any], Tuple[Optional[str], str, Any]],
    label_w: int,
) -> None:
    tally = {"green": 0, "yellow": 0, "red": 0, "black": 0, "?": 0, "skip": 0}
    out = {}
    # Track which curated demo overrides actually landed as a cell's demo, and why
    # the rest were dropped -- so a harvested demo silently going unused is visible.
    overrides = demo_overrides()
    ov_used: Dict[str, str] = {}  # seedkey -> color
    ov_drop: Dict[str, str] = {}  # seedkey -> reason
    print(f"{'op':{label_w}s} {'result':6s}  verdict")
    print("-" * (label_w + 30))
    done = 0
    for key, label, res in _run_tasks(tasks, worker, args.jobs):
        done += 1
        if res is None:
            tally["skip"] += 1
            print(f"  [{done}/{len(tasks)}] skip {label}", file=sys.stderr, flush=True)
            continue
        color, verdict, example = res
        tally[color] += 1
        if key:
            rec = {"color": color, "verdict": verdict}
            if example:  # runnable crosshair-web source for this op
                rec["example"] = example
            out[key] = rec
        if label in overrides:  # label is the op seedkey for a measured cell
            if example is not None and example in demo_sources(label):
                ov_used[label] = color
            else:
                ov_drop[label] = (
                    "black cell (kept the wrong-answer repro)"
                    if color == "black"
                    else (
                        'unmeasured "?" cell'
                        if color == "?"
                        else "curated demo did not solve within budget"
                    )
                )
        print(
            f"{label:{label_w}s} {color:6s}  {verdict}  [{done}/{len(tasks)}]",
            flush=True,  # long runs are block-buffered otherwise -> looks hung
        )
    print("-" * (label_w + 30))
    print(
        f"green={tally['green']} yellow={tally['yellow']} red={tally['red']} "
        f"black={tally['black']} defer(?)={tally['?']} skipped={tally['skip']}"
    )
    _report_overrides(ov_used, ov_drop)
    _emit(args, out)


def _report_overrides(used: Dict[str, str], dropped: Dict[str, str]) -> None:
    """Summarize curated demo-override usage for this run: how many landed as a
    cell's demo vs. were discarded (grouped by why), plus any harvested override
    that matches no operation in the catalog at all (a permanently-dead demo,
    usually a stale/renamed op -- worth fixing)."""
    exercised = len(used) + len(dropped)
    catalog_seedkeys = {op.seedkey for op in _CATALOG.values()}
    dead = sorted(k for k in demo_overrides() if k not in catalog_seedkeys)
    print(
        f"demo overrides: {len(demo_overrides())} harvested, "
        f"{exercised} exercised this run"
    )
    if exercised:
        n_used, n_drop = len(used), len(dropped)
        print(f"  used:      {n_used:3d}/{exercised} ({100 * n_used // exercised}%)")
        print(f"  discarded: {n_drop:3d}/{exercised} ({100 * n_drop // exercised}%)")
        by_reason: Dict[str, List[str]] = {}
        for seedkey, reason in sorted(dropped.items()):
            by_reason.setdefault(reason, []).append(seedkey)
        for reason, seedkeys in by_reason.items():
            print(f"    {reason} ({len(seedkeys)}): {', '.join(seedkeys)}")
    if dead:
        print(f"  no matching op in catalog ({len(dead)}): {', '.join(dead)}")


# The ONE surface -- built once at import from crosshair.inputgen.catalog and
# reused by every worker (forkserver inherits it; spawn rebuilds it on re-import).
# probe=False: static classification is complete and safe on this curated surface --
# the I/O ops are caught statically (gzip/bz2/... open -> probe_hazard; builtins
# open/print -> side_effect override) -- so we never need a live probe here.  Keyed
# by op.key (the exact key generate_treemap looks up).
_CATALOG = {op.key: op for op in catalog(probe=False)}

_ALL_TIERS = ("builtin-methods", "functions", "stdlib-methods")


def _tier(op: Any) -> str:
    """Which map tier a catalogued op belongs to -- the successor of the old
    surface / funcs / methods split, now just a VIEW over the one catalog."""
    if op.kind == "func":
        return "functions"
    return "builtin-methods" if op.module == "builtins" else "stdlib-methods"


def cmd_measure(args: argparse.Namespace) -> None:
    """Measure the whole catalog (or a scoped subset).

    One code path replaces the old surface / funcs / methods commands, which each
    kept their own drifting module list.  ``--tiers`` selects among the three map
    tiers (the docs map is ``builtin-methods,functions`` -- no stdlib class
    methods); ``--modules`` / ``--types`` scope it further, chiefly so a wedged
    module can be re-measured in its own killable invocation."""
    tiers = set(args.tiers.split(",")) if args.tiers else set(_ALL_TIERS)
    bad = tiers - set(_ALL_TIERS)
    if bad:
        raise SystemExit(f"unknown --tiers {sorted(bad)}; pick from {_ALL_TIERS}")
    modules = set(args.modules.split(",")) if args.modules else None
    types = set(args.types.split(",")) if args.types else None
    # Optional usage prefilter: only measure ops the weighted treemap would draw.
    # The catalog is ~22k ops but a weighted map draws only the few hundred whose
    # usage clears --min-weight, so measuring the rest is wasted work.  Mirror
    # generate_treemap.render_weighted's cutoff EXACTLY (same weights file, metric,
    # and raw >= min_weight test) so the measured set can't drift from the drawn one.
    used = None
    if args.weights_path:
        weights = json.loads(Path(args.weights_path).read_text())
        used = (
            lambda k: float((weights.get(k) or {}).get(args.metric, 0.0))
            >= args.min_weight
        )  # noqa: E731
    tasks = []
    for op in _CATALOG.values():
        if _tier(op) not in tiers:
            continue
        if modules is not None and op.module not in modules:
            continue
        if (
            types is not None
            and _tier(op) == "builtin-methods"
            and op.owner not in types
        ):
            continue
        if used is not None and not used(op.key):
            continue
        tasks.append(op.key)
    if used is not None:
        print(
            f"usage prefilter (--weights {args.weights_path} --metric {args.metric} "
            f"--min-weight {args.min_weight}): measuring {len(tasks)} of "
            f"{len(_CATALOG)} catalog ops"
        )
    _measure_cmd(args, tasks, _measure_task, 44)


def _emit(args: argparse.Namespace, results: Dict[str, Any]) -> None:
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {len(results)} entries to {args.json_path}")
    _cleanup_tmp()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="measure the operation catalog (or a subset)")
    m.add_argument(
        "--tiers",
        help="comma-separated map tiers to measure "
        f"(default: all of {','.join(_ALL_TIERS)})",
    )
    m.add_argument(
        "--modules",
        help="comma-separated owning-module filter (e.g. math,json); scopes "
        "functions and stdlib methods, chiefly for per-module re-measurement",
    )
    m.add_argument("--types", help="comma-separated builtin type filter (e.g. str,int)")
    m.add_argument(
        "--weights",
        dest="weights_path",
        help="usage JSON from mine_usage; measure ONLY the ops the weighted "
        "treemap would draw (usage >= --min-weight), skipping the long tail. "
        "Pass the same --weights/--metric/--min-weight you'll give generate_treemap.",
    )
    m.add_argument(
        "--metric",
        default="packages",
        choices=["packages", "sites"],
        help="usage metric for the --weights prefilter (default: packages)",
    )
    m.add_argument(
        "--min-weight",
        type=float,
        default=1.0,
        dest="min_weight",
        help="with --weights, drop ops whose usage is below this (default 1.0 = "
        "<1 package); must match generate_treemap's --min-weight",
    )
    m.add_argument("--json", dest="json_path")
    m.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel worker processes (default 1); each pins a core with a z3 "
        "solve, so keep it at or below your core count",
    )
    m.set_defaults(func=cmd_measure)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
