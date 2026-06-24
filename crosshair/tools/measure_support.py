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

The operation surface and the valid-input generation both come from
``crosshair.inputgen`` (typeshed signatures + fuzzing).

CLI (both emit the JSON that ``generate_treemap.py`` renders):
    python -m crosshair.tools.measure_support surface [--types str,int]    --json out.json
    python -m crosshair.tools.measure_support funcs   [--modules math,json] --json out.json
"""

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from hypothesis import HealthCheck, assume, given
from hypothesis import seed as hyp_seed
from hypothesis import settings
from hypothesis import strategies as st

import crosshair.core_and_libs  # noqa: F401  -- loads opcode patches / registrations
from crosshair.core_and_libs import analyze_function, run_checkables
from crosshair.inputgen import (  # shared surface + valid-input generation
    _ROUNDTRIP,
    ANN_NS,
    RECV,
    SKIP_DUNDERS,
    TYPES,
    _ann,
    _arg_strategy,
    _candidate_sigs,
    _func_candidate_sigs,
    _module_funcs,
    _resolve_arg,
    call_expr,
    func_call,
    op_call,
    surface,
)
from crosshair.options import AnalysisOptionSet
from crosshair.statespace import MessageType
from crosshair.test_util import DIFFERENTIAL_SKIP, run_differential

_DEVNULL = open(os.devnull, "w")


# ---------------------------------------------------------------------------
# measuring an op: fuzz an input, run forward, then ask CrossHair to invert it
# ---------------------------------------------------------------------------
PRE = "from typing import *\nimport collections, datetime, itertools, json, math, random, re, time\n\n"
HOLDOUT_OPTS = AnalysisOptionSet(
    per_condition_timeout=8, max_uninteresting_iterations=200_000
)
SIZES = range(1, 6)
_ctr = 0
_TMPDIR = None  # per-process scratch dir for synthesized modules (system temp)


def _tmpdir():
    global _TMPDIR
    if _TMPDIR is None:
        _TMPDIR = Path(tempfile.mkdtemp(prefix="ch_measure_"))
    return _TMPDIR


def _load(src, lib, inject=None):
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


def fuzz_valid(args, n, forward, rngseed):
    """A VALID, non-degenerate input at size n: assume() away exceptions, and take a
    developed (not Hypothesis's first minimal) example."""
    import copy

    strat = st.tuples(*[_arg_strategy(spec, n) for _, spec in args])
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
        pre = copy.deepcopy(t)  # value-returning mutators (list.pop) mutate t;
        try:
            v = forward(*t)
        except Exception:
            assume(False)
            return
        assume(v is not None)
        found.append((pre, v))  # ... store the PRE-call input, not the mutated one

    try:
        run()
    except Exception:
        return None
    return found[-1] if found else None


def fuzz_valid_mut(args, n, forward, rngseed):
    """Like ``fuzz_valid`` but for an in-place mutator: a VALID, non-degenerate
    input at size n whose call returns None *and* actually changes the receiver
    (compared against its deepcopied original).  Captures the post-call state of
    the receiver as the holdout target."""
    import copy

    strat = st.tuples(*[_arg_strategy(spec, n) for _, spec in args])
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
        orig = copy.deepcopy(t[0])
        try:
            r = forward(*t)
        except Exception:
            assume(False)
            return
        assume(r is None)  # mutators return None
        assume(t[0] != orig)  # ... and actually mutated the receiver
        # store the PRE-call input (receiver before mutation) + the post-call
        # state as the target, so re-running it (determinism check / per-arg
        # pinning) starts from the original, not the already-mutated, receiver.
        found.append(((orig, *t[1:]), copy.deepcopy(t[0])))

    try:
        run()
    except Exception:
        return None
    return found[-1] if found else None


def _invert_one(header, params, expr, free_i, t, V, scope, lib):
    """Invert ONE argument: make param ``free_i`` symbolic, pin the rest to their
    values in ``t``, and ask CrossHair to find a ``free_i`` that makes the op
    produce ``V`` (``post: _ != V``).  This is the non-degenerate inversion -- the
    other arguments can't be set freely to absorb the output.  Pinned args are
    injected as concrete globals; a *mutated* receiver must stay a fresh per-call
    parameter, so it is pinned with a ``pre`` instead.  Returns
    'sat'/'unsat'/'unknown'/'error'."""
    sig, pres, inject = [], [], {"_V": V}
    for j, (name, ann, _spec) in enumerate(params):
        if j == free_i:
            sig.append(f"{name}: {ann}")
        elif scope and j == 0:  # mutated receiver: fresh param, pinned via pre
            sig.append(f"{name}: {ann}")
            pres.append(f"pre: {name} == _pin_{name}")
            inject[f"_pin_{name}"] = t[j]
        else:  # pinned argument: a concrete global referenced by the expression
            inject[name] = t[j]
    target = f"post[{scope}]: {scope} != _V" if scope else "post: _ != _V"
    doc = "\n    ".join(pres + [target])
    body = (
        f"{header}def f({', '.join(sig)}):\n"
        f'    """\n    {doc}\n    """\n'
        f"    try:\n        return {expr}\n"
        f"    except Exception:\n        return None"
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
    if MessageType.POST_FAIL in states:
        return "sat"
    if not states or states <= {MessageType.CONFIRMED}:
        return "unsat"
    return "unknown"


# Classes that ARE a meaningful inversion target: the canonical builtin types,
# where ``type(x) -> int`` is stable and identity-eq.  Any OTHER class is as
# uncheckable as a bare function (see _is_opaque).
_GRADABLE_CLASSES = frozenset(TYPES)


def _is_opaque(v, _depth=0):
    """True when value-comparing ``v`` is meaningless for inversion: identity-eq
    objects (functions, hash objects, file handles, ...) or containers of them.
    For these, holdout ``_ != V`` can never fail, so it would read a misleading
    red; we defer ('?') instead.

    Classes are split: a *canonical* builtin type (``type(x) -> int``) is a
    stable, identity-eq inversion target, so it stays gradable.  But an arbitrary
    class -- e.g. the per-encoding StreamReader that ``codecs.getreader`` returns
    -- is no more checkable than a bare function: we can't tell whether it
    *behaves* correctly without composing further operations, which this per-op
    map can't express.  So a non-canonical class is opaque like any callable."""
    if v is None:
        return False
    if isinstance(v, type):
        return v not in _GRADABLE_CLASSES
    if callable(v):
        return True
    if isinstance(v, (str, bytes, bytearray, range)):
        return False
    if _depth < 3:
        if isinstance(v, dict):
            return any(_is_opaque(x, _depth + 1) for x in v.values())
        if isinstance(v, (list, tuple, set, frozenset)):
            return any(_is_opaque(x, _depth + 1) for x in v)
    return type(v).__eq__ is object.__eq__


def _deterministic(fwd, t, v, mut):
    """Does the op give the same output for the same input?  If not (hash with a
    per-process seed aside, set/dict-ordering-derived values, time/random/id) the
    inversion target isn't a function of the inputs, so the verdict is meaningless.
    Recomputed on a deepcopy so an in-place mutator's input isn't disturbed."""
    import copy

    try:
        tc = copy.deepcopy(t)
        r = fwd(*tc)
        v2 = tc[0] if mut else r
        return bool(v == v2)
    except Exception:
        return True  # flaked on recompute -> don't penalize it


def _sweep(params, expr, header, lib, seedkey, defer_on_norun=False, mut=False):
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
            fuzz_valid_mut(specs, n, fwd, seed)
            if mut
            else fuzz_valid(specs, n, fwd, seed)
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
            if not _deterministic(fwd, t, v, mut):
                return ("?", "nondeterministic output", None)
        samples.append((n, t, v))
    if not samples:
        return ("?" if defer_on_norun else "red", "no valid input found", None)

    # 2. per argument, find the largest size at which CrossHair can invert it.
    # With no params (a zero-arg op) we still run one inversion of the return.
    per_arg = []  # (free_i, ok_max, errored, ex_t, ex_v)
    for i in range(len(params) or 1):
        ok_max, misses, errored, ex_t, ex_v = None, 0, 0, None, None
        for n, t, v in samples:
            out = _invert_one(header, params, expr, i, t, v, scope, lib)
            if out == "unsat":  # provably-reachable output confirmed unreachable
                return (
                    "black",
                    "unsound: confirms no input yields an output that one does",
                    _pinned_demo(header, params, expr, i, t, v, scope),
                )
            if out == "error":
                errored += 1
            elif out == "unknown":
                misses += 1
                if misses >= 2:
                    break
            elif out == "sat":
                ok_max, misses, ex_t, ex_v = n, 0, t, v
        per_arg.append((i, ok_max, errored, ex_t, ex_v))

    # 3. the op is only as invertible as its worst argument (ignoring any whose
    # inversion never even ran -- those are "couldn't check", not "not invertible")
    measurable = [e for e in per_arg if not (e[1] is None and e[2] >= len(samples))]
    if not measurable:
        return ("?" if defer_on_norun else "red", "couldn't run", None)
    # worst = smallest ok_max; ties -> highest index (a non-receiver arg makes the
    # more interesting demo than re-deriving the receiver).
    worst = min(measurable, key=lambda e: (e[1] if e[1] is not None else -1, -e[0]))
    wi, w_ok, _, w_t, w_v = worst
    # Demo the worst arg's inversion.  If it never inverted (a red), there's no
    # successful witness -- fall back to a forward sample so the cell still links
    # to a runnable contract (CrossHair failing to satisfy it IS the "struggles
    # here" demonstration).
    if w_t is None:
        _, w_t, w_v = samples[-1]
    demo = _pinned_demo(header, params, expr, wi, w_t, w_v, scope)
    if w_ok is not None and w_ok >= SIZES[-1]:
        return ("green", "handled at every size", demo)
    if w_ok is None or w_ok <= SIZES[0]:
        return ("red", "only the trivial case", demo)
    return ("yellow", f"slows down past size {w_ok}", demo)


def _pinned_demo(header, params, expr, free_i, t, V, scope):
    """A runnable crosshair-web source inverting ONLY ``params[free_i]`` -- the
    other arguments are pinned by ``pre`` to their values in ``t`` (so the demo
    shows real backward reasoning, not a degenerate set-an-arg-to-the-output).
    None if any pinned value or the target won't round-trip through repr."""
    if V is None or t is None:
        return None

    def repr_ok(x):
        try:
            return eval(repr(x), dict(ANN_NS)) == x
        except Exception:
            return False

    if not repr_ok(V):
        return None
    sig, pres = [], []
    for j, (name, ann, _spec) in enumerate(params):
        sig.append(f"{name}: {ann}")
        if j != free_i:
            if not repr_ok(t[j]):
                return None
            pres.append(f"pre: {name} == {t[j]!r}")
    target = f"post[{scope}]: {scope} != {V!r}" if scope else f"post: _ != {V!r}"
    doc = "\n    ".join(pres + [target])
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
def _synth_candidates(typ, method):
    """[(params, expr, header), ...] candidate calls for typ.method, best-effort
    first.  ``params`` is [(name, annotation, fuzz_spec), ...] (receiver first)."""
    if method in SKIP_DUNDERS:
        return []
    recv_ann = RECV[typ][0]
    cands = []
    for sig in _candidate_sigs(typ, method):
        argnames = [n for n, _, _ in sig]
        expr = call_expr(method, argnames)
        if expr is None:  # operator form needs an arg the sig doesn't supply
            continue
        try:
            params = [("a", recv_ann, _ann(recv_ann))] + [
                (n, ann, _resolve_arg(n, ann, lits, "builtins", method))
                for n, ann, lits in sig
            ]
        except Exception:
            continue
        cands.append((params, expr, ""))
    return cands


MUTABLE = (list, dict, set, bytearray)

DIFF_K = 3  # valid inputs to try in the forward-soundness check
# Keep the soundness probe cheap: don't grind the full pin-search on hard-to-pin
# container shapes (an input that can't pin in budget is skipped; the holdout
# still runs).  Budget must stay high enough to pin extreme floats, which need a
# few branching iterations -- e.g. the float// bug pins at ~8 (green at 4).
DIFF_MAX_PIN_ITERS = 12


def _diff_demo(call, div):
    """A runnable crosshair-web source that reproduces a forward divergence: pin
    the inputs to the divergent values and assert the CORRECT (concrete) result,
    so CrossHair visibly returns the wrong answer (or raises) on it.  Returns None
    when concrete itself raised (can't assert a value) or anything won't
    repr-round-trip."""
    if div.concrete.exc is not None:  # "should have raised" isn't a simple post
        return None
    fn, expr, names = call[0], call[1], call[2]
    ret = div.concrete.ret

    def repr_ok(v):
        try:
            return eval(repr(v), dict(ANN_NS)) == v
        except Exception:
            return False

    if not repr_ok(ret):
        return None
    pins, params = [], []
    for name, val in zip(names, div.args):
        if not repr_ok(val):
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


def _diff_black(label, call):
    """If the op's symbolic FORWARD execution disagrees with concrete Python on
    any valid input, it's unsound -> return a black ``(color, verdict, example)``.
    Else None (let the holdout decide green/yellow/red and inverse-soundness).

    This is the thorough soundness signal: the holdout only catches *inverse*
    unsoundness (falsely confirming no input yields an output), so forward bugs
    like ``float // float`` returning an int read GREEN without it.  Ops whose
    output legitimately varies (order/identity) are left to the holdout."""
    if call is None or label in DIFFERENTIAL_SKIP:
        return None
    fn, expr, names, eval_globals = call
    try:
        result = run_differential(
            fn, expr, names, eval_globals, k=DIFF_K, max_pin_iters=DIFF_MAX_PIN_ITERS
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


def measure_op(typ, method):
    """(color, verdict, example) for a builtin operation, via a synthesized call.

    Tries the value path first (invert the return).  If every candidate defers
    (the op returns None -- likely an in-place mutator) and the receiver type is
    mutable, retry the mutation path: invert the post-call receiver state.
    ``example`` is a runnable crosshair-web source (or None).  Returns color "?"
    when nothing is drivable, and None when nothing could be synthesized.
    """
    cands = _synth_candidates(typ, method)
    if not cands:
        return None
    seedkey = f"{typ.__name__}.{method}"
    black = _diff_black(seedkey, op_call(typ, method))  # forward-soundness first
    if black:
        return black
    deferred = None
    for params, expr, header in cands:
        color, verdict, demo = _sweep(
            params, expr, header, "builtins", seedkey, defer_on_norun=True
        )
        if color != "?":  # this candidate produced valid inputs -> trust it
            return (color, verdict, demo)
        deferred = (color, verdict, None)
    if typ in MUTABLE:  # value path found nothing -> measure as an in-place mutator
        for params, expr, header in cands:
            color, verdict, demo = _sweep(
                params, expr, header, "builtins", seedkey, defer_on_norun=True, mut=True
            )
            if color != "?":
                return (color, (verdict + " [mut]").strip(), demo)
            deferred = (color, verdict, None)
    return deferred


def measure_func(module, func):
    """(color, verdict, example) for a module-level function, or None if nothing
    could be synthesized / it isn't in this runtime."""
    try:  # latest typeshed may stub functions absent from the running interpreter
        if not hasattr(importlib.import_module(module), func):
            return None
    except Exception:
        return None
    cands = _func_candidate_sigs(module, func)
    if not cands:
        return None
    seedkey = f"{module}.{func}"
    black = _diff_black(seedkey, func_call(module, func))  # forward-soundness first
    if black:
        return black
    deferred = None
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
        # zero-arg funcs are still measurable: invert the RETURN (CrossHair models
        # some, e.g. time.time(), as a symbolic value), so we don't skip them.
        rt = _ROUNDTRIP.get((module, func))  # feed a decoder a real encoded input
        if rt:
            params[0] = (params[0][0], params[0][1], ("roundtrip",) + rt)
        color, verdict, demo = _sweep(
            params, expr, f"import {module}\n", module, seedkey, defer_on_norun=True
        )
        if color != "?":
            return (color, verdict, demo)
        deferred = (color, verdict, None)
    return deferred


def func_surface(module):
    """Public free-function names of a module that the funcs measurement targets:
    typeshed-known, public, present + callable at runtime, and not a type
    (type constructors are covered by the type surface, not here)."""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return []
    out = []
    for name in sorted(_module_funcs(module)):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        if obj is None or not callable(obj) or isinstance(obj, type):
            continue
        out.append(name)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# parallel execution -- ops are independent, but CrossHair's global state
# (statespace/tracing/patches) is NOT thread-safe, so we fan out over PROCESSES.
# Worker fns are module-level (picklable under spawn); scratch modules live in a
# per-process system temp dir.
# ---------------------------------------------------------------------------
def _cleanup_tmp():
    global _TMPDIR
    if _TMPDIR is not None:
        shutil.rmtree(_TMPDIR, ignore_errors=True)
        _TMPDIR = None


def _safe(call):
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


def _op_task(item):
    typ, meth = item
    res = _safe(lambda: measure_op(typ, meth))
    return (f"builtins.{typ.__name__}_{meth}_method", f"{typ.__name__}.{meth}", res)


def _func_task(item):
    module, name = item
    res = _safe(lambda: measure_func(module, name))
    return (f"{module}.{name}", f"{module}.{name}", res)


# A single op occasionally hangs a worker in native code (CrossHair's own
# per-condition timeout can't interrupt C extensions).  If no task finishes for
# this many seconds we give up on whatever is still running, mark it skipped, and
# move on -- otherwise one bad op would block the whole run and lose the results.
_STALL_TIMEOUT = 90


def _label(t):
    return ".".join(getattr(x, "__name__", str(x)) for x in t)


def _run_tasks(tasks, worker, jobs):
    """Yield (key, label, result) for each task, in parallel when jobs > 1."""
    if not jobs or jobs <= 1:
        for t in tasks:
            yield worker(t)
        return
    # Recycle workers periodically: a long symbolic run accumulates memory, and
    # without recycling the pool eventually OOMs (losing the whole run). 3.11+ only.
    kwargs = {"max_workers": jobs}
    if sys.version_info >= (3, 11):
        kwargs["max_tasks_per_child"] = 8
    # NOT a `with` block: a wedged worker in native code (z3) makes the context
    # manager's shutdown(wait=True) hang forever.  Manage it manually and always
    # tear down without waiting, killing any (possibly respawning) workers.
    ex = ProcessPoolExecutor(**kwargs)
    try:
        futs = {ex.submit(worker, t): t for t in tasks}
        pending = set(futs)
        last_progress = time.monotonic()
        while pending:
            try:
                done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            except BaseException:  # pool broken (worker segfault/OOM) -> salvage
                for fut in pending:
                    yield (None, _label(futs[fut]), None)
                break
            if done:
                last_progress = time.monotonic()
                for fut in done:
                    try:
                        yield fut.result()
                    except Exception:  # worker died (e.g. solver crash) -> skip
                        yield (None, _label(futs[fut]), None)
            elif time.monotonic() - last_progress > _STALL_TIMEOUT:
                # everything still pending is wedged; report as skipped and bail.
                for fut in pending:
                    yield (None, _label(futs[fut]), None)
                break
    finally:
        for proc in list(ex._processes.values()):
            proc.kill()  # don't let a wedged native worker block teardown
        ex.shutdown(wait=False, cancel_futures=True)


def _measure_cmd(args, tasks, worker, label_w):
    tally = {"green": 0, "yellow": 0, "red": 0, "black": 0, "?": 0, "skip": 0}
    out = {}
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
        print(f"{label:{label_w}s} {color:6s}  {verdict}  [{done}/{len(tasks)}]")
    print("-" * (label_w + 30))
    print(
        f"green={tally['green']} yellow={tally['yellow']} red={tally['red']} "
        f"black={tally['black']} defer(?)={tally['?']} skipped={tally['skip']}"
    )
    _emit(args, out)


def cmd_surface(args):
    """Measure every builtin (type, method) pair via synthesized calls."""
    wanted = set(args.types.split(",")) if args.types else None
    tasks = [
        (typ, meth)
        for typ in TYPES
        if not wanted or typ.__name__ in wanted
        for meth in surface(typ)
    ]
    _measure_cmd(args, tasks, _op_task, 32)


# A broad, pure (no side effects on import/call) set of stdlib modules to measure
# free functions in -- the default corpus for the `funcs` command and the docs map.
STDLIB_MODULES = (
    "builtins,math,cmath,statistics,decimal,fractions,numbers,random,secrets,string,"
    "textwrap,re,unicodedata,stringprep,difflib,shlex,html,json,base64,binascii,quopri,"
    "zlib,gzip,bz2,lzma,hashlib,hmac,itertools,functools,operator,heapq,bisect,"
    "collections,copy,reprlib,pprint,calendar,colorsys,fnmatch,posixpath,ntpath,"
    "genericpath,urllib.parse,ipaddress,uuid,struct,ast,graphlib,codecs,time"
).split(",")


def cmd_funcs(args):
    """Measure module-level (free) functions in the given stdlib modules."""
    modules = args.modules.split(",") if args.modules else STDLIB_MODULES
    tasks = [
        (module, name)
        for module in modules
        for name in sorted(_module_funcs(module))
        if not name.startswith("_")
    ]
    _measure_cmd(args, tasks, _func_task, 36)


def _emit(args, results):
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"wrote {len(results)} entries to {args.json_path}")
    _cleanup_tmp()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("surface", help="measure every builtin (type, method) pair")
    s.add_argument("--types", help="comma-separated builtin type filter (e.g. str,int)")
    s.add_argument("--json", dest="json_path")
    s.add_argument(
        "--jobs", type=int, default=1, help="parallel worker processes (default 1)"
    )
    s.set_defaults(func=cmd_surface)
    fn = sub.add_parser("funcs", help="measure module-level (free) functions")
    fn.add_argument(
        "--modules",
        help="comma-separated module list "
        "(default: the curated stdlib set in STDLIB_MODULES)",
    )
    fn.add_argument("--json", dest="json_path")
    fn.add_argument(
        "--jobs", type=int, default=1, help="parallel worker processes (default 1)"
    )
    fn.set_defaults(func=cmd_funcs)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
