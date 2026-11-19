import itertools
from collections import Counter

import pytest

from crosshair.behavior_compare import (
    UNREALIZABLE,
    CallSpec,
    SkipReason,
    compare_returns,
    flexible_equal,
    run_differential,
    run_symbolic_pinned,
    safe_deep_realize,
    summarize_execution,
)
from crosshair.core import deep_realize, proxy_for_type
from crosshair.core_and_libs import standalone_statespace
from crosshair.libimpl.iolib import BackedStringIO
from crosshair.tracers import ResumedTracing
from crosshair.util import CrossHairInternal


def test_flexible_equal():
    assert float("nan") != float("nan")
    assert flexible_equal(float("nan"), float("nan"))
    assert flexible_equal((42, float("nan")), (42, float("nan")))
    assert not flexible_equal([float("nan"), 11], [float("nan"), 22])

    def gen():
        yield 11
        yield 22

    assert flexible_equal(gen(), iter([11, 22]))
    assert not flexible_equal(gen(), iter([11, 22, 33]))
    assert not flexible_equal(gen(), iter([11]))

    ordered_set_1 = {10_000, 20_000} | {30_000}
    ordered_set_2 = {30_000, 20_000} | {10_000}
    assert list(ordered_set_1) != list(ordered_set_2)  # (different orderings)
    assert flexible_equal(ordered_set_1, ordered_set_2)

    ordered_dict_1 = {1: 2, 3: 4}
    ordered_dict_2 = {3: 4, 1: 2}
    assert list(ordered_dict_1.items()) != list(ordered_dict_2.items())
    assert flexible_equal(ordered_dict_1, ordered_dict_2)


def test_summarize_execution_realizes_post_args_on_exception():
    def raiser(x):
        raise ValueError

    with standalone_statespace:
        symbolic_int = proxy_for_type(int, "i")
        with ResumedTracing():
            result = summarize_execution(raiser, (symbolic_int,))
    assert type(result.exc) is ValueError
    assert type(result.post_args[0]) is int


def test_compare_returns_handles_closed_stream_post_args():
    """End-to-end smoke test: a function that closes its IO stream and then
    raises during a follow-up operation should still produce a clean
    comparison via `compare_returns` (which discards post_args)."""

    def close_and_truncate(s: BackedStringIO):
        s.close()
        return s.truncate()

    with standalone_statespace:
        stream = BackedStringIO("abc")
        with ResumedTracing():
            result = compare_returns(close_and_truncate, stream)
    assert bool(result)


def test_summarize_execution_uses_sentinel_for_unrealizable_post_args():
    class Stubborn:
        def __reduce__(self):
            raise RuntimeError("intentionally unrealizable")

    def consume(s):
        return None

    with standalone_statespace:
        with ResumedTracing():
            result = summarize_execution(consume, (Stubborn(),))
    assert result.post_args[0] is UNREALIZABLE


def test_safe_deep_realize_returns_sentinel_on_failure():
    class Stubborn:
        def __reduce__(self):
            raise RuntimeError("nope")

    assert safe_deep_realize(Stubborn()) is UNREALIZABLE


def test_flexible_equal_treats_unrealizable_as_equal():
    assert flexible_equal(UNREALIZABLE, 42)
    assert flexible_equal("anything", UNREALIZABLE)
    assert flexible_equal((1, UNREALIZABLE, 3), (1, "different", 3))
    assert flexible_equal({"k": UNREALIZABLE}, {"k": object()})


# Nested/heap-backed containers whose elements cannot be reduced to a single SMT
# equality: ``run_symbolic_pinned`` must still drive a symbolic proxy to each,
# pinning it by structural descent rather than spinning until the budget runs out.
# (A dict backed by an SMT array stores nested values behind HeapRefs, so
# ``smt_for_unification`` returns None and pinning has to fork.)
@pytest.mark.parametrize(
    "value",
    [
        {1: {2: 3}},
        {"a": [1, 2]},
        [[1, 2], [3]],
        ([1], [2]),
        [(1, 2), (3, 4)],
    ],
)
def test_run_symbolic_pinned_descends_into_nested_containers(value):
    captured = []

    def grab(x):
        captured.append(deep_realize(x))
        return x

    result = run_symbolic_pinned(grab, ["x"], [value])
    assert not isinstance(result, SkipReason)
    assert captured[-1] == value


def test_run_differential_reports_pin_budget_exhaustion():
    call = CallSpec(abs, "abs(a)", ("a",), {}, 0)
    result = run_differential(call, [(-1,), (0,), (1,)], max_pin_iters=0)
    assert result.checked == 0
    assert result.skipped == Counter({SkipReason.UNPINNED: 3})
    assert result.divergence is None


def test_run_differential_keeps_counts_before_divergence(monkeypatch):
    from crosshair import behavior_compare

    call = CallSpec(abs, "abs(a)", ("a",), {}, 0)
    wrong = summarize_execution(lambda a: 99, [1], detach_path=False)
    results = iter([SkipReason.UNPINNED, SkipReason.UNSUPPORTED, wrong])
    monkeypatch.setattr(
        behavior_compare, "run_symbolic_pinned", lambda *args: next(results)
    )
    result = run_differential(call, [(1,), (1,), (1,), (1,)])
    assert result.checked == 1
    assert result.skipped == Counter(
        {SkipReason.UNPINNED: 1, SkipReason.UNSUPPORTED: 1}
    )
    assert result.divergence is not None
    assert result.divergence.args == (1,)


def _raise_internal(a):
    raise CrossHairInternal("boom")


_OPAQUE = object()
_COUNTER = itertools.count()


@pytest.mark.parametrize(
    "fn,reason",
    [
        (_raise_internal, SkipReason.ERRORED),
        (lambda a: next(_COUNTER), SkipReason.NONDETERMINISTIC),
        (lambda a: _OPAQUE, SkipReason.OPAQUE),
    ],
)
def test_run_differential_counts_skip_reasons(fn, reason):
    call = CallSpec(fn, "f(a)", ("a",), {"f": fn}, 0)
    result = run_differential(call, [(1,), (2,)])
    assert result.checked == 0
    assert result.skipped == Counter({reason: 2})
    assert result.divergence is None
