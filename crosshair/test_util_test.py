from crosshair.core import proxy_for_type
from crosshair.core_and_libs import standalone_statespace
from crosshair.libimpl.iolib import BackedStringIO
from crosshair.test_util import (
    UNREALIZABLE,
    compare_returns,
    flexible_equal,
    safe_deep_realize,
    summarize_execution,
)
from crosshair.tracers import ResumedTracing


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
