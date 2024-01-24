import dis

import pytest

from crosshair.tracers import (
    COMPOSITE_TRACER,
    CompositeTracer,
    CoverageTracingModule,
    PatchingModule,
    PushedModule,
    TraceSwap,
    TracingModule,
    is_tracing,
)


def overridefn(*a, **kw):
    assert a[0] == 42
    return 2


def examplefn(x: int, *a, **kw) -> int:
    return 1


def overridemethod(*a, **kw):
    assert a[1] == 42
    return 2


class Example:
    def example_method(self, a: int, **kw) -> int:
        return 1


tracer = CompositeTracer()

tracer.push_module(
    PatchingModule(
        {
            examplefn: overridefn,
            Example.__dict__["example_method"]: overridemethod,
            tuple.__len__: (lambda a: 42),
        }
    )
)


@pytest.fixture(autouse=True)
def check_tracer_state():
    assert not is_tracing()
    assert not tracer.ctracer.enabled()
    yield None
    assert not is_tracing()
    assert not tracer.ctracer.enabled()


def test_CALL_FUNCTION():
    with tracer:
        assert examplefn(42) == 2


def test_CALL_FUNCTION_KW():
    with tracer:
        assert examplefn(42, option=1) == 2


def test_CALL_FUNCTION_EX():
    with tracer:
        a = (42, 1, 2, 3)
        assert examplefn(*a, option=1) == 2


def test_CALL_METHOD():
    with tracer:
        assert Example().example_method(42) == 2


def test_override_method_in_c():
    with tracer:
        assert (1, 2, 3).__len__() == 42


def test_no_tracing():
    with tracer:
        # TraceSwap(tracer.ctracer, True) is the same as NoTracing() for `tracer`:
        with TraceSwap(tracer.ctracer, True):
            assert (1, 2, 3).__len__() == 3


def test_measure_fn_coverage() -> None:
    def called_by_foo(x: int) -> int:
        return x

    def foo(x: int) -> int:
        if called_by_foo(x) < 50:
            return x
        else:
            return (x - 50) + (called_by_foo(2 + 1) > 3) + -abs(x)

    def calls_foo(x: int) -> int:
        return foo(x)

    cov1 = CoverageTracingModule(foo)
    cov2 = CoverageTracingModule(foo)
    cov3 = CoverageTracingModule(foo)
    with COMPOSITE_TRACER:
        with PushedModule(cov1):
            calls_foo(5)

        with PushedModule(cov2):
            calls_foo(100)

        with PushedModule(cov3):
            calls_foo(5)
            calls_foo(100)

        assert 0.4 > cov1.get_results().opcode_coverage > 0.1
        assert 0.95 > cov2.get_results().opcode_coverage > 0.6
        # Note that we can't get 100% - there's an extra "return None"
        # at the end that's unreachable.
        assert cov3.get_results().opcode_coverage > 0.85


class Explode(ValueError):
    pass


class ExplodingModule(TracingModule):
    opcodes_wanted = frozenset(
        [
            dis.opmap.get("BINARY_ADD", 256),
            dis.opmap.get("BINARY_OP", 256),  # on >3.11
        ]
    )
    was_called = False

    def __call__(self, frame, codeobj, codenum):
        self.was_called = True
        raise Explode("I explode")


def test_tracer_propagates_errors():
    mod = ExplodingModule()
    COMPOSITE_TRACER.push_module(mod)
    try:
        with COMPOSITE_TRACER:
            x, y = 1, 3
            print(x + y)
    except Explode:
        pass
    else:
        assert mod.was_called
    COMPOSITE_TRACER.pop_config(mod)
