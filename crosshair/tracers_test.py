from crosshair.tracers import (
    COMPOSITE_TRACER,
    CompositeTracer,
    CoverageTracingModule,
    NoTracing,
    PatchingModule,
    PushedModule,
)


def overridefn(*a, **kw):
    assert a[0] == 42
    return 2


def examplefn(x: int, *a, **kw) -> int:
    return 1


def overridemethod(*a, **kw):
    # assert type(a[0]) is Example
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
        with NoTracing():
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

    with COMPOSITE_TRACER:
        with PushedModule(cov := CoverageTracingModule(foo)):
            calls_foo(5)
        print("cov.get_results()", cov.get_results())
        assert 0.4 > cov.get_results().opcode_coverage > 0.1

        with PushedModule(cov := CoverageTracingModule(foo)):
            calls_foo(100)
        assert 0.95 > cov.get_results().opcode_coverage > 0.6

        with PushedModule(cov := CoverageTracingModule(foo)):
            calls_foo(5)
            calls_foo(100)
        # Note that we can't get 100% - there's an extra "return None"
        # at the end that's unreachable.
        assert cov.get_results().opcode_coverage > 0.85
