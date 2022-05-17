from crosshair.tracers import CompositeTracer, NoTracing, PatchingModule


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
