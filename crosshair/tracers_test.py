from crosshair.tracers import PatchingModule
from crosshair.tracers import CompositeTracer


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


tracer = CompositeTracer([])

tracer.add(
    PatchingModule(
        {
            examplefn: overridefn,
            Example.__dict__["example_method"]: overridemethod,
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
