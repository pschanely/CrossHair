import gc
import sys

import pytest

from _crosshair_tracers import CTracer  # type: ignore


class ExampleModule:
    opcodes_wanted = frozenset([42, 255])


def test_CTracer_module_refcounts_dont_leak():
    mod = ExampleModule()
    assert sys.getrefcount(mod) == 2
    tracer = CTracer()
    tracer.push_module(mod)
    assert sys.getrefcount(mod) == 3
    tracer.push_module(mod)
    tracer.start()
    tracer.stop()
    assert sys.getrefcount(mod) == 4
    tracer.pop_module(mod)
    assert sys.getrefcount(mod) == 3
    del tracer
    gc.collect()
    assert sys.getrefcount(mod) == 2


class Explode(ValueError):
    pass


class ExplodingModule:
    opcodes_wanted = frozenset([23, 122])  # (BINARY_ADD, BINARY_OP on >3.11)

    def __call__(self, frame, codeobj, codenum, extra):
        raise Explode("I explode")


def test_CTracer_propagates_errors():
    mod = ExplodingModule()
    tracer = CTracer()
    tracer.push_module(mod)
    try:
        tracer.start()
        x, y = 1, 3
        print(x + y)
    except Explode:
        tracer.stop()
        tracer.pop_module(mod)
    else:
        assert False


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="getrusage not available on windows"
)
def test_CTracer_does_not_leak_memory():
    import resource  # (available only on unix; delay import)

    for i in range(1_000):
        tracer = CTracer()
        tracer.start()
        mods = [ExampleModule() for _ in range(6)]
        for mod in mods:
            tracer.push_module(mod)
        for mod in reversed(mods):
            tracer.pop_module(mod)
        tracer.stop()
        if i == 100:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    usage_increase = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - usage
    if sys.platform == "darwin":
        usage_increase /= 1024  # (it's bytes on osx)
    assert usage_increase < 25
