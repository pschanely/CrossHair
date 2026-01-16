from inspect import signature
from typing import Annotated, get_type_hints

from crosshair.tracers import ResumedTracing


def test_signature(space):
    def example_function(x: int, y: Annotated[int, "the annotation"]) -> int:
        return x + y

    sig1 = signature(example_function)
    hints1 = get_type_hints(example_function, include_extras=True)
    with ResumedTracing():
        sig2 = signature(example_function)
        hints2 = get_type_hints(example_function, include_extras=True)
        assert sig1 == sig2
        assert hints1 == hints2
