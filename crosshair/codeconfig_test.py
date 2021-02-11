import inspect

import pytest  # type: ignore

from crosshair.codeconfig import *


def _example1():
    # crosshair : First comment
    print("# crosshair : this is a string, not a comment")
    pass  # crosshair:comment with trailing space


def test_get_directives_example1():
    lines, _ = inspect.getsourcelines(_example1)
    assert get_directives(lines) == [
        (2, "First comment"),
        (4, "comment with trailing space"),
    ]
