import json
from typing import List

import pytest

from crosshair.core import proxy_for_type, realize
from crosshair.core_and_libs import NoTracing, standalone_statespace
from crosshair.statespace import POST_FAIL
from crosshair.test_util import check_states
from crosshair.tracers import ResumedTracing


def test_disallow_unicode_digits():
    with standalone_statespace:
        float("0E٠")  # This is a valid float!
        with pytest.raises(json.JSONDecodeError):
            json.loads("0E٠")  # But not a valid JSON float.


@pytest.mark.demo("yellow")
def test_dumps():
    def f(lst: List[int]):
        """
        Can a JSON-serialized list be larger than 10 characters?

        NOTE: Although this example is fast, most JSON serialization
        tasks require at least a few minutes of analysis, and many may
        not be solvable in any reasonable time frame.

        post: len(_) <= 10
        """
        return json.dumps(lst)

    check_states(f, POST_FAIL)


@pytest.mark.demo("yellow")
def test_loads():
    def f(s: str):
        """
        Can we parse an empty JSON array out of a 3 character string?

        NOTE: Although this example is fast, most JSON deserialization
        tasks require at least a few minutes of analysis, and many may
        not be solvable in any reasonable time frame.

        pre: len(s) == 3
        raises: json.JSONDecodeError
        post: _ != []
        """
        return json.loads(s)

    check_states(f, POST_FAIL)


@pytest.mark.parametrize(
    "doc,pos",
    [
        ("abc", 1),
        ("ab\ncd", 4),
        ("ab\ncd", 2),
        ("\n\n\nx", 3),
        ("a\nb\nc", 5),
        ("a\nb", 9),
        ("", 0),
    ],
)
def test_decode_error_on_symbolic_string_matches_concrete(doc, pos):
    expected = json.JSONDecodeError("Expecting value", doc, pos)
    with standalone_statespace as space, NoTracing():
        symbolic_doc = proxy_for_type(str, "doc")
        with ResumedTracing():
            space.add(symbolic_doc == doc)
            error = json.JSONDecodeError("Expecting value", symbolic_doc, pos)
        assert realize(error.lineno) == expected.lineno
        assert realize(error.colno) == expected.colno
        assert realize(str(error)) == str(expected)
        assert realize(error.args[0]) == expected.args[0]
