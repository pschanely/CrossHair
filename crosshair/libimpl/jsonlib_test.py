import json
from typing import List

import pytest

from crosshair.core_and_libs import standalone_statespace
from crosshair.statespace import POST_FAIL
from crosshair.test_util import check_states


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
