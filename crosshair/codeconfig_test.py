import importlib
import inspect
import sys
import textwrap
from pathlib import Path

import pytest  # type: ignore

from crosshair.codeconfig import (
    AnalysisOptionSet,
    InvalidDirective,
    collect_options,
    get_directives,
    parse_directives,
)
from crosshair.test_util import simplefs
from crosshair.util import add_to_pypath


# crosshair: off
def override_on():
    # crosshair: on
    pass


def timeout_of_10():
    # crosshair: per_condition_timeout=10
    pass


def _example1():
    # crosshair : First comment
    # does not lead with crosshair: not present
    print("# crosshair : this is a string, not a comment")
    pass  # crosshair:comment with trailing space


def test_get_directives_example1() -> None:
    lines, _ = inspect.getsourcelines(_example1)
    assert get_directives("".join(lines)) == [
        (2, 4, "First comment"),
        (5, 10, "comment with trailing space"),
    ]


def test_parse_directives() -> None:
    assert parse_directives([(1, 0, "per_condition_timeout=42")]) == AnalysisOptionSet(
        per_condition_timeout=42
    )
    assert parse_directives([(1, 0, "on")]) == AnalysisOptionSet(enabled=True)
    assert parse_directives([(1, 0, "off")]) == AnalysisOptionSet(enabled=False)
    assert parse_directives([(1, 0, "on per_path_timeout=42")]) == AnalysisOptionSet(
        enabled=True, per_path_timeout=42
    )


def test_parse_directive_errors() -> None:
    with pytest.raises(InvalidDirective, match='Malformed option: "noequals"'):
        parse_directives([(1, 0, "noequals")])
    with pytest.raises(InvalidDirective, match='Unknown option: "notafield"'):
        parse_directives([(1, 0, "notafield=42")])
    with pytest.raises(
        InvalidDirective,
        match='"notanumber" is not a valid "per_condition_timeout" value',
    ):
        parse_directives([(1, 0, "per_condition_timeout=notanumber")])
    with pytest.raises(
        InvalidDirective,
        match='Option "enabled" is set multiple times at the same scope',
    ):
        parse_directives([(1, 0, "on off")])


def test_get_directives_multiline_string() -> None:
    # This tests a regression where we'd previously thrown a "EOF in multi-line string"
    # TokenError.
    foo = '''"""

"""
'''
    assert list(get_directives(foo)) == []


def test_collection_options() -> None:
    this_module = sys.modules[__name__]
    assert collect_options(this_module) == AnalysisOptionSet(enabled=False)
    assert collect_options(override_on) == AnalysisOptionSet(enabled=True)
    assert collect_options(timeout_of_10) == AnalysisOptionSet(
        enabled=False, per_condition_timeout=10
    )


DIRECTIVES_TREE = {
    "pkg1": {
        "__init__.py": textwrap.dedent(
            """\
            # crosshair: off
            # crosshair: per_condition_timeout=42
            """
        ),
        "pkg2": {
            "pkg3": {
                "__init__.py": "# crosshair: max_iterations=5",
                "mod.py": "# crosshair: on",
            }
        },
    }
}


def test_package_directives(tmp_path: Path):
    simplefs(tmp_path, DIRECTIVES_TREE)
    with add_to_pypath(tmp_path):
        innermod = importlib.import_module("pkg1.pkg2.pkg3.mod")
        assert collect_options(innermod) == AnalysisOptionSet(
            enabled=True, max_iterations=5, per_condition_timeout=42
        )
