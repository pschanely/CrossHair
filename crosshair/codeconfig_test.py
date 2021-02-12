import inspect

import pytest  # type: ignore

from crosshair.codeconfig import *


def _example1():
    # crosshair : First comment
    print("# crosshair : this is a string, not a comment")
    pass  # crosshair:comment with trailing space


def test_get_directives_example1() -> None:
    lines, _ = inspect.getsourcelines(_example1)
    assert get_directives(lines) == [
        (2, "First comment"),
        (4, "comment with trailing space"),
    ]


def test_parse_directives() -> None:
    assert parse_directives([(1, "per_condition_timeout=42")]) == AnalysisOptionSet(
        per_condition_timeout=42
    )
    assert parse_directives([(1, "on")]) == AnalysisOptionSet(enabled=True)
    assert parse_directives([(1, "off")]) == AnalysisOptionSet(enabled=False)
    assert parse_directives([(1, "on per_path_timeout=42")]) == AnalysisOptionSet(
        enabled=True, per_path_timeout=42
    )


def test_parse_directive_errors() -> None:
    with pytest.raises(InvalidDirective, match='Malformed option: "noequals"'):
        parse_directives([(1, "noequals")])
    with pytest.raises(InvalidDirective, match='Unknown option: "notafield"'):
        parse_directives([(1, "notafield=42")])
    with pytest.raises(
        InvalidDirective,
        match='"notanumber" is not a valid "per_condition_timeout" value',
    ):
        parse_directives([(1, "per_condition_timeout=notanumber")])
    with pytest.raises(
        InvalidDirective,
        match='Option "enabled" is set multiple times at the same scope',
    ):
        parse_directives([(1, "on off")])
