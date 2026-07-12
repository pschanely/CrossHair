"""Tests for the curated demo-override harvester.

The ``@pytest.mark.demo`` tests themselves are self-verifying (each asserts its
``check_states`` behavior in its own libimpl ``*lib_test`` module).  These tests
guard the HARVEST -- that the name->seedkey convention still holds, that every
demo yields a self-contained, parseable crosshair-web source, and that the keys
line up with what ``measure_support`` looks up -- so the map's demo links can't
silently rot when a demo is renamed or the convention drifts."""

import ast
import re

import pytest

# a crosshair contract line: pre:/post:, the latter optionally scoped (post[ls]:)
_CONTRACT_RE = re.compile(r"\b(?:pre|post(?:\[[^\]]*\])?):")

from crosshair.tools.demo_overrides import (
    _seedkey,
    _source,
    demo_overrides,
    demo_sources,
)


def test_harvest_is_populated():
    ov = demo_overrides()
    # dozens of @pytest.mark.demo tests exist across the libimpl test modules
    assert len(ov) >= 40


@pytest.mark.parametrize(
    "module,test,expected",
    [
        ("builtins", "test_int___add___method", "int.__add__"),
        ("builtins", "test_str_replace_method", "str.replace"),
        ("builtins", "test_float___pow___operator", "float.__pow__"),
        ("builtins", "test_sorted", "builtins.sorted"),
        ("json", "test_dumps", "json.dumps"),
        ("collections", "test_deque_extendleft_method", "collections.deque.extendleft"),
        ("datetime", "test_timedelta___add___method", "datetime.timedelta.__add__"),
    ],
)
def test_seedkey_mapping(module, test, expected):
    assert _seedkey(module, test) == expected


def test_known_ops_are_covered():
    ov = demo_overrides()
    for seedkey in ("int.__add__", "str.replace", "str.join", "json.dumps"):
        assert seedkey in ov, f"expected a curated demo for {seedkey}"


def test_every_source_is_self_contained_and_has_a_contract():
    for seedkey, candidates in demo_overrides().items():
        for _color, source in candidates:
            # parses as a module and defines exactly the demo function ``f``
            tree = ast.parse(source)
            fns = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            assert fns == ["f"], f"{seedkey}: expected one def f, got {fns}"
            # carries a contract crosshair can act on
            assert _CONTRACT_RE.search(source), f"{seedkey}: no contract"


def test_source_imports_typing_and_owning_module():
    # a builtins demo needs only typing; a stdlib demo must also import its module
    assert _source("builtins", "def f(): pass\n").startswith("from typing import *")
    stdlib = _source("json", "def f(): pass\n")
    assert "import json" in stdlib and "from typing import *" in stdlib


def test_demo_sources_helper_drops_colors():
    both = demo_overrides()["int.__add__"]
    assert demo_sources("int.__add__") == [src for _c, src in both]
    assert demo_sources("no.such.op") == []
