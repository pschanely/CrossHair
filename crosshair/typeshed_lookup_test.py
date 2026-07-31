"""Tests for crosshair.typeshed_lookup -- typeshed stub resolution."""

import ast
import sys

import pytest

from crosshair.typeshed_lookup import (
    class_chain,
    decorator_names,
    method_funcdefs,
    module_funcdefs,
    params,
    qualname_funcdefs,
    resolve_name,
    search_context,
    stub_class,
    stub_names,
    stub_source,
)


def test_search_context_pins_the_running_interpreter():
    ctx = search_context()
    assert ctx.version == sys.version_info[:2]
    assert ctx.platform == sys.platform


def test_missing_module_resolves_empty():
    assert stub_names("no_such_module_at_all_xyz") == {}
    assert stub_source("no_such_module_at_all_xyz") is None
    assert module_funcdefs("no_such_module_at_all_xyz") == {}


def test_overloads_are_grouped():
    defs, module = method_funcdefs("str", "center", "builtins")
    assert module == "builtins"
    assert len(defs) > 1
    assert all(isinstance(d, ast.FunctionDef) for d in defs)
    assert all("overload" in decorator_names(d) for d in defs)


@pytest.mark.parametrize(
    "module,name,expect_module",
    [
        ("bisect", "bisect_left", "_bisect"),  # re-export of a function
        ("io", "StringIO", "_io"),  # re-export of a class
    ],
)
def test_re_exports_are_followed(module, name, expect_module):
    resolved = resolve_name(name, module)
    assert resolved is not None
    assert resolved[1] == expect_module


def test_method_on_a_re_exported_class_resolves():
    """io.pyi re-exports StringIO from _io; the method still resolves."""
    defs, module = method_funcdefs("StringIO", "getvalue", "io")
    assert module == "_io"
    assert len(defs) == 1
    assert defs[0].name == "getvalue"


def test_class_chain_crosses_modules():
    chain = class_chain("Fraction", "fractions")
    assert [ni.name for ni, _ in chain][:2] == ["Fraction", "Rational"]
    # Fraction's bases live in `numbers`, not `fractions`.
    assert dict(((ni.name, mod) for ni, mod in chain))["Rational"] == "numbers"
    assert chain[-1][0].name == "object"


def test_method_resolves_up_the_chain():
    """A method typeshed declares only on a base class is found from the subclass."""
    defs, module = method_funcdefs("Fraction", "__sizeof__", "fractions")
    assert module == "builtins"
    assert len(defs) == 1


def test_stub_class_falls_back_to_builtins_and_typing():
    assert stub_class("int", "no_such_module_xyz")[1] == "builtins"
    assert stub_class("MutableSequence", "no_such_module_xyz")[1] == "typing"


# ---------------------------------------------------------------------------
# parameter splitting: the receiver goes by POSITION, not by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["__neg__", "__abs__", "__bool__"])
def test_unnamed_self_receiver_is_claimed(method):
    """fractions.pyi writes `def __neg__(a) -> Fraction`; `a` is the receiver."""
    defs, _ = method_funcdefs("Fraction", method, "fractions")
    assert defs
    parsed = params(defs[0], is_method=True)
    assert parsed.receiver is not None and parsed.receiver.arg == "a"
    assert parsed.required_positional == ()


def test_unnamed_self_receiver_leaves_the_real_argument():
    defs, _ = method_funcdefs("Fraction", "__add__", "fractions")
    assert defs
    for funcdef in defs:
        parsed = params(funcdef, is_method=True)
        assert parsed.receiver.arg == "a"
        assert [a.arg for a in parsed.required_positional] == ["b"]


def test_staticmethod_keeps_its_first_parameter():
    """str.maketrans is a @staticmethod: its leading parameter is a real argument."""
    defs, _ = method_funcdefs("str", "maketrans", "builtins")
    assert defs
    for funcdef in defs:
        parsed = params(funcdef, is_method=True)
        assert parsed.receiver is None
        assert parsed.required_positional, funcdef.name


def test_classmethod_receiver_is_claimed():
    defs, _ = method_funcdefs("dict", "fromkeys", "builtins")
    assert defs
    parsed = params(defs[0], is_method=True)
    assert parsed.receiver is not None and parsed.receiver.arg == "cls"
    assert "iterable" in [a.arg for a in parsed.required_positional]


def test_free_function_keeps_a_parameter_named_cls():
    """builtins.issubclass(cls, class_or_tuple) -- `cls` is an argument here."""
    defs = module_funcdefs("builtins")["issubclass"]
    parsed = params(defs[0], is_method=False)
    assert parsed.receiver is None
    assert [a.arg for a in parsed.required_positional] == ["cls", "class_or_tuple"]


def test_defaults_split_required_from_optional():
    defs, _ = method_funcdefs("str", "split", "builtins")
    parsed = params(defs[0], is_method=True)
    assert [a.arg for a in parsed.positional] == ["sep", "maxsplit"]
    assert parsed.required_positional == ()  # both are defaulted
    assert [a.arg for a in parsed.optional_positional] == ["sep", "maxsplit"]


def test_optional_positional_is_the_defaulted_tail():
    """re.Pattern.subn(repl, string, count=0): only ``count`` is optional."""
    defs, _ = method_funcdefs("Pattern", "subn", "re")
    parsed = params(defs[0], is_method=True)
    assert [a.arg for a in parsed.required_positional] == ["repl", "string"]
    assert [a.arg for a in parsed.optional_positional] == ["count"]


def test_keyword_only_parameters_are_exposed():
    parsed = params(module_funcdefs("json")["dumps"][0])
    kwonly = [a.arg for a in parsed.kwonly]
    assert "ensure_ascii" in kwonly and "sort_keys" in kwonly
    assert parsed.required_kwonly == ()  # json.dumps defaults all of them
    optional = [a.arg for a in parsed.optional_kwonly]
    assert "ensure_ascii" in optional and "sort_keys" in optional


def test_vararg_is_reported():
    parsed = params(module_funcdefs("math")["gcd"][0])
    assert parsed.vararg is not None
    assert parsed.required_positional == ()


def test_qualname_funcdefs_handles_module_level_and_methods():
    defs, _ = qualname_funcdefs("math", ["sqrt"])
    assert len(defs) == 1 and defs[0].name == "sqrt"
    defs, _ = qualname_funcdefs("random", ["Random", "randint"])
    assert len(defs) == 1 and defs[0].name == "randint"
    assert qualname_funcdefs("math", []) == ([], "math")
    assert qualname_funcdefs("math", ["no_such_function_xyz"]) == ([], "math")
