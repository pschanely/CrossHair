"""
Symbolic containers must behave like the builtin types they model.

Each case runs an operation on a symbolic input and compares how the *unrealized*
result behaves against the same operation's result on the concrete input. Probes
focus on type-sensitive behaviors, which are easy to get wrong when a result is
represented by a lazy, general-purpose structure.
"""

import copy
from array import array
from typing import Dict, FrozenSet, List, Set, Tuple

import pytest

from crosshair.core import deep_realize, proxy_for_type
from crosshair.tracers import ResumedTracing


def _outcome(thunk):
    try:
        return ("returns", thunk())
    except Exception as exc:
        return ("raises", type(exc).__name__)


def _type_name(value):
    return type(value).__name__


def _same_contents(value):
    """Containers of several builtin types holding the items of a concrete value."""
    items = list(value)
    comparands = {"list": items, "tuple": tuple(items)}
    try:
        comparands["set"] = set(items)
        comparands["frozenset"] = frozenset(items)
    except TypeError:
        pass
    if isinstance(value, dict):
        comparands["dict"] = dict(value)
    if isinstance(value, array):
        comparands["array"] = array(value.typecode, items)
    return comparands


def _probe(value, comparands):
    outcomes = {
        "type": _outcome(lambda: _type_name(value)),
        "repr": _outcome(lambda: repr(value)),
        "len": _outcome(lambda: len(value)),
        "bool": _outcome(lambda: bool(value)),
        "hashable": _outcome(lambda: hash(value) is not None),
        "* 2": _outcome(lambda: _type_name(value * 2)),
        "2 *": _outcome(lambda: _type_name(2 * value)),
    }
    for name, other in comparands.items():
        outcomes.update(
            {
                f"isinstance(_, {name})": _outcome(
                    lambda: isinstance(value, type(other))
                ),
                f"_ == {name}": _outcome(lambda: value == other),
                f"{name} == _": _outcome(lambda: other == value),
                f"_ != {name}": _outcome(lambda: value != other),
                f"_ < {name}": _outcome(lambda: value < other),
                f"_ + {name}": _outcome(lambda: _type_name(value + other)),
                f"{name} + _": _outcome(lambda: _type_name(other + value)),
                f"_ | {name}": _outcome(lambda: _type_name(value | other)),
            }
        )
    return outcomes


def _append_seven(container):
    container.append(7)
    return container


def _repeat_through_alias(container):
    alias = container
    alias *= 2
    return container


def _extend_through_alias(container):
    alias = container
    alias += [7]
    return container


CASES = [
    ("tuple", Tuple[int, ...], lambda t: t),
    ("tuple + tuple", Tuple[int, ...], lambda t: t + (7,)),
    ("tuple radd", Tuple[int, ...], lambda t: (7,) + t),
    ("tuple * int", Tuple[int, ...], lambda t: t * 2),
    ("tuple slice", Tuple[int, ...], lambda t: t[1:]),
    ("tuple step slice", Tuple[int, ...], lambda t: t[::2]),
    ("tuple concat slice", Tuple[int, ...], lambda t: (t + (7,))[1:]),
    ("list", List[int], lambda ls: ls),
    ("list + list", List[int], lambda ls: ls + [7]),
    ("list radd", List[int], lambda ls: [7] + ls),
    ("list * int", List[int], lambda ls: ls * 2),
    ("list slice", List[int], lambda ls: ls[1:]),
    ("list append", List[int], _append_seven),
    ("list *= via alias", List[int], _repeat_through_alias),
    ("list += via alias", List[int], _extend_through_alias),
    ("array", array, lambda a: a),
    ("array slice", array, lambda a: a[1:]),
    ("array append", array, _append_seven),
    ("str + str", str, lambda s: s + "x"),
    ("str slice", str, lambda s: s[1:]),
    ("bytes + bytes", bytes, lambda b: b + b"x"),
    ("bytes slice", bytes, lambda b: b[1:]),
    ("bytearray append", bytearray, _append_seven),
    ("set", Set[int], lambda s: s),
    ("set | set", Set[int], lambda s: s | {7}),
    ("set & set", Set[int], lambda s: s & {7}),
    ("set - set", Set[int], lambda s: s - {7}),
    ("frozenset", FrozenSet[int], lambda s: s),
    ("frozenset | frozenset", FrozenSet[int], lambda s: s | frozenset({7})),
    ("dict", Dict[int, int], lambda d: d),
    ("dict copy", Dict[int, int], lambda d: d.copy()),
]


@pytest.mark.parametrize("size", [0, 1, 2])
@pytest.mark.parametrize(
    "input_type,operation",
    [(typ, op) for (_, typ, op) in CASES],
    ids=[case_id for (case_id, _, _) in CASES],
)
def test_result_behaves_like_modeled_type(space, input_type, operation, size):
    symbolic_input = proxy_for_type(input_type, "x")
    with ResumedTracing():
        space.add(len(symbolic_input) == size)
        input_snapshot = copy.deepcopy(symbolic_input)
        symbolic_result = operation(symbolic_input)
    concrete_result = operation(deep_realize(input_snapshot))
    comparands = _same_contents(concrete_result)

    concrete_outcomes = _probe(concrete_result, comparands)
    with ResumedTracing():
        symbolic_outcomes = _probe(symbolic_result, comparands)
    symbolic_outcomes = deep_realize(symbolic_outcomes)
    symbolic_outcomes["realized type"] = _type_name(deep_realize(symbolic_result))
    concrete_outcomes["realized type"] = _type_name(concrete_result)

    mismatches = [
        f"  {probe}: symbolic {symbolic_outcomes[probe]!r}, concrete {expected!r}"
        for probe, expected in concrete_outcomes.items()
        if symbolic_outcomes[probe] != expected
    ]
    assert not mismatches, "\n" + "\n".join(mismatches)
