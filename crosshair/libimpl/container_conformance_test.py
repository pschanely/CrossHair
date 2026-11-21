"""
Symbolic containers must behave like the builtin types they model.

Each case runs an operation on a symbolic input and compares how the *unrealized*
result behaves against the same operation's result on the concrete input. Probes
focus on type-sensitive behaviors, which are easy to get wrong when a result is
represented by a lazy, general-purpose structure.

Two sources of cases live here:

* ``CASES`` -- hand-written operations, each pinned to symbolic inputs of size
  0, 1 and 2. The size sweep matters: empty and single-item containers often
  behave differently from larger ones, and catalog inputs do not reliably
  include them.
* The operation catalog (``crosshair.inputgen``) -- every catalogued method on
  a container type, driven with catalog inputs. Coverage grows with the catalog
  instead of by hand (issue #518, part 1). An op only contributes when a
  catalog input's concrete result is a container; other inputs are skipped.
"""

import copy
from array import array
from collections import deque
from typing import Dict, FrozenSet, List, Set, Tuple

import pytest

pytest.importorskip("hypothesis")  # imported by crosshair.inputgen

import crosshair.core_and_libs  # noqa: F401  -- ensure patches/plugins load
from crosshair.behavior_compare import SkipReason, run_symbolic_pinned
from crosshair.core import deep_realize, proxy_for_type
from crosshair.inputgen import catalog, inputs_for
from crosshair.tracers import NoTracing, ResumedTracing


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


def _inner_values(container):
    return container.values() if isinstance(container, dict) else container


def _append_seven_inside(container):
    for inner in _inner_values(container):
        inner.append(7)
    return container


def _pop_inside(container):
    for inner in _inner_values(container):
        if inner:
            inner.pop()
    return container


def _toggle_seven_inside(container):
    for inner in _inner_values(container):
        inner ^= {7}
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
    ("nested list", List[List[int]], lambda ls: ls),
    ("nested list inner append", List[List[int]], _append_seven_inside),
    ("nested list inner pop", List[List[int]], _pop_inside),
    ("tuple of lists inner append", Tuple[List[int], ...], _append_seven_inside),
    ("list of sets inner toggle", List[Set[int]], _toggle_seven_inside),
    ("dict of lists inner append", Dict[int, List[int]], _append_seven_inside),
    ("dict of sets inner toggle", Dict[int, Set[int]], _toggle_seven_inside),
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


# ---------------------------------------------------------------------------
# Catalog-driven cases (issue #518, part 1): the probe battery above, run on
# every catalogued method on a container type, with catalog inputs.  The
# symbolic run is pinned to each concrete input, and the probe runs INSIDE the
# pinned run (as the applier), so it observes the unrealized result in the
# same StateSpace that produced it -- probing the result afterwards, under a
# different StateSpace, does not see the pinned values.
# ---------------------------------------------------------------------------

# Catalog inputs checked per operation. Small for CI: the pinned run dominates
# the cost, and each input exercises the same lazy structures.
INPUTS_PER_OP = 2

# Per-input pin budget, mirroring the differential fuzz test: an input that pins
# at all does so on the first iteration in well under a second.
PIN_ITERS = 12
PIN_TIMEOUT = 4.0

# Receiver types whose methods this test covers: the builtin container types,
# plus array and deque. Methods on other types cannot return a symbolic
# container result through the lazy structures this test targets.
_CONTAINER_OWNERS = (
    "list",
    "tuple",
    "dict",
    "set",
    "frozenset",
    "str",
    "bytes",
    "bytearray",
    "array",
    "deque",
)

# Concrete result types worth probing: containers, including dict views.
_CONTAINER_RESULT_TYPES = (
    list,
    tuple,
    dict,
    set,
    frozenset,
    str,
    bytes,
    bytearray,
    array,
    type({}.keys()),
    type({}.values()),
    type({}.items()),
)

# Ops whose unrealized result diverges from the concrete one on type-sensitive
# probes -- real model gaps surfaced by this test (distinct from the
# differential's divergences: the differential compares REALIZED results, so it
# cannot see a wrong unrealized structure). xfail'd NON-strict: which gaps
# reproduce varies by Python version, so a strict xfail would flake; run
# `pytest -rX crosshair/libimpl/container_conformance_test.py` to spot fixes
# (XPASS) and prune this list.
KNOWN_FAILURES = {
    # Symbolic dicts iterate in solver order, not insertion order, so the
    # unrealized result's repr diverges from the concrete one. (The view
    # *type* now matches: MapBase returns CrossHair's dict_keys/dict_items.)
    "dict.keys": "symbolic dict iteration order is solver order, not insertion order (repr diverges)",
    "dict.items": "symbolic dict iteration order is solver order, not insertion order (repr diverges)",
    "dict.copy": "symbolic dict iteration order is solver order, not insertion order (repr diverges)",
    "dict.__or__": "symbolic dict iteration order is solver order, not insertion order (repr diverges)",
    # Symbolic sets iterate in solver order, not CPython's hash-table order.
    "frozenset.copy": "symbolic frozenset iteration order diverges from concrete (repr diverges)",
    # Large repeats blow the symbolic recursion budget.
    "list.__mul__": "symbolic list * large int raises RecursionError (timing-dependent)",
}


def _op_marks(op):
    """Marks for one catalogued op. Skip what can't/shouldn't be probed -- out
    of scope (OS handle), a probe hazard (blocks/crashes), a side effect (real
    I/O), or an op whose output isn't a comparable value function -- and xfail
    known model gaps. The skip reasons come straight off the catalog's
    classification (crosshair.inputgen), the same fields the differential fuzz
    test reads."""
    marks = []
    if op.out_of_scope:
        marks.append(pytest.mark.skip(reason=f"out of scope: {op.out_of_scope}"))
    elif op.no_inputs:
        marks.append(pytest.mark.skip(reason=f"no inputs: {op.no_inputs}"))
    elif op.probe_hazard:
        marks.append(pytest.mark.skip(reason=f"probe hazard: {op.probe_hazard}"))
    elif op.side_effect:
        marks.append(pytest.mark.skip(reason=f"side effect: {op.side_effect}"))
    elif op.not_value_function:
        marks.append(
            pytest.mark.skip(reason=f"not a value function: {op.not_value_function}")
        )
    elif op.seedkey in KNOWN_FAILURES:
        marks.append(pytest.mark.xfail(reason=KNOWN_FAILURES[op.seedkey], strict=False))
    return marks


# Enumerate the ONE canonical surface (crosshair.inputgen.catalog) -- the same
# set the differential fuzz test draws from, restricted to methods on container
# types, so the test and the catalog can't drift. Static classification only
# (probe=False): fast, and complete here since this pure surface reaches for no
# live-probed side effects. Keyed by the rendered key; the test looks the
# Operation back up, so params stay picklable (xdist-safe).
_CATALOG = {
    op.key: op
    for op in catalog(probe=False)
    if op.call is not None
    and not op.no_inputs
    and op.kind == "method"
    and op.owner in _CONTAINER_OWNERS
}


def _catalog_params():
    # sorted() so collection order is deterministic across processes. catalog()'s
    # yield order isn't stable process-to-process (it iterates object-keyed
    # collections whose order depends on address/ASLR), and pytest-xdist aborts
    # the run if its workers collect tests in different orders. Parametrization
    # order has no bearing on outcomes (each op is checked independently).
    for key in sorted(_CATALOG):
        yield pytest.param(key, id=key, marks=_op_marks(_CATALOG[key]))


def _check_catalog_op(key, op):
    """Probe the unrealized symbolic result of one catalogued op against the
    concrete result, on each catalog input whose concrete result is a container.
    Returns after the first mismatch is found (pytest reports it); inputs that
    raise, don't pin, or don't return a container are skipped -- the
    differential fuzz test owns exception equality, and only container results
    have the lazy structures this test targets."""
    call = op.call
    inputs = inputs_for(call, k=INPUTS_PER_OP, seedkey=op.seedkey)
    checked = 0
    for vals in inputs:
        with NoTracing():
            try:
                concrete_result = call.invoke(copy.deepcopy(list(vals)))
            except Exception:
                continue
        if not isinstance(concrete_result, _CONTAINER_RESULT_TYPES):
            continue
        comparands = _same_contents(concrete_result)
        concrete_outcomes = _probe(concrete_result, comparands)

        def applier(*vs, _call=call, _comparands=comparands):
            return _probe(_call.invoke(vs), _comparands)

        res = run_symbolic_pinned(
            applier,
            call.arg_names,
            vals,
            max_pin_iters=PIN_ITERS,
            pin_timeout=PIN_TIMEOUT,
        )
        if isinstance(res, SkipReason):
            continue
        if res.exc is not None:
            continue
        checked += 1
        symbolic_outcomes = res.ret
        mismatches = [
            f"  {probe}: symbolic {symbolic_outcomes[probe]!r}, concrete {expected!r}"
            for probe, expected in concrete_outcomes.items()
            if symbolic_outcomes[probe] != expected
        ]
        assert not mismatches, (
            f"{key} diverges on `{call.expr}` with {vals!r}:\n" + "\n".join(mismatches)
        )
    if checked == 0:
        pytest.skip(f"no container-returning inputs for {key}")


@pytest.mark.parametrize("key", list(_catalog_params()))
def test_catalog_result_behaves_like_modeled_type(key):
    """Probe battery on one catalogued container operation's unrealized result."""
    _check_catalog_op(key, _CATALOG[key])
