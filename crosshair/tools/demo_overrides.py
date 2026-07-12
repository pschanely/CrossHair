"""
Curated demo overrides for the support treemap.

The support map (:mod:`crosshair.tools.measure_support`) colors each operation
cell from an honest fuzzed sweep, then auto-generates a runnable crosshair-web
demo from the sweep's own witness.  Those generated demos are always *consistent*
with the cell color, but they are often pedagogically weak: the contract is
``post: _ != <the value we just computed>``, so for an idempotent op (``str.lower``,
``str.strip``, ...) the reader can "solve" it by pasting the literal, and the
fuzzed literals are frequently unreadable astral-plane noise.

A better demo is a hand-written one that poses a natural-language question.  We
already have 50+ of those -- the ``@pytest.mark.demo`` tests in the ``libimpl``
``*lib_test`` modules::

    @pytest.mark.demo
    def test_int___add___method():
        def f(a: int, b: int) -> int:
            '''
            Can the sum of two consecutive integers be 37?

            pre: a + 1 == b
            post: _ != 37
            '''
            return a + b

        check_states(f, POST_FAIL)

These are curated, readable, AND self-verifying (a real test asserts they behave).
This module HARVESTS them -- by parsing the test *source*, so it pulls in no test
dependencies (no pytest import, no test-module side effects) -- into a
``seedkey -> crosshair-web source`` registry that ``measure_support`` consults for
the demo link.

Crucially, an override changes ONLY the link.  The cell color stays the honest
measured verdict, and a curated demo can never paint the map.  To improve a cell's
demo you just write (or fix) a ``@pytest.mark.demo`` test -- CI keeps it honest and
this harvester picks it up automatically.

Because those tests assert ``check_states(f, POST_FAIL)`` in CI, every harvested
demo is *winnable* (CrossHair solves it) by construction.  A winnable demo can only
honestly illustrate a cell CrossHair handles, so ``measure_support`` applies
overrides to GREEN/YELLOW cells only (re-confirming the demo still solves under the
measurement budget).  Red/black cells keep their generated demo, which shows the
*limitation* itself -- something a CI-passing test could never demonstrate; a
curated "watch it struggle" demo would have to be a hand-authored source here, not
a harvested test.

The mapping from a test name to the op's ``seedkey`` (its dotted identity, the
same key ``measure_support`` looks up) is::

    test_<owner>___<dunder>___method  ->  <owner>.<dunder>        (builtin method)
    test_<owner>_<name>_method        ->  <owner>.<name>          (builtin method)
    test_<owner>___<dunder>___method  ->  <module>.<owner>.<dunder>  (stdlib method)
    test_<name>                       ->  <module>.<name>         (free function)

with ``<module>`` taken from the owning ``*lib_test`` file.
"""

import re
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

_LIBIMPL = Path(__file__).resolve().parent.parent / "libimpl"

# A demo test: the marker (optional color arg), the def line, then everything up
# to the closing ``check_states(`` call, from which we lift the inner ``def f``.
_DEMO_RE = re.compile(r'@pytest\.mark\.demo(?:\("(\w+)"\))?\s*\ndef (test_\w+)\(')
_F_RE = re.compile(r"( *def f\(.*?)\n\s*check_states\(", re.DOTALL)


def _iter_demo_tests(src: str) -> Iterator[Tuple[str, str, str]]:
    """Yield ``(test_name, marker_color, dedented_f_source)`` for every
    ``@pytest.mark.demo`` test in a module's source text."""
    for m in _DEMO_RE.finditer(src):
        color = m.group(1) or "green"
        test = m.group(2)
        # the test body runs from just after the def line to the next top-level
        # ``def``/``class``/decorator (or EOF); the demo's ``f`` lives inside it.
        tail = src[m.end() :]
        nxt = re.search(r"\n(?=@pytest\.mark|def |class )", tail)
        block = tail[: nxt.start()] if nxt else tail
        fm = _F_RE.search(block)
        if fm:
            yield test, color, textwrap.dedent(fm.group(1))


def _seedkey(module: str, test: str) -> str:
    """The op seedkey a demo test targets (see the module docstring for the
    conventions).  A trailing ``_method`` or ``_operator`` marks a method/operator
    test (``test_float___pow___operator`` -> ``float.__pow__``); anything else is a
    free function (``test_sorted`` -> ``builtins.sorted``)."""
    name = test[len("test_") :]
    for suffix in ("_method", "_operator"):
        if name.endswith(suffix):
            owner, _, meth = name[: -len(suffix)].partition("_")
            return (
                f"{owner}.{meth}"
                if module == "builtins"
                else f"{module}.{owner}.{meth}"
            )
    return f"{module}.{name}"


def _source(module: str, f_src: str) -> str:
    """Wrap a harvested ``def f`` in a self-contained crosshair-web source.  Stdlib
    demos may reference either qualified (``json.dumps``) or bare (``deque``) names,
    so we make both available; ``from typing import *`` covers the annotations."""
    if module == "builtins":
        return "from typing import *\n\n" + f_src
    return f"from typing import *\nimport {module}\nfrom {module} import *\n\n" + f_src


@lru_cache(maxsize=None)
def demo_overrides() -> Dict[str, List[Tuple[str, str]]]:
    """``seedkey -> [(marker_color, crosshair_web_source), ...]`` harvested from the
    ``@pytest.mark.demo`` tests, best (first-declared) candidate first.

    A list per seedkey so a cell can offer several curated demos;
    ``measure_support`` uses the first that survives its color-consistency check.
    """
    out: Dict[str, List[Tuple[str, str]]] = {}
    for path in sorted(_LIBIMPL.glob("*lib_test.py")):
        module = path.stem[: -len("lib_test")] or "builtins"
        src = path.read_text()
        for test, color, f_src in _iter_demo_tests(src):
            out.setdefault(_seedkey(module, test), []).append(
                (color, _source(module, f_src))
            )
    return out


def demo_sources(seedkey: str) -> List[str]:
    """Curated crosshair-web sources for an op (best first), or ``[]`` if none."""
    return [src for _color, src in demo_overrides().get(seedkey, [])]
