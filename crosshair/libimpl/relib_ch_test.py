import re
import sys
from typing import Optional, Sequence

import pytest  # type: ignore

from crosshair.core_and_libs import MessageType, analyze_function, run_checkables
from crosshair.options import AnalysisOptionSet
from crosshair.test_util import ResultComparison, compare_results


def groups(match: Optional[re.Match]) -> Optional[Sequence]:
    if match is None:
        return None
    return match.groups(), match.start(), match.end()


def check_inverted_categories(text: str, flags: int) -> ResultComparison:
    """
    pre: len(text) == 3
    post: _
    """
    return compare_results(
        lambda t, f: groups(re.fullmatch(r"\W\S\D", t, f)), text, flags
    )


def check_nongreedy(text: str, flags: int) -> ResultComparison:
    """post: _"""
    return compare_results(
        lambda t, f: groups(re.match("a+?(a*?)a", t, f)), text, flags
    )


def check_match_repr(text: str) -> ResultComparison:
    """
    pre: len(text) < 2
    post: _
    """
    return compare_results(lambda t: repr(re.match(r"[^ab]", t)), text)


def check_match_with_sliced_string(text: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t: groups(re.match(r"^[ab]{2}\Z", t)), text[1:])


def check_findall(text: str, flags: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t, f: re.findall("aa", t, f), text, flags)


def check_findall_with_groups(text: str, flags: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t, f: re.findall("a(a)", t, f), text, flags)


def check_findall_with_empty_matches(text: str, flags: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t, f: re.findall("a?", t, f), text, flags)


def check_finditer(text: str, flags: int) -> ResultComparison:
    """post: _"""
    return compare_results(
        lambda t, f: list(map(groups, re.finditer("(^|a?)", t, f))), text, flags
    )


def check_finditer_with_bounds(text: str, pos: int) -> ResultComparison:
    """post: _"""
    regex = re.compile("a?")
    return compare_results(
        lambda *a: list(map(groups, regex.finditer(*a))), text, pos, pos * 2
    )


def check_finditer_with_sliced_string(text: str) -> ResultComparison:
    """post: _"""
    return compare_results(
        lambda t: list(map(groups, re.finditer("(a|bb)", t))), text[1:]
    )


def check_search(text: str, flags: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t, f: groups(re.search("aa", t, f)), text, flags)


def check_search_with_offset(text: str, pos: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda *a: groups(re.compile("a").search(*a)), text, pos)


def check_search_with_bounds(text: str, pos: int, endpos: int) -> ResultComparison:
    """post: _"""
    return compare_results(
        lambda *a: groups(re.compile("a").search(*a)), text, pos, endpos
    )


def check_search_anchored_begin(text: str, flags: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t, f: groups(re.search("^a", t, f)), text, flags)


def check_search_anchored_end(text: str, flags: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t, f: groups(re.search("a$", t, f)), text, flags)


def check_subn(text: str, flags: int) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t, f: re.subn("aa", "ba", t, f), text, flags)


def check_lookahead(text: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t: groups(re.match("a(?=b)", t)), text)


def check_lookbehind(text: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t: groups(re.search("(?<=a)b", t)), text)


def check_negative_lookahead(text: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t: groups(re.match("a(?!b)", t)), text)


def check_negative_lookbehind(text: str) -> ResultComparison:
    """post: _"""
    return compare_results(lambda t: groups(re.search(".(?<!b)", t)), text)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    opts = AnalysisOptionSet(max_iterations=20, per_condition_timeout=30)
    this_module = sys.modules[__name__]
    fn = getattr(this_module, fn_name)
    messages = run_checkables(analyze_function(fn, opts))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
