import sys
from typing import DefaultDict, Deque, Optional, Sequence

import pytest  # type: ignore

from crosshair.core_and_libs import MessageType, analyze_function, run_checkables
from crosshair.options import AnalysisOptionSet
from crosshair.test_util import ResultComparison, compare_results

# deque


def check_deque_append(queue: Deque[int], item: int):
    """post: _"""

    def checker(q, i):
        q.append(i)
        return q

    return compare_results(checker, queue, item)


def check_deque_appendleft(queue: Deque[int], item: int):
    """post: _"""

    def checker(q, i):
        q.appendleft(i)
        return q

    return compare_results(checker, queue, item)


def check_deque_copy(queue: Deque[int]):
    """post: _"""
    return compare_results(lambda d: d.copy(), queue)


def check_deque_count(queue: Deque[int], item: int):
    """post: _"""
    return compare_results(lambda d, i: d.count(i), queue, item)


def check_deque_extend(queue: Deque[int], items: Sequence[int]):
    """post: _"""

    def checker(q, i):
        q.extend(i)
        return q

    return compare_results(checker, queue, items)


def check_deque_extendleft(queue: Deque[int], items: Sequence[int]):
    """post: _"""

    def checker(q, i):
        q.extendleft(i)
        return q

    return compare_results(checker, queue, items)


def check_deque_index(
    queue: Deque[int], item: int, start: Optional[int], end: Optional[int]
):
    """post: _"""
    return compare_results(lambda q, i, s, e: q.index(i, s, e), queue, item, start, end)


def check_deque_insert(queue: Deque[int], position: int, item: int):
    """post: _"""

    def checker(q, p, i):
        q.insert(p, i)
        return q

    return compare_results(checker, queue, position, item)


def check_deque_pop(queue: Deque[int]):
    """post: _"""

    def checker(q):
        item = q.pop()
        return (item, q)

    return compare_results(checker, queue)


def check_deque_popleft(queue: Deque[int]):
    """post: _"""

    def checker(q):
        item = q.popleft()
        return item

    return compare_results(checker, queue)


def check_deque_remove(queue: Deque[int], item: int):
    """post: _"""

    def checker(q, n):
        q.remove(n)
        return q

    return compare_results(checker, queue, item)


def check_deque_reverse(queue: Deque[int]):
    """post: _"""

    def checker(q):
        q.reverse()
        return q

    return compare_results(checker, queue)


def check_deque_rotate(queue: Deque[int], amount: int):
    """post: _"""

    def checker(q, n):
        q.rotate(n)
        return q

    return compare_results(checker, queue, amount)


def check_deque_maxlen(queue: Deque[int]):
    return compare_results(lambda q: q.maxlen, queue)


def check_deque_eq(queue: Deque[int]):
    return compare_results(lambda q: q, queue)


def check_deque_getitem(queue: Deque[int], idx: int):
    """post: _"""
    return compare_results(lambda q, i: q[i], queue, idx)


def check_deque_contains(queue: Deque[int], item: int):
    """post: _"""
    return compare_results(lambda q, i: i in q, queue, item)


def check_deque_add(queue: Deque[int], items: Deque[int]):
    """post: _"""
    return compare_results(lambda q, i: q + i, queue, items)


def check_deque_mul(queue: Deque[int], count: int):
    """post: _"""
    return compare_results(lambda q, i: q * i, queue, count)


# defaultdict


def check_defaultdict_getitem(container: DefaultDict[int, int], key: int):
    """post: _"""
    return compare_results(lambda d, k: d[k], container, key)


def check_defaultdict_delitem(container: DefaultDict[int, int], key: int):
    """post: _"""

    def checker(d, k):
        del d[k]
        return d

    return compare_results(checker, container, key)


def check_defaultdict_inplace_mutation(container: DefaultDict[int, int]):
    """post: _"""

    def setter(c):
        if c:
            c[0] &= 42
        return c

    return compare_results(setter, container)


def check_defaultdict_iter(dictionary: DefaultDict[int, int]) -> ResultComparison:
    """post: _"""
    return compare_results(lambda d: list(d), dictionary)


def check_defaultdict_clear(dictionary: DefaultDict[int, int]) -> ResultComparison:
    """post: _"""

    def checker(d):
        d.clear()
        return d

    return compare_results(checker, dictionary)


def check_defaultdict_pop(dictionary: DefaultDict[int, int]) -> ResultComparison:
    """post: _"""

    def checker(d):
        x = d.pop()
        return (x, d)

    return compare_results(checker, dictionary)


def check_defaultdict_popitem(
    dictionary: DefaultDict[int, int], key: int
) -> ResultComparison:
    """post: _"""

    def checker(d, k):
        x = d.popitem(k)
        return (x, d)

    return compare_results(checker, dictionary)


def check_defaultdict_update(
    left: DefaultDict[int, int], right: DefaultDict[int, int]
) -> ResultComparison:
    """post: _"""

    def checker(d1, d2):
        d1.update(d2)
        return d1

    return compare_results(checker, left, right)


def check_defaultdict_values(dictionary: DefaultDict[int, int]) -> ResultComparison:
    """post: _"""
    # TODO: value views compare false even with new views from the same dict.
    # Ensure we match this behavior.
    return compare_results(lambda d: list(d.values()), dictionary)


# This is the only real test definition.
# It runs crosshair on each of the "check" functions defined above.
@pytest.mark.parametrize("fn_name", [fn for fn in dir() if fn.startswith("check_")])
def test_builtin(fn_name: str) -> None:
    opts = AnalysisOptionSet(max_iterations=7, per_condition_timeout=20)
    this_module = sys.modules[__name__]
    fn = getattr(this_module, fn_name)
    messages = run_checkables(analyze_function(fn, opts))
    errors = [m for m in messages if m.state > MessageType.PRE_UNSAT]
    assert errors == []
