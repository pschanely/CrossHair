from typing import Callable, Dict, List, Sequence, Tuple, TypeVar

from icontract import ensure, require, snapshot

T = TypeVar("T")


@require(lambda x: x > 0)
@ensure(lambda result: result > 0)
def some_func(x: int) -> int:
    # Bug when the constant makes the result negative.
    return x - 1000


@ensure(lambda s, result: len(result) == len(s))
def list_to_dict(s: Sequence[T]) -> Dict[T, T]:
    # CrossHair finds a counterexample with duplicate values in the input.
    return dict(zip(s, s))


@ensure(lambda x, result: len(result) == len(x) - 1)
def consecutive_pairs(x: List[T]) -> List[Tuple[T, T]]:
    # Bug on an empty input list
    return [(x[i], x[i + 1]) for i in range(len(x) - 1)]


@ensure(lambda result: result != 42)
def higher_order(fn: Callable[[int], int]) -> int:
    # Crosshair can find models for pure callables over atomic types.
    # Bug when given something like lambda a: 42 if (a == 0) else 0.
    return fn(fn(100))


@snapshot(lambda lists: lists[:])
@ensure(
    lambda lists, OLD: all(len(x) == len(OLD.lists[i]) + 1 for i, x in enumerate(lists))
)
def append_fourtytwo_to_each(lists: List[List[int]]):
    # Bug when two elements of the input are the SAME list!
    for ls in lists:
        ls.append(42)
