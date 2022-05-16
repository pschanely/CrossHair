from typing import Callable, Dict, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def list_to_dict(s: Sequence[T]) -> Dict[T, T]:
    """
    post: len(__return__) == len(s)
    # False; CrossHair finds a counterexample with duplicate values in the input.
    """
    return dict(zip(s, s))


def consecutive_pairs(x: List[T]) -> List[Tuple[T, T]]:
    """
    post: len(__return__) == len(x) - 1
    # False (on an empty input list)
    """
    return [(x[i], x[i + 1]) for i in range(len(x) - 1)]


def higher_order(fn: Callable[[int], int]) -> int:
    """
    Crosshair can find models for pure callables over atomic types.

    post: _ != 42
    # False (when given something like lambda a: 42 if (a == 0) else 0)
    """
    return fn(fn(100))


def append_fourtytwo_to_each(lists: List[List[int]]):
    """
    pre: len(lists) >= 2
    post: all(len(x) == len(__old__.lists[i]) + 1 for i, x in enumerate(lists))
    # False when two elements of the input are the SAME list!
    """
    for ls in lists:
        ls.append(42)
