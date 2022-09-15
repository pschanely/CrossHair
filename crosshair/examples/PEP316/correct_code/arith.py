from typing import List, Optional, Tuple


def perimiter_length(length: int, width: int) -> int:
    """
    pre: l > 0 and w > 0

    The perimeter of a rectangle is longer than any single side:
    post: _ > l and _ > w
    """
    return 2 * length + 2 * width


def swap(things: Tuple[int, int]) -> Tuple[int, int]:
    """
    Swap the arguments.

    post: _[0] == things[1]
    post: _[1] == things[0]
    """
    return (things[1], things[0])


# NOTE: To perform additional testing, you can write extra private functions like this one:
def _assert_double_swap_does_nothing(things: Tuple[int, int]) -> Tuple[int, int]:
    """
    post: _ == things
    """
    ret = swap(swap(things))
    return ret


def double(items: List[str]) -> List[str]:
    """
    Return a new list that is the input list, repeated twice.

    post: len(_) == len(items) * 2
    """
    return items + items


# NOTE: This is an example of contracts on recursive functions.
def smallest_two(numbers: Tuple[int, ...]) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the two smallest numbers.

    pre: len(numbers) > 0
    # The left return value is always the smallest
    post: _[0] == min(numbers)
    """
    if len(numbers) == 1:
        return (numbers[0], None)
    (smallest, second) = smallest_two(numbers[1:])
    n = numbers[0]
    if smallest is None or n < smallest:
        return (n, smallest)
    elif second is None or n < second:
        return (smallest, n)
    else:
        return (smallest, second)
