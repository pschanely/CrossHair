from typing import List, Optional, Tuple

import icontract


@icontract.require(lambda ln, w: ln > 0 and w > 0)
@icontract.ensure(
    lambda ln, w, result: result > ln and result > w,
    "The perimeter of a rectangle is longer than any single side.",
)
def perimiter_length(length: int, width: int) -> int:
    return 2 * length + 2 * width


@icontract.ensure(lambda things, result: result[0] == things[1])
@icontract.ensure(lambda things, result: result[1] == things[0])
def swap(things: Tuple[int, int]) -> Tuple[int, int]:
    """
    Swap the arguments.
    """
    return (things[1], things[0])


@icontract.ensure(lambda items, result: len(result) == len(items) * 2)
def double(items: List[str]) -> List[str]:
    """
    Return a new list that is the input list, repeated twice.
    """
    return items + items


# NOTE: This is an example of contracts on recursive functions.


@icontract.require(lambda numbers: len(numbers) > 0)
@icontract.ensure(
    lambda numbers, result: result[0] == min(numbers),
    "The left return value is always the smallest.",
)
def smallest_two(numbers: Tuple[int, ...]) -> Tuple[Optional[int], Optional[int]]:
    """Find the two smallest numbers."""
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
