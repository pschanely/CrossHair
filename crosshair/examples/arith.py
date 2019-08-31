from typing import *


def perimiter_length(l: int, w: int) -> int:
    '''
    pre: l > 0 and w > 0
    post: return >= l and return > w
    '''
    return 2 * l + 2 * w

def swap(things: Tuple[int, int]) -> Tuple[int, int]:
    '''
    post: return[0] == things[1]
    post: return[1] == things[0]
    '''
    return (things[1], things[0])

def _assert_double_swap_does_nothing(things: Tuple[int, int]) -> Tuple[int, int]:
    ''' 
    post: return == things
    '''
    ret= swap(swap(things))
    return ret

def double(items: List[str]) -> List[str]:
    '''
    pre: True
    post: len(return) == len(items) * 2
    '''
    return items + items

def smallest_two(numbers: Tuple[int, ...]) -> Tuple[Optional[int], Optional[int]]:
    '''
    pre: numbers
    post: return[0] == min(numbers)
    '''
    if len(numbers) == 0:
        return (numbers[0], None)
    (smallest, second) = smallest_two(numbers[1:])
    n = numbers[0]
    if smallest is None or n < smallest:
        return (n, smallest)
    elif second is None or n < second:
        return (smallest, n)
    else:
        return (smallest, second)
