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

def smallest_two(numbers: Tuple[int, ...]) -> Tuple[int, int]:
    '''
    pre: numbers
    post: return[0] == min(numbers)
    '''
    if not tuple:
        return (None, None)
    (smallest, second) = smallest_two(numbers[1:])
    n = numbers[0]
    if smallest is None or n < smallest:
        return (n, smallest)
    elif second is None or n < second:
        return (smallest, n)
    else:
        return (smallest, second)

#def second_smallest(numbers: Tuple[int, ...]) -> Optional[int]:
    '''
    pre: len(numbers) > 1
    post: return == sorted(number)[1]
    '''
#    smallest, second = None, None
#    for n in numbers:
#        if smallest is None or n < smallest:
#            smallest, second = n, smallest
#        elif second is None or n < second:
#            second = n
#    return second

