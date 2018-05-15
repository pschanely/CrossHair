from typing import *


def perimiter_length(l: int, w: int) -> int:
    '''
    pre: l > 0 and w > 0
    post: return >= max(l, w)
    '''
    return 2 * l + 2 * w


def avg(numbers: List[int]) -> float:
    '''
    pre: numbers
    post: True
    # min(numbers) <= return <= max(numbers)
    '''
    return sum(numbers) / len(numbers)
