import dataclasses
from typing import List


@dataclasses.dataclass
class AverageableStack:
    """
    A stack of numbers with a O(1) average() operation.

    inv: self._total == sum(self._values)
    """

    _values: List[int]
    _total: int

    def __init__(self):
        self._values = []
        self._total = 0

    def push(self, val: int):
        """post: True"""
        self._values.append(val)
        self._total += val

    def pop(self) -> int:
        """
        pre: self._values
        """
        val = self._values.pop()
        self._total -= val
        return val

    def average(self) -> float:
        """pre: self._values"""
        return self._total / len(self._values)
