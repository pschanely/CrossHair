from typing import List

import deal  # type: ignore


@deal.pre(lambda numbers: len(numbers) > 0)
@deal.ensure(lambda numbers, result: min(numbers) <= result <= max(numbers))
def average(numbers: List[float]) -> float:
    return sum(numbers) / len(numbers)
