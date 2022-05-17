from typing import Dict, Sequence, TypeVar

import deal  # type: ignore

T = TypeVar("T")

# False; CrossHair finds a counterexample with duplicate values in the input.
@deal.ensure(lambda _: len(_.result) == len(_["items"]))
def list_to_dict(items: Sequence[T]) -> Dict[T, T]:
    return dict(zip(items, items))
