import random
from typing import List

import deal  # type: ignore


@deal.pre(lambda items, rng: bool(items))
@deal.post(lambda result: result != "boo")
@deal.has()
def choice(items: List[str], rng: random.Random) -> str:
    """Get a random element from the given list."""
    return rng.choice(items)
