from typing import Dict, List, Optional, Tuple


class ShoppingCart:
    """
    inv: all(quantity > 0 for (_, quantity) in self.items)
    """

    def __init__(self, items: Optional[List[Tuple[str, int]]] = None):
        self.items = items or []


def compute_total(cart: ShoppingCart, prices: Dict[str, float]) -> float:
    """
    pre: len(cart.items) > 0
    pre: all(pid in prices for (pid, _) in cart.items)

    We try to ensure that you can't check out with a zero (or less) total.
    However, we forgot a precondition; namely that the `prices` dictionary only has
    prices that are greater than zero.

    post: __return__ > 0
    """
    return sum(prices[pid] * quantity for (pid, quantity) in cart.items)
