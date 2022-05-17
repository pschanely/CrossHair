# crosshair: per_condition_timeout=2


class Farm:
    def visit_chickens(self) -> str:
        return "cluck"

    def visit_cows(self) -> str:
        return "moo"


def visit_animals(animal: str) -> str:
    """
    post: __return__ != "moo"
    """
    try:
        return getattr(Farm(), "visit_" + animal)()
    except BaseException:
        return ""
