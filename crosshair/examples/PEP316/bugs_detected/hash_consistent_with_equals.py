import dataclasses


class HasConsistentHash:
    """
    A mixin to enforce that classes have hash methods that are consistent
    with their equality checks.
    """

    def __eq__(self, other: object) -> bool:
        """
        post: implies(__return__, hash(self) == hash(other))
        """
        raise NotImplementedError


@dataclasses.dataclass
class Apples(HasConsistentHash):
    """
    Uses HasConsistentHash to discover that the __eq__ method is
    missing a test for the `count` attribute.
    """

    count: int
    kind: str

    def __hash__(self):
        return self.count + hash(self.kind)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Apples) and self.kind == other.kind
