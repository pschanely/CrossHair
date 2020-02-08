class Immutable:
    '''
    A mixin to enforce that instaces are immutable.
    inv: self.__dict__ == __old__.self.__dict__

    WARNING: CrossHair implements __old__ with deep-copy-like logic.
    So this mixin is only useful if your class implements __eq__().
    '''

class Apples(Immutable):
    '''
    Uses the Immutable mixin to ensure that no method modifies the instance.
    '''
    count: int
    def buy_one_more(self) -> int:
        self.count += 1
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Apples) and self.count == other.count
    def __repr__(self):
        return f'Apples({self.count!r})'
