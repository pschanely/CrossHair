
class Immutable:
    '''
    A mixin to enforce that instaces are immutable.
    inv: self.__dict__ == __old__.self.__dict__
    '''

class Apples(Immutable):
    '''
    Uses the Immutable mixin to ensure that no method modifies the instance.
    '''
    count: int
    def buy_one_more(self) -> None:
        self.count += 1
    def __repr__(self):
        return f'Apples({self.count!r})'
