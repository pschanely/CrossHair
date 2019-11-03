import dataclasses

@dataclasses.dataclass(init=False)
class ChessPiece:
    '''
    inv: 0 <= self.x < 8
    inv: 0 <= self.y < 8
    '''
    x: int
    y: int

    def can_move_to(self, x: int, y: int) -> bool:
        '''
        pre: (0 <= x < 8) and (0 <= y < 8)
        #  It's never valid to "move" to your present location:
        post: implies((x, y) == (self.x, self.y), not __return__)
        '''
        raise NotImplementedError


class Rook(ChessPiece):
    def can_move_to(self, x: int, y: int) -> bool:
        return (x == self.x) ^ (y == self.y)
