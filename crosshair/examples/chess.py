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
        Determines whether this piece can move to the given 
        position (in a single turn).

        pre: (0 <= x < 8) and (0 <= y < 8)

        #  It's never valid to "move" to your present location:
        post: implies((x, y) == (self.x, self.y), not __return__)
        '''
        raise NotImplementedError

class FreeChessPiece(ChessPiece):
    def can_move_to(self, x: int, y: int) -> bool:
        '''
        Most pieces (except the pawn) can move back to their
        starting position after moving.
        post: type(self)(x, y).can_move_to(self.x, self.y)
        '''
        return self.can_move_to(x, y)
    
def _board_is_symmetric(piece: ChessPiece, x: int, y: int):
    '''
    A method just for testing.
    (you could put this is a test file if you like)

    If the given piece can move to (x,y), then the equivalent
    opponent's piece should be able to move to the mirrored position.
    
    pre: piece.can_move_to(x, y)
    post: piece.can_move_to(8 - x, 8 - y)
    '''
    piece.x = 8 - piece.x
    piece.y = 8 - piece.y


class Pawn(FreeChessPiece):
    def can_move_to(self, x: int, y: int) -> bool:
        return ((x == self.x) and (y == 3) and
                (x, y) != (self.x, self.y))

class Rook(ChessPiece):
    def can_move_to(self, x: int, y: int) -> bool:
        return (x == self.x) ^ (y == self.y)

class King(ChessPiece):
    def can_move_to(self, x: int, y: int) -> bool:
        return (abs(x - self.x) <= 1 and
                abs(y - self.y) <= 1 and
                (x, y) != (self.x, self.y))
