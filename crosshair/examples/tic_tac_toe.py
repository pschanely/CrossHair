import enum
from typing import *


def perimiter_length(l: int, w: int) -> int:
    '''
    pre: l > 0 and w > 0
    post: return > l and return > w
    '''
    return 2 * l + 2 * w


def avg(numbers: List[int]) -> float:
    '''
    pre: len(numbers) > 0
    post: True
    '''
    return sum(numbers) / len(numbers)


class Mark(enum.Enum):
    Empty = 0
    x = 1
    o = 2


class Board(NamedTuple):
    squares: Tuple[Mark, ...]

    def isvalid(self):
        return len(self.squares) == 9

    def get(self, col: int, row: int) -> Mark:
        '''
        pre: self.isvalid()
        pre: 0 <= col < 3
        pre: 0 <= row < 3
        post: True
        '''
        return self.squares[row * 3 + col]

    def play(self, player: Mark, col: int, row: int) -> 'Board':
        '''
        pre: self.isvalid()
        pre: 0 <= col < 3
        pre: 0 <= row < 3
        pre: self.get(col, row) == Mark.Empty
        pre: player != Mark.Empty
        post: return.isvalid()
        post: return.get(col, row) == player
        '''
        squares = self.squares
        idx = row * 3 + col
        assert player in (Mark.x, Mark.o)
        assert squares[idx] == Mark.Empty
        return Board(tuple(squares[:idx] + (player,) + squares[idx + 1:]))

    def winner(self) -> Mark:
        '''
        Returns the winning player, or the empty value if nobody has won yet.
        pre: self.isvalid()
        post: return in (Mark.x, Mark.o, None)
        post: return == winner(Board(tuple(reversed(self.squares))))
        '''
        for patt in ((0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
                     (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
                     (0, 4, 8), (2, 4, 6)):  # diagonals
            values = set(self.squares[i] for i in patt)
            if Mark.Empty not in values and len(values) == 1:
                return list(values)[0]
        return None

    def winners(self) -> Set[Mark]:
        '''
        Returns the winning players.
        pre: self.isvalid()
        post: Mark.Empty not in return
        post: return == winners(Board(tuple(reversed(self.squares))))
        '''
        winners = set()
        for patt in ((0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
                     (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
                     (0, 4, 8), (2, 4, 6)):  # diagonals
            values = set(self.squares[i] for i in patt)
            if Mark.Empty not in values and len(values) == 1:
                winners.add(list(values)[0])
        return winners

