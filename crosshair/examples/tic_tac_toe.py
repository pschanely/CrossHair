import enum
from typing import *


class Mark(enum.Enum):
    Empty = 0
    x = 1
    o = 2


class Board(NamedTuple):
    squares: List[Mark]

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
        post: _.isvalid()
        post: _.get(col, row) == player
        '''
        squares = self.squares
        idx = row * 3 + col
        assert player in (Mark.x, Mark.o)
        assert squares[idx] == Mark.Empty
        return Board(squares[:idx] + [player] + squares[idx + 1:])

    def __str__(self) -> str:
        return str(self.squares)

    def winner(self) -> Optional[Mark]:
        '''
        Returns the winning player, or the empty value if nobody has won yet.
        pre: self.isvalid()
        post: _ in (Mark.x, Mark.o, None)
        post: _ == Board(list(reversed(self.squares))).winner()
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
        post: Mark.Empty not in _
        post: _ == Board(tuple(reversed(self.squares))).winners()
        '''
        winners = set()
        for patt in ((0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
                     (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
                     (0, 4, 8), (2, 4, 6)):  # diagonals
            values = set(self.squares[i] for i in patt)
            if Mark.Empty not in values and len(values) == 1:
                winners.add(list(values)[0])
        return winners
