import builtins as orig_builtins
from typing import *

_T = TypeVar('_T')
def sum(i: Iterable[_T]) -> Union[_T, int]:
    '''
    post: return == 0 or len(i) > 0
    '''
    ...

