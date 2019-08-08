import builtins as orig_builtins
from typing import *

# We don't do hash() because it is called by typing.py to hash types
# (and gets called indirectly via core.py)
# TODO: consider blocking interceptions in crosshair code stacks
#def hash(obj: Hashable) -> int:
#    '''
#    post: -2**63 <= return < 2**63
#    '''
#    ...

_T = TypeVar('_T')
def sum(i: Iterable[_T]) -> Union[_T, int]:
    '''
    post: return == 0 or len(i) > 0
    '''
    ...

