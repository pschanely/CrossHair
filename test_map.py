from crosshair import *

def map_empty() -> istrue: return tmap(isint, ()) == ()
def map_empty_is_defined(f:isfunc) -> istrue: return isdefined(tmap(f, ()))
def map_literals1() -> istrue: return tmap(isint, (2,)) == (True,)
def map_literals2() -> istrue: return tmap(isint, (2,3)) == (True, True)
def map_literals3() -> istrue: return not all(tmap(isint, (2, False)))
