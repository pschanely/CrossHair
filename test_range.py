from crosshair import *

def range_isnat(x:isint) -> istrue: return all(tmap(isnat, trange(x)))
def range_isint(x:isint) -> istrue: return all(tmap(isint, trange(x)))
def range_last(x:isint) -> istrue: return implies(x>0, trange(x)[-1] == x-1)

# ground terms are greedy and screw up general theorems above, so we put them later:
def range_defined1() -> istrue: return isdefined(trange(5))
def range_literals1() -> istrue: return trange(1) == (0,)
def range_literals2() -> istrue: return trange(2) == (0,1)

# # Too tricky! Requires induction?
#def range_first(x:isnat) -> istrue: return implies(x>0, trange(x)[0] == 0)

