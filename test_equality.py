from crosshair import *

def equality1() -> istrue: return True == True

def equality2() -> istrue: return (4, 5) == (4, 5)

def inequality2() -> istrue: return (4, 5) != (5, 4)

def not_equal(x:isdefined, y:isdefined) -> istrue:
    return (x != y) == (not (x == y))
