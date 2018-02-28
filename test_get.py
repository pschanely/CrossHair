from crosshair import *

def get_on_literals1(t:istuple) -> istrue: return implies(len(t)>0, isdefined(t[0]))
def get_on_literals2() -> istrue: return len((0,1)) == 2
def get_on_literals3() -> istrue: return (0,1)[0] == 0
def get_on_literals4() -> istrue: return (0,1,2)[-2] == 1

def get_last_on_tuple(x :isdefined, t :istuple) -> istrue:
    return (*t, x)[-1] == x

#def string_endswith() -> istrue: return "cats"[-1] == "s"

