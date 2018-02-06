from crosshair import *

def is_true() -> istrue: return True

def not1() -> istrue: return not False

def not2() -> istrue: return not 0

def equality1() -> istrue: return True == True

def equality2() -> istrue: return (4, 5) == (4, 5)

def inequality2() -> istrue: return (4, 5) != (5, 4)

def x_or_not_x(x :isdefined) -> istrue: return x or not x

def zero_is_falsy() -> istrue: return not 0

def nat_is_truthy() -> istrue: return 7

def int_compare() -> istrue: return (4 < 7)

def int_compare_to_bool() -> istrue: return (4 != True)

def int_compare_and_conjunction() -> istrue: return (3 < 7 and 7 >= 7)

def not_equal(x:isdefined, y:isdefined) -> istrue:
    return (x != y) == (not (x == y))

def implies1() -> istrue: return implies(False, False)

def implies2(x:isdefined) -> istrue: return implies(x, x != 0)

def bool_is_defined(x:isint) -> istrue: return isdefined(isbool(x))

def isdefined_predicates() -> istrue: return isbool(isint(7))

def true_is_bool() -> istrue: return isbool(True)

def seven_is_not_bool() -> istrue: return not isbool(7)

def distribute_not_through_and(x:isdefined, y:isdefined) -> istrue:
    return implies(not (x and y), (not x) or (not y))

def distribute_not_through_or(x:isdefined, y:isdefined) -> istrue:
    return implies(not (x or y), (not x) and (not y))
