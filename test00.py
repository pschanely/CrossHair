from crosshair import *

def is_true() -> istrue: return True

def not1() -> istrue: return not False

def not2() -> istrue: return not 0

def zero_is_falsy() -> istrue: return not 0

def nat_is_truthy() -> istrue: return 7

def bool_is_defined(x:isint) -> istrue: return isdefined(isbool(x))

def isdefined_predicates() -> istrue: return isbool(isint(7))

def true_is_bool() -> istrue: return isbool(True)

def seven_is_not_bool() -> istrue: return not isbool(7)

def int_compare() -> istrue: return (4 < 7)

def int_compare_to_bool() -> istrue: return (4 != True)

def int_compare_as_input(x :isdefined, y :isdefined) -> istrue: return (x == x) == (not x != x)

