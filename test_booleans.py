from crosshair import *

def implies_true_on_false_condition() -> istrue: return implies(False, False)

@ch(pattern=(lambda x: isdefined(isbool(x))))
def implies_truthy_is_not_zero(x:isdefined) -> istrue: return implies(x, x != 0)

@ch(pattern=(lambda x: isdefined(isbool(x))))
def x_or_not_x(x :isdefined) -> istrue: return x or not x

@ch(pattern=(lambda x, y: not (x and y)))
def distribute_not_through_and(x:isdefined, y:isdefined) -> istrue:
    return implies(not (x and y), (not x) or (not y))

@ch(pattern=(lambda x, y: not (x or y)))
def distribute_not_through_or(x:isdefined, y:isdefined) -> istrue:
    return implies(not (x or y), (not x) and (not y))

def int_compare_and_conjunction() -> istrue: return (3 < 7 and 7 >= 7)
