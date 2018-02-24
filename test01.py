from crosshair import *

def one_plus_one_is_not_zero() -> istrue: return (1 + 1 != 0)

def literal_subtraction() -> istrue: return 1 - 1 < 1

def adding_symmetry(x :isint, y :isint) -> istrue:
    return x + y == y + x

def adding_increases(x :isint) -> istrue: return x + 1 > x

def lambdas_are_functions() -> istrue: return isfunc(isint)

def lambda_execution() -> istrue: return ((lambda x:x)(7) == 7)

#@ch(pattern=(lambda x:x - x))
def lambda_in_annotation(x : lambda z:isint(z)) -> istrue: return x - x == 0

def complex_lambda_in_annotation(x : lambda z:(isint(z) and z > 10)) -> istrue:
    return x > 5

def complex_lambda_in_return() -> (lambda z: z > 10):
    return 15
