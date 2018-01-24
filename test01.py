from crosshair import *



def isbyte(x:isdefined) -> isbool:
    return isint(x) and 0 <= x and x < 256

def _assert_isbyte_IsAsDefined(x):
    return isbyte(x) == (isint(x) and 0 <= x and x < 256)

def isbytes(x:isdefined) -> isbool:
    return istuple(x) and all(tmap(isbyte, x))




def _assert_isbytes_Sums(b):
    return implies(isbyte(b), b < 256)


#def plural(s :isstring) -> isstring:
#    return s + "s"

#def _assert_plural_Length(s :isstring):
#    return len(plural(s)) == len(s) + 1

