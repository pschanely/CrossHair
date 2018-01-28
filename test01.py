from crosshair import *




def isbyte(x:isdefined) -> isbool:
    return isint(x) and 0 <= x and x < 256

def _assert_isbyte_IsAsDefined(x):
    return isbyte(x) == (isint(x) and 0 <= x and x < 256)

#def isbytes(x:isdefined) -> isbool:
#    return istuple(x) and all(tmap(isbyte, x))




#def _assert_isbytes_Sums(b):
#    return implies(isbyte(b), b < 256)

#def _assert_isbyte_Listof():
#    return listof(isbyte)( (1000,) )


def nullterminate(l :listof(isbyte)) -> listof(isbyte):
    return l + (0,)



def _assert_nullterminate_Length(l:listof(isbyte)):
    return len(nullterminate(l)) == len(l) + 1


def _assert_nullterminate_EndsWithNull(l:listof(isbyte)):
    return nullterminate(l)[-1] == 0

def plural(s :isstring) -> isstring:
    return s + "s"

def _assert_plural_Cats():
    return plural("cat") == "cats"

def _assert_plural_Length(s :isstring):
    return len(plural(s)) == len(s) + 1

def _assert_plural_Length(s :isstring):
    return len(plural(plural(s))) >= 2

def _assert_plural_EndsWithS(s :isstring):
    return plural(s)[-1] == "s"
    

