from crosshair import *




#def _assert_TrueOrFalse(x:isdefined):
#    return x or not x






def isbyte(x:isdefined) -> isbool:
    return isint(x) and 0 <= x < 256



def _assert_isbyte_Undefined(x):
    return isdefined(isbyte(x)) == isdefined(x)


#def _assert_isbyte_NullTerminate(l:listof(isint)):
#    return istuple(tmap(isint, l))
#def _assert_isbyte_NullTerminate(x):# :isdefined):
#    return implies(isbyte(x), isdefined(isbyte(x)))
def _assert_isbyte_NullTerminate(l :istuple):
    return implies(all(tmap(isbyte, l)), istuple(tmap(isbyte, l)))
def _assert_isbyte_NullTerminate(l:listof(isbyte)):
    #return istuple(tmap(isbyte, l))
    return all(tmap(isbyte, l + (3,)))
    #return all(tmap(isbyte, l) + tmap(isbyte, (3,)))
    #return all(tmap(isbyte, l + (3,))) == all(tmap(isbyte, l) + tmap(isbyte, (3,)))
    #return all(tmap(isbyte, l) + tmap(isbyte, (3,))) == all(tmap(isbyte, l)) and all(tmap(isbyte, (3,)))




def _assert_isbyte_Foo(l:istuple, t:istuple):
    return all(l + t) == all(l) and all(t)

#return all(tmap(isbyte, l + (3,)))

#def null_terminate(l:listof(isbyte)) -> listof(isbyte):
#    return l + (0,)

#def _assert_isbyte_Foo():
#    return listof(isbyte)((1,2))



'''
def _assert_IntsAreLessThanOne(x :isint):
    return x < 1

def _assert_IntsArePositiveOrNegative(x :isint):
    return x > 0 or x <= 0



def incr(x :isint) -> isint:
    return x + 1

def _assert_incr_IsGreaterThanInput(x :isint):
    return incr(x) > x



def double(x :isint) -> isint:
    return x + x

def _assert_double_IsGreaterThanInput(y :isnat):
    return double(y) > y

'''
