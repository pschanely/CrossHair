from crosshair import *



#def _assert_TrueOrFalse(x:isdefined):
#    return x or not x



def isbyte(x:isdefined) -> isbool:
    return isint(x) and 0 <= x < 256

#def _assert_isbyte_IsAsDefined(x):
#    return isbyte(x) == (isint(x) and 0 <= x < 256)

#def _assert_isbyte_Undefined(x:isint):
#    return isdefined(isbyte(x)) == isdefined(x)


#def _assert_isbyte_TmapDefinedWhen(l :istuple):
#    return implies(all(tmap(isbyte, l)), istuple(tmap(isbyte, l)))

#def _assert_Foo(l:istuple, t:istuple):
#    return all(l + t) == all(l) and all(t)



def nullterminate(l:listof(isbyte)) -> listof(isbyte):
    return l + (0,)

#def _assert_isbyte_NullTerminatedByteString(l:listof(isbyte)):
#    return all(tmap(isbyte, l + (0,)))
#def _assert_nullterminate_NullTerminatedByteString(l:listof(isbyte)):
#    return all(tmap(isbyte, nullterminate(l)))

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
