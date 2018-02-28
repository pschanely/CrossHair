from crosshair import *

#@ch(pattern=lambda x, z:z in trange(x))
#def range_map_lambda(x :isint, z :isint) -> istrue:
#    return implies(z in trange(x), z >= 0)

def ident(x):
    return x + ""

def range_map_ident(x :isint) -> istuple:
    return tmap(ident, trange(x))

#def addone(x :isint) -> isint:
#    return x + 1

#def range_map_addone(x :isint) -> istrue:
#    return all(tmap(addone, trange(x)))



                    
#def range_defined2(x:isint) -> istrue: return isdefined(all(tmap(isint, trange(x))))
#def foo() -> istrue: return False




# # This is difficult.
# # Induction doesn't work well because of lambda equality.
#  def p(x:isint): return all(tuple(i+1 for i in trange(x)))
#  def p(x:isint): return min(x+1,y+1) == min(x,y)+1
#  def p(x:isint): return all(maplambdap1(trange(x)))

#def myincr(x):
#    return x + 1

#def foo(x:isint) -> istrue: return all(trange(x))

#def foo(x:isint) -> istrue: return all(tmap((lambda x:x), trange(x)))


'''

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

'''
















'''
def fib(x):
    return 1 if x <= 1 else fib(x-1) + fib(x-2)

def int_measure(x):
    return x if isnat(x) else 0

#def _assert_fib_IsIntBaseCase():
#    return isint(fib(0))


def fibFoo(x :isdefined) -> isint:
    return x

#def _assert_int_measure_IsIntPositiveInduction(x: isnat):
#    return int_measure(x) > int_measure(x-1)

#def _assert_fib_IsIntNegativeInduction(x: isint):
#    return implies(isint(fib(x)) and x < 0, isint(fib(x-1)))

#def isinttree(x :isdefined) -> isbool:
#    return isint(x) or (istuple(x) and len(x) == 2 and isinttree(x[0]) and isinttree(x[1]))

'''


'''

def isbyte(x:isdefined) -> isbool:
    return isint(x) and 0 <= x < 256

#def _assert_isbyte_TrueOrFalse(x:isdefined):
#    return x or not x


#def _assert_isbyte_IsAsDefined(x):
#    return isbyte(x) == (isint(x) and 0 <= x < 256)
#return implies(isbyte(x), isint(x) and (0 <= x and x < 256))



#def _assert_isbyte_Undefined1(x :isint):
#    return isdefined(x)

#def _assert_isbyte_Undefined2(x :isint):
#    return implies(isdefined(isbyte(x)), isdefined(x))


#def _assert_isbyte_TmapDefinedWhen(l :istuple):
#    return implies(all(tmap(isbyte, l)), istuple(tmap(isbyte, l)))

#def _assert_Foo(l:istuple, t:istuple):
#    return all(l + t) == all(l) and all(t)


#def nullterminate(l:listof(isbyte)) -> listof(isbyte):
#    return l + (0,)

def plural(s :isstring) -> isstring:
    return s + "s"

def _assert_plural_Length(s :isstring):
    return len(plural(s)) == len(s) + 1




def _assert_plural_Things(s :isstring):
    return plural(s)[len(s)-1:len(s)] == "s"



#def _assert_nullterminate_Length(l:listof(isbyte)):
#    return len(nullterminate(l)) == len(l) + 1

#def _assert_nullterminate_NullTerminatedByteString(l:listof(isbyte)):
#    return all(tmap(isbyte, nullterminate(l)))

#def _assert_isbyte_Foo():
#    return listof(isbyte)((1,2))



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
