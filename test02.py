from crosshair import *


'''
def tuples1() -> istrue: return (1,2) == (*(1,), 2)
def tuples2() -> istrue: return (1, 2) == (1, *(2,))
def tuples3() -> istrue: return (1,2,2) == (1, *(2,), 2)
def tuples4() -> istrue: return istuple((1, *(2,3)))

def len1() -> istrue: return len(()) == 0
'''

@ch(prove_with=['TruthyPredicateDefinition', '_builtin_len_IsOneOnSingleton', '_builtin_len_ValueOnDecomposition', '_op_Add_Z3DefinitionOnInts', '_op_And_Z3Definition', '_op_Eq_Z3Definition', 'isdefined_Z3Definition', 'isint_Z3Definition', 'istuple_Z3Definition'])
def len2() -> istrue:
    return len((1,2)) == 2
    #return len((1,2)) == len((1,)) + 1  and len((1,2)) == 2

'''
def len3() -> istrue: return isdefined(len((1,3)))
def len4() -> istrue: return len((1, *(2,))) == 2
def len5() -> istrue: return len((1,3)) == 2
def len6() -> istrue: return len((1,*(2,3,4),5)) == 5
def len7(t:istuple) -> istrue: return len(t) < len((*t,1))

def map_empty() -> istrue: return tmap(isint, ()) == ()
def map_literals1() -> istrue: return tmap(isint, (2,3)) == (True, True)
def map_literals2() -> istrue: return all(tmap(isint, (2, 3)))
def map_literals3() -> istrue: return not all(tmap(isint, (2, False)))


def all_true_on_empty() -> istrue: return all(())
def all_on_literals() -> istrue: return all((True, True, True))
def all_ignore_true_values1(t:istuple) -> istrue: return implies(all(t), all((*t, True)))
def all_ignore_true_values2(t:istuple) -> istrue: return implies(all(t), all((True, *t)))
'''

#def get_on_literals1(t:istuple) -> istrue: return implies(len(t)>0, isdefined(t[0]))
#def get_on_literals2() -> istrue:
#    return len((0,1)) == 2
#def get_on_literals3() -> istrue: return (0,1)[0] == 0
#def get_on_literals2() -> istrue: return (0,1,2)[-1] == 2
#def _op_Get_LastOnTuple(x :isdefined, t :istuple) -> istrue:
#    return (*t, x)[len(t)] == x


'''
def range_defined1() -> istrue: return isdefined(trange(5))
def range_isnat(x:isint) -> istrue: return all(tmap(isnat, trange(x)))
def range_literals1() -> istrue: return trange(1) == (0,)
def range_literals2() -> istrue: return trange(2) == (0,1)
def range_isint(x:isint) -> istrue: return all(tmap(isint, trange(x)))
def range_map_lambda(x:isnat) -> istrue: return implies(x>0, trange(x)[-1] == x-1)
'''

'''

def dddaddone(x :isint) -> isint:
    return x + x

#@ch(pattern=lambda x, z: z in trange(x))
def range_map_lambda_lemma1(x :isint, z :isint) -> istrue:
    return implies(z in trange(x), isnat(z))

#def range_map_lambda_lemma1(x :isint, z :isint) -> istrue:
#    return implies(z in trange(x), isnat(z))

#def range_map_lambda_lemma2(z) -> istrue:
#    return implies(isnat(z), z >= 0)

#def range_map_lambda_lemma3(x :isint, z :isint) -> istrue:
#    return implies(z in trange(x), (isnat(z) and z >= 0))

#def range_map_lambda(x :isint, z :isint) -> istrue:
#    return implies(z in trange(x), z >= 0)

#def range_map_lambda_addone(x :isint) -> istrue:
#    return all(map(addone, trange(x)))

#def range_map_lambda(x :isint, z :isint) -> istrue:
#    return implies(z in trange(x), z >= 0)

#def range_map_lambda(x :isint) -> istuple:
#    return tmap(ident, trange(x))
#def range_map_lambda(x :isint) -> istuple:
#    return tmap(addone, trange(x))


                    
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
