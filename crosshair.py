import types
import functools


def isdefined(x):
    return True
def _assert_isdefined(x):
    return _z_wrapbool(_z_implies(_z_t(isdefined(x)), _z_or(_z_f(x),_z_t(x))))

def isbool(x):
    return type(x) is bool
def _assert_isbool(x :isdefined):
    return isbool(isbool(x))
def _assert_isbool(x :isdefined):
    return _z_wrapbool(_z_eq(_z_T(isbool(x)), _z_or(_z_eq(x, False), _z_eq(x, True))))
def _assert_isbool(x): # booleans are defined
    return _z_wrapbool(_z_implies(_z_T(isbool(x)), _z_T(isdefined(x))))

def isint(x :isdefined) -> (isbool):
    return type(x) is int
def _assert_isint(x): # ints are defined
    return _z_wrapbool(_z_implies(_z_T(isint(x)), _z_T(isdefined(x))))

def isnat(x :isdefined) -> (isbool):
    return isint(x) and x >= 0
def _assert_isnat(x :isnat): # TODO revisit
    return isint(x)
def _assert_isnat(x :isnat): # TODO revisit
    return x > 0

def istuple(x :isdefined) -> (isbool):
    return type(x) is tuple
def _assert_istuple():
    return _z_wrapbool(_z_Eq(istuple(()), True))
def _assert_istuple(x:isdefined):
    return _z_wrapbool(_z_Eq(istuple((x,)), True))
def _assert_istuple(x:isdefined, t:istuple):
    return _z_wrapbool(_z_Eq(istuple((*t, x)), True))
def _assert_istuple(x): # tuples are defined
    return _z_wrapbool(_z_implies(_z_T(istuple(x)), _z_T(isdefined(x))))

def isfunc(x :isdefined) -> (isbool):
    return type(x) is types.LambdaType
def _assert_isfunc(x): # functions are defined
    return _z_wrapbool(_z_implies(_z_T(isfunc(x)), _z_T(isdefined(x))))

def isnone(x :isdefined) -> (isbool):
    return x is None
def _assert_isnone(x): # "None" is defined
    return _z_wrapbool(_z_implies(_z_T(isnone(x)), _z_T(isdefined(x))))

def reduce(f :isfunc, l :istuple, i):
    return functools.reduce(f, l, i)

def implies(x :isdefined, y :isdefined) -> (isbool):
    return bool(y or not x)
def _assert_implies(x, y):
    return _z_wrapbool(_z_eq(_z_t(implies(x, y)), _z_implies(_z_t(x), _z_t(y))))

def check(val, f :isfunc):
    return val

def _assert_():
    '''Distinctness.'''
    return _z_wrapbool(_z_distinct(True, False, None, 0, ()))

# def _assert_(x):
#     ''' Values are never truthy and falsey. ''' # follows from definition of others?
#     return _z_wrapbool(_z_not(_z_and(_z_t(x), _z_f(x))))
def _assert_(x): # TODO: should there be isdefined guards on these?
    '''List all possibilities for truthy values. '''
    return _z_wrapbool(
        _z_eq(_z_t(x), _z_or(
            _z_eq(x, True),
            _z_and(_z_T(isint(x)), _z_neq(x, 0)),
            _z_and(_z_T(istuple(x)), _z_neq(x, ())),
            _z_T(isfunc(x)),
        ))
    )
def _assert_(x):
    '''List all possibilities for falsey values. '''
    return _z_wrapbool(_z_eq(_z_f(x), _z_or(
        _z_eq(x, False),
        _z_eq(x, 0),
        _z_eq(x, ()),
        _z_eq(x, None))))

# Wonder whether axiomitizing with types as a first class value is easier...
# type(x1) != type(x2) -> x1 != x2
def _assert_(x): # seems unnecessary
    ''' Basic types define disjoint sets. '''
    return _z_wrapbool(_z_and(
        _z_implies(_z_t(isbool(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(istuple(x)), _z_t(isfunc(x)), _z_t(isnone(x))))),
        _z_implies(_z_t(isint(x)), _z_not(_z_or(_z_t(isbool(x)), _z_t(istuple(x)), _z_t(isfunc(x)), _z_t(isnone(x))))),
        _z_implies(_z_t(istuple(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(isbool(x)), _z_t(isfunc(x)), _z_t(isnone(x))))),
        _z_implies(_z_t(isfunc(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(istuple(x)), _z_t(isbool(x)), _z_t(isnone(x))))),
        _z_implies(_z_t(isnone(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(istuple(x)), _z_t(isfunc(x)), _z_t(isbool(x))))),
    ))


#  In order to close the universe, we need an error type. It's unclear whether
# closing the universe helps anyone.
# def _assert_(x):
#     ''' Every value is of some type. '''
#     #  isint( 4 if 5 / 0 == 0 else 4 )
#     return _z_wrapbool(_z_or(
#         _z_t(isbool(x)),  _z_t(isint(x)), _z_t(istuple(x)), _z_t(isfunc(x)), _z_t(isnone(x))))



def _builtin_any(l:istuple) -> (isbool): ...

def _builtin_all(t :istuple) -> (isbool): ...
def _assert__builtin_all():
    return _z_wrapbool(_z_eq(all(()), True))
def _assert__builtin_all(t :istuple, x :isdefined):
    return _z_wrapbool(_z_implies(_z_T(all(t)),
        _z_Eq(_z_T(all((*t, x))), _z_T(x))))

def _builtin_len(l:istuple) -> (isnat) : ...
def _assert__builtin_len():
    return _builtin_len(()) == 0
def _assert__builtin_len(x:isdefined): # TODO can this be deduced?
    return _builtin_len((x,)) == 1
# def _assert__builtin_len(x:isdefined, t:istuple): # TODO can this be deduced?
#     return _builtin_len((x, *t)) == _builtin_len(t) + 1
def _assert__builtin_len(x:isdefined, t:istuple): # TODO can this be deduced?
    return _builtin_len((*t, x)) == _builtin_len(t) + 1

def _builtin_map(f, l): ... # TODO assertions. must check f(i) is defined
def _assert__builtin_map(t:istuple, f:isfunc, g:isfunc):
    # (forall (x) f(x)->g(x))   -> (forall t:istuple, all(map(f,t)) -> all(map(g,t)))
    # (exists (x) -g(x) & f(x)) or (...)
    # (-g(skolem(f,g)) & f(skolem(f,g))) or (...)
    # In theory, should be provable with recursion, but just making an axiom for now
    return implies(forall(lambda x:implies(f(x),g(x))),
        implies(all(map(f, t)), all(map(g, t))))
    # return _z_wrapbool(_z_implies(
    #     _z_forall(_, _z_implies(_z_T(f(_)), _z_T(g(_)))),
    #     _z_implies(_z_T(all(map(f,t))), _z_T(all(map(g,t))))
    # ))
def _builtin_range(x:isint) -> (lambda l: all(map(isnat, l))) : ...

def _builtin_filter(f:isfunc, l:istuple): ...
# def _assert__builtin_filter(f, l): # TODO f(i) must be defined for i in l
#     return all(map((lambda i: i in l), filter(f, l)))
# def _assert__builtin_filter(f, l, g):
#     return implies(all(map(g,l)), all(map(g, filter(f, l))))

def _builtin_tuple(*values:lambda l:all(map(isdefined,l))) -> (istuple) : ...

def _op_Sub(a, b): ...
def _assert__op_Sub(a, b):
  return isint(_op_Sub(a, b)) == (isint(a) and isint(b))
# def _assert__op_Sub(a, b): # TODO required?
#   return _z_wrapbool(_z_implies(_z_t(isint(_)), _z_eq(_z_int(_), _z_sub(_z_int(a),_z_int(b)))))

def _op_Add(a, b): ...
def _assert__op_Add(a, b):
    return isint(_op_Add(a, b)) == (isint(a) and isint(b))
def _assert__op_Add(a :isint, b :isint):
    return _z_wrapbool(
        _z_eq(_op_Add(a, b), _z_wrapint(_z_add(_z_int(a), _z_int(b)))))

def _op_Eq(x :isdefined,  y :isdefined) -> isbool: ...
def _assert__op_Eq(x, y):
    return _z_wrapbool(_z_eq(_z_t(_op_Eq(x, y)), _z_eq(x, y)))

def _op_NotEq(a :isdefined,  b :isdefined) -> isbool: ...
def _assert__op_NotEq(a, b):
    return _z_wrapbool(_z_eq(_z_t(_op_NotEq(a, b)), _z_neq(a, b)))

# TODO: tuple comparisons
def _op_Lt(a, b): ...
def _assert__op_Lt(a :isint, b :isint):
    return _z_wrapbool(_z_eq(_z_t(_op_Lt(a, b)), _z_lt(_z_int(a), _z_int(b))))

def _op_Gt(a, b): ...
def _assert__op_Gt(a :isint, b :isint):
    return _z_wrapbool(_z_eq(_z_t(_op_Gt(a, b)), _z_gt(_z_int(a), _z_int(b))))

def _op_LtE(a, b): ...
def _assert__op_LtE(a :isint, b :isint):
    return _z_wrapbool(_z_eq(_z_t(_op_LtE(a, b)), _z_lte(_z_int(a), _z_int(b))))

def _op_GtE(a, b): ...
def _assert__op_GtE(a :isint, b :isint):
    return _z_wrapbool(_z_eq(_z_t(_op_GtE(a, b)), _z_gte(_z_int(a), _z_int(b))))

def _op_And(a :isdefined, b :isdefined) -> isbool: ...
def _assert__op_And(a, b):
    return _z_wrapbool(_z_eq(_z_t(_op_And(a, b)), _z_and(_z_t(a), _z_t(b))))
def _assert__op_And(a, b):
    return _z_wrapbool(_z_eq(_z_t(_op_And(a, b)), _z_and(_z_t(a), _z_t(b))))

def _op_Or(a :isdefined, b :isdefined) -> isbool: ...
def _assert__op_Or(a :isdefined, b :isdefined):
    return _z_wrapbool(_z_eq(_z_t(_op_Or(a, b)), _z_or(_z_t(a), _z_t(b))))
def _assert__op_Or(a :isdefined, b :isdefined):
    return _z_wrapbool(_z_implies(_z_t(a), _z_eq(_op_Or(a, b), a)))
def _assert__op_Or(a :isdefined, b :isdefined):
    return _z_wrapbool(_z_implies(_z_f(a), _z_eq(_op_Or(a, b), b)))

def _op_Not(x :isdefined) -> isbool: ...
def _assert__op_Not(x :isdefined):
    return _z_wrapbool(_z_eq(_z_t(_op_Not(x)), _z_f(x)))
def _assert__op_Not(x :isbool): # these both seem necessary, and I don't know why exactly:
    return _z_wrapbool(_z_eq(_z_t(_op_Not(x)), _z_not(_z_bool(x))))
def _assert__op_Not(x :isbool):
    return _z_wrapbool(_z_eq(_z_f(_op_Not(x)), _z_bool(x)))
def _assert__op_Not(x :isdefined):
    return _z_wrapbool(_z_eq(_z_eq(False,_op_Not(x)), _z_t(x)))
def _assert__op_Not(x :isdefined):
    return _z_wrapbool(_z_eq(_z_eq(True,_op_Not(x)), _z_f(x)))

def _op_Get(l, i):
    return l[i]
def _assert__op_Get(l :istuple, i :isnat, f :isfunc):
    return implies(all(map(f, l)) and 0 <= i < len(l), f(_op_Get(l, i)))

def _op_In(x :isdefined, l :istuple) -> isbool:
    return x in l
def _assert__op_In(x :isdefined, l :istuple):
    return implies(l == (), _op_In(x, l) == False)
def _assert__op_In(x :isdefined, l :istuple):
    return _op_In(x, l + (x,))
def _assert__op_In(x  :isdefined, l :istuple, y :isdefined):
    return implies(y != x, _op_In(x, l) == _op_In(x, l + (y,)))
