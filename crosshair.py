import types
import functools

# The builtin is*() predicates swallow exception values.
# We do this so that, by default, function declarations are also a presumed
# assertion about not-thowing-an-exception.
# Normally, we wouldn't want this behavior in runtime code, but the whole point
# of this setup is that we've proven the absence of exceptions already.
_logical_exceptions = (
    # Needless to say, some exceptions we want to let fly no matter what
    # (memory, kill signals, etc). This is the whitelist of exceptions that
    # we'll catch.
    ArithmeticError,
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    NameError,
    ReferenceError,
    RuntimeError,
    TypeError,
    ValueError,
)

def ch_pattern(*fns):
    return lambda x:x
def ch_weight(*weight):
    return lambda x:x
def isbool(x):
    try:
        return type(x) is bool
    except e: # TODO repeat this guard for other predicates
        if isinstance(e, _logical_exceptions): return False
        raise

# def _assert_isbool(x :isdefined):
#     return isbool(isbool(x))
def _assert_isbool(x):
    return _z_wrapbool(_z_eq(isbool(x), _z_wrapbool(_z_isbool(x))))
# def _assert_isbool(x :isdefined):
#     return _z_wrapbool(_z_eq(_z_t(isbool(x)), _z_or(_z_eq(x, False), _z_eq(x, True))))
# def _assert_isbool(x): # booleans are defined
    # return _z_wrapbool(_z_implies(_z_t(isbool(x)), _z_t(isdefined(x))))

def isdefined(x) -> (isbool):
    return True
def _assert_isdefined(x):
    return _z_wrapbool(_z_eq(isdefined(x), _z_wrapbool(_z_isdefined(x))))
# def _assert_isdefined(x):
#     return _z_wrapbool(_z_implies(_z_t(isdefined(x)), _z_or(_z_f(x),_z_t(x))))

def isint(x) -> (isbool):
    return type(x) is int
def _assert_isint(x):
    return _z_wrapbool(_z_eq(isint(x), _z_wrapbool(_z_isint(x))))
# def _assert_isint(x): # ints are defined
    # return _z_wrapbool(_z_implies(_z_t(isint(x)), _z_t(isdefined(x))))

def isnat(x) -> (isbool):
    return isint(x) and x >= 0
@ch_pattern(lambda x:isint(x))
@ch_pattern(lambda x:isnat(x))
def _assert_isnat(x):
    return _z_wrapbool(_z_eq(isnat(x), _z_wrapbool(_z_and(_z_isint(x), _z_gte(_z_int(x), _z_int(0))))))
    # return isnat(x) == (isint(x) and x >= 0)

def istuple(x) -> (isbool):
    return type(x) is tuple
def _assert_istuple(x):
    return _z_wrapbool(_z_eq(istuple(x), _z_wrapbool(_z_istuple(x))))
# def _assert_istuple():
#     return _z_wrapbool(_z_eq(istuple(()), True))
# def _assert_istuple(x:isdefined):
#     return _z_wrapbool(_z_eq(istuple((x,)), True))
# def _assert_istuple(x:isdefined, t:istuple):
#     return _z_wrapbool(_z_eq(istuple((*t, x)), True))
# def _assert_istuple(x): # tuples are defined
    # return _z_wrapbool(_z_implies(_z_t(istuple(x)), _z_t(isdefined(x))))

def isfunc(x) -> (isbool):
    return type(x) is types.LambdaType # same as types.FuncitonType
def _assert_isfunc(x):
    return _z_wrapbool(_z_eq(isfunc(x), _z_wrapbool(_z_isfunc(x))))
# def _assert_isfunc(x): # functions are defined
    # return _z_wrapbool(_z_implies(_z_t(isfunc(x)), _z_t(isdefined(x))))

def isnone(x) -> (isbool):
    return x is None
def _assert_isnone(x):
    return _z_wrapbool(_z_eq(isnone(x), _z_wrapbool(_z_isnone(x))))
# def _assert_isnone(x): # "None" is defined
#     return _z_wrapbool(_z_implies(_z_t(isnone(x)), _z_t(isdefined(x))))

def reduce(f :isfunc, l :istuple, i):
    return functools.reduce(f, l, i)

def implies(x :isdefined, y :isdefined) -> (isbool):
    return bool(y or not x)
@ch_pattern(lambda x, y:implies(x, y))
def _assert_implies(x :isdefined, y :isdefined):
    return _z_wrapbool(_z_eq(_z_t(implies(x, y)),        _z_implies(_z_t(x), _z_t(y))))
# def _assert_implies(x, y):
#     return _z_wrapbool(_z_eq(_z_f(implies(x, y)), _z_not(_z_implies(_z_t(x), _z_t(y)))))

# def forall(f :isfunc) -> (isbool):
#     raise RuntimeError('Unable to directly execute forall().')
# def _assert_forall(f :isfunc):
#     return _z_wrapbool(_z_eq(_z_t(forall(f)), _z_forall(f)))

def thereexists(f :isfunc) -> (isbool):
    raise RuntimeError('Unable to directly execute thereexists().')
def _assert_thereexists(f :isfunc):
    return _z_wrapbool(_z_eq(_z_t(thereexists(f)), _z_thereexists(f)))

def check(val, f :isfunc):
    return val

# def _assert_():
#     '''Distinctness.'''
#     return _z_wrapbool(_z_distinct(True, False, None, 0, ()))

# def _assert_(x):
#     ''' Values are never truthy and falsey. ''' # follows from definition of others?
#     return _z_wrapbool(_z_not(_z_and(_z_t(x), _z_f(x))))
def _assert_(x): # TODO: should there be isdefined guards on these?
    '''List all possibilities for truthy values. '''
    return _z_wrapbool(_z_eq(_z_t(x), _z_or(
            _z_eq(x, True),
            _z_and(_z_isint(x), _z_neq(x, 0)),
            _z_and(_z_istuple(x), _z_neq(x, ())),
            _z_isfunc(x),
        ))
    )
def _assert_(x):
    '''List all possibilities for falsey values. '''
    return _z_wrapbool(_z_eq(_z_f(x), _z_or(
        _z_eq(x, False),
        _z_eq(x, 0),
        _z_eq(x, ()),
        _z_eq(x, None))))

# # Wonder whether axiomitizing with types as a first class value is easier...
# # type(x1) != type(x2) -> x1 != x2
# def _assert_(x): # seems unnecessary
#     ''' Basic types define disjoint sets. '''
#     return _z_wrapbool(_z_and(
#         _z_implies(_z_t(isbool(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(istuple(x)), _z_t(isfunc(x)), _z_t(isnone(x))))),
#         _z_implies(_z_t(isint(x)), _z_not(_z_or(_z_t(isbool(x)), _z_t(istuple(x)), _z_t(isfunc(x)), _z_t(isnone(x))))),
#         _z_implies(_z_t(istuple(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(isbool(x)), _z_t(isfunc(x)), _z_t(isnone(x))))),
#         _z_implies(_z_t(isfunc(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(istuple(x)), _z_t(isbool(x)), _z_t(isnone(x))))),
#         _z_implies(_z_t(isnone(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(istuple(x)), _z_t(isfunc(x)), _z_t(isbool(x))))),
#     ))


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
    ''' truth of all() on empty tuple. '''
    return _z_wrapbool(_z_eq(all(()), True))
@ch_pattern(lambda t, x:all((*t, x)))
def _assert__builtin_all(t :istuple, x :isdefined):
    ''' truth of all() under decomposition. '''
    return _z_wrapbool(_z_eq(_z_t(all((*t, x))), _z_and(_z_t(x), _z_t(all(t)))))
@ch_pattern(lambda t, x:all((x, *t)))
def _assert__builtin_all(t :istuple, x :isdefined):
    ''' truth of all() under decomposition. '''
    return _z_wrapbool(_z_eq(_z_t(all((x, *t))), _z_and(_z_t(x), _z_t(all(t)))))

def _builtin_len(l:istuple) -> (isnat) : ...
def _assert__builtin_len():
    return _z_wrapbool(_z_eq(_builtin_len(()), 0))
def _assert__builtin_len(x:isdefined): # TODO can this be deduced?
    return _z_wrapbool(_z_eq(_builtin_len((x,)), 1))
# def _assert__builtin_len(x:isdefined, t:istuple): # TODO can this be deduced?
    # return _z_wrapbool(_z_eq(_builtin_len((x, *t)), _builtin_len(t) + 1))
def _assert__builtin_len(x:isdefined, t:istuple): # TODO can this be deduced?
    return _z_wrapbool(_z_eq(_builtin_len((*t, x)), _builtin_len(t) + 1))

def tmap(f, l):
    return tuple(map(f, l))
def _assert_tmap(f:isfunc):
    return tmap(f, ()) == ()
def _assert_tmap(f:isfunc, x:isdefined):
    return tmap(f, (x,)) == (f(x),)
def _assert_tmap(f:isfunc, t:istuple, x:isdefined):
    return tmap(f, (x, *t)) == (f(x), *tmap(f, t))
def _assert_tmap(f:isfunc, t:istuple, x:isdefined):
    return tmap(f, (*t, x)) == (*tmap(f, t), f(x))
# TODO figure out definedness propagation for map()
# def _assert__builtin_map(t:istuple):
#     return implies(all(tmap(isnat, t)), all(tmap(isint, t)))

@ch_pattern(lambda t, f, g,tx: tmap(f, t), lambda t,f,g,tx: tmap(g, tx))
def _assert_tmap(t:istuple, f:isfunc, g:isfunc, tx):
    '''Map preserves predicate.'''
    return _z_wrapbool(_z_implies(
        _z_forall(lambda x:implies(f(x),g(x))),
        _z_implies(_z_t(all(tmap(f, t))), _z_t(all(tmap(g, t))))))

# # all(map(P,t)) & forall(i, P(i) -> R(f(i))  ->  all(map(R,map(f(t))))
# # @ch_pattern(lambda t, f, g,tx: tmap(f, t), lambda t,f,g,tx: tmap(g, tx))
# def _assert_tmap(t:istuple, f:isfunc, g:isfunc, tx):
#     '''Map preserves predicate.'''
#     return _z_wrapbool(_z_implies(
#         _z_forall(lambda x:implies(f(x),g(x))),
#         _z_implies(_z_t(all(tmap(f, t))), _z_t(all(tmap(g, t))))))

    # (forall (x) f(x)->g(x))   -> (forall t:istuple, all(map(f,t)) -> all(map(g,t)))
    # (exists (x) -g(x) & f(x)) or (...)
    # (-g(skolem(f,g)) & f(skolem(f,g))) or (...)
    # In theory, should be provable with recursion, but just making an axiom for now
    # return _z_wrapbool(_z_implies(
    #     _z_forall(_, _z_implies(_z_t(f(_)), _z_t(g(_)))),
    #     _z_implies(_z_t(all(map(f,t))), _z_t(all(map(g,t))))
    # ))

def trange(x:isint) -> (istuple):
    return tuple(range(x))
@ch_pattern(lambda x:trange(x))
def _assert_trange(x:isint):
    return all(tmap(isnat, trange(x)))
@ch_pattern(lambda x:trange(x))
def _assert_trange(x):
    return implies(x <= 0, trange(x) == ())
@ch_pattern(lambda x:trange(x))
def _assert_trange(x:isint):
    return implies(x > 0, trange(x) == trange(x-1) + (x-1,))
# def _builtin_trange(x:isint) -> (lambda l: all(tmap(isnat, l))) : ...

def _builtin_filter(f:isfunc, l:istuple): ...
# def _assert__builtin_filter(f, l): # TODO f(i) must be defined for i in l
#     return all(tmap((lambda i: i in l), filter(f, l)))
# def _assert__builtin_filter(f, l, g):
#     return implies(all(tmap(g,l)), all(tmap(g, filter(f, l))))

def _builtin_tuple(*values:lambda l:all(tmap(isdefined,l))) -> (istuple) : ...

def _op_Sub(a, b): ...
def _assert__op_Sub(a, b):
  return isint(_op_Sub(a, b)) == (isint(a) and isint(b))
def _assert__op_Sub(a :isint, b:isint):
    return _z_wrapbool(
        _z_eq(_op_Sub(a, b), _z_wrapint(_z_sub(_z_int(a), _z_int(b)))))

def _op_Add(a, b): ...
def _assert__op_Add(a, b):
    return isint(_op_Add(a, b)) == (isint(a) and isint(b))
def _assert__op_Add(a, b):
    return istuple(_op_Add(a, b)) == (istuple(a) and istuple(b))
def _assert__op_Add(a :isint, b :isint):
    return _z_wrapbool(
        _z_eq(_op_Add(a, b), _z_wrapint(_z_add(_z_int(a), _z_int(b)))))

# TODO: We probably want an axiomization of concatenation that is More
# amenable to inductive proof (?)
def _assert__op_Add(a :istuple, b :istuple, x :isdefined): # provable from below?
    ''' Everything in a+b is in a or is in b (set usage) '''
    return (x in (a + b))  ==  (x in a or x in b)
def _assert__op_Add(a :istuple, b :istuple):
    ''' transition tuple addition to internal '''
    return a + b == _z_concat(a, b)
def _assert__op_Add(a :istuple, b :istuple): # provable from below?
    ''' Size after concatenation (bag usage) '''
    return len(a + b)  ==  len(a) + len(b)
# TODO: Easier to define slice in terms of add than the other way around:
# def _assert__op_Add(a :istuple, b :istuple):
#     ''' Concatenation preserves element ordering, left side '''
#     return (a + b)[:len(a)]  ==  a
# def _assert__op_Add(a :istuple, b :istuple):
#     ''' Concatenation preserves element ordering, right side '''
#     return (a + b)[len(a):]  ==  b


def _op_Eq(x :isdefined,  y :isdefined) -> isbool: ...
@ch_pattern(lambda x, y: x == y)
def _assert__op_Eq(x, y):
    return _z_wrapbool(_z_eq(_z_t(_op_Eq(x, y)), _z_eq(x, y)))

def _op_NotEq(a :isdefined,  b :isdefined) -> isbool: ...
@ch_pattern(lambda a, b: a != b)
def _assert__op_NotEq(a :isdefined, b :isdefined):
    return _z_wrapbool(_z_eq(_op_NotEq(a, b), _z_wrapbool(_z_neq(a, b))))

# TODO: tuple comparisons
def _op_Lt(a, b): ...
def _assert__op_Lt(a :isint, b :isint):
    return _z_wrapbool(_z_eq(_op_Lt(a, b), _z_wrapbool(_z_lt(_z_int(a), _z_int(b)))))

def _op_Gt(a, b): ...
def _assert__op_Gt(a :isint, b :isint):
    return _z_wrapbool(_z_eq(_op_Gt(a, b), _z_wrapbool(_z_gt(_z_int(a), _z_int(b)))))

def _op_LtE(a, b): ...
def _assert__op_LtE(a :isint, b :isint):
    return _z_wrapbool(_z_eq(_op_LtE(a, b), _z_wrapbool(_z_lte(_z_int(a), _z_int(b)))))

def _op_GtE(a, b): ...
def _assert__op_GtE(a :isint, b :isint):
    return _z_wrapbool(_z_eq(_op_GtE(a, b), _z_wrapbool(_z_gte(_z_int(a), _z_int(b)))))

def _op_And(a :isdefined, b :isdefined) -> (isbool): ...
def _assert__op_And(a, b):
    return _z_wrapbool(_z_eq(_z_t(_op_And(a, b)),        _z_and(_z_t(a), _z_t(b))))

def _assert__op_And(a, b):
    return _z_wrapbool(_z_eq(_z_f(_op_And(a, b)), _z_not(_z_and(_z_t(a), _z_t(b)))))

def _op_Or(a :isdefined, b :isdefined) -> (isbool): ...
def _assert__op_Or(a :isdefined, b :isdefined):
    return _z_wrapbool(_z_eq(_z_t(_op_Or(a, b)),        _z_or(_z_t(a), _z_t(b))))
def _assert__op_Or(a :isdefined, b :isdefined):
    return _z_wrapbool(_z_eq(_z_f(_op_Or(a, b)), _z_not(_z_or(_z_t(a), _z_t(b)))))

def _op_Not(x :isdefined) -> (isbool): ...
@ch_pattern(lambda x: _op_Not(x))
def _assert__op_Not(x :isdefined):
    return _z_wrapbool(_z_eq(_op_Not(x), _z_wrapbool(_z_f(x))))

def _op_Get(l, i):
    return l[i]
def _assert__op_Get(l :istuple, i :isnat, f :isfunc):
    return implies(all(tmap(f, l)) and 0 <= i < len(l), f(_op_Get(l, i)))

def _op_In(x :isdefined, l :istuple) -> (isbool):
    return x in l
def _assert__op_In(x :isdefined, l :istuple):
    return implies(l == (), _op_In(x, l) == False)
def _assert__op_In(x :isdefined, l :istuple):
    return _op_In(x, l + (x,))
def _assert__op_In(x  :isdefined, l :istuple, y :isdefined):
    return implies(y != x, _op_In(x, l + (y,)) == _op_In(x, l))

def _op_Sublist(t :istuple, start :isint, end :isint) -> (istuple):
    return l[start:end]
def _assert__op_Sublist(t :istuple):
    return l[0:] == l
def _assert__op_Sublist(t :istuple):
    return l[:0] == ()
def _assert__op_Sublist(t :istuple, i :isint):
    return l[i:i] == ()
def _assert__op_Sublist(t :istuple, i :isint):
    return t == t[:i] + t[i:]
