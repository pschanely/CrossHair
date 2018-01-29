import types
import functools

def ch(**kwargs):
    return lambda x:x

@ch(axiom=True, use_definition=False)
def istrue(x):
    return x
@ch(axiom=True, pattern=(lambda x:istrue(x)))
def _assert_istrue_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(istrue(x), x))

@ch(axiom=True, use_definition=False)
def isbool(x):
    return type(x) is bool
@ch(axiom=True, pattern=(lambda x:isbool(x)))
def _assert_isbool_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isbool(x), _z_wrapbool(_z_isbool(x))))

@ch(axiom=True, use_definition=False)
def isdefined(x) -> isbool:
    return True
@ch(axiom=True)
def _assert_isdefined_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isdefined(x), _z_wrapbool(_z_isdefined(x))))

@ch(axiom=True, use_definition=False)
def isfunc(x) -> (isbool):
    return type(x) is types.LambdaType # same as types.FunctionType
@ch(axiom=True, pattern=(lambda x:isfunc(x)))
def _assert_isfunc_Z3Definition(x):
    return _z_wrapbool(_z_eq(isfunc(x), _z_wrapbool(_z_isfunc(x))))

'''
@ch(axiom=True, use_definition=False)
def isint(x) -> isbool:
    return type(x) is int
@ch(axiom=True, pattern=(lambda x:isint(x)))
def _assert_isint_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isint(x), _z_wrapbool(_z_isint(x))))

@ch(axiom=True, use_definition=False)
def isnat(x) -> isbool:
    return isint(x) and x >= 0
@ch(axiom=True, patterns=[(lambda x:isint(x)), (lambda x:isnat(x))])
def _assert_isnat_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isnat(x), _z_wrapbool(_z_and(_z_isint(x), _z_gte(_z_int(x), _z_int(0))))))

@ch(axiom=True, use_definition=False)
def isstring(x) -> isbool:
    return type(x) == str
@ch(axiom=True, pattern=(lambda x:isstring(x)))
def _assert_isstring_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isstring(x), _z_wrapbool(_z_isstring(x))))
    
@ch(axiom=True, use_definition=False)
def istuple(x) -> isbool:
    return type(x) is tuple
@ch(axiom=True, pattern=(lambda x:istuple(x)))
def _assert_istuple_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(istuple(x), _z_wrapbool(_z_istuple(x))))

@ch(axiom=True, use_definition=False)
def isnone(x) -> isbool:
    return x is None
@ch(axiom=True, pattern=(lambda x:isnone(x)))
def _assert_isnone_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isnone(x), _z_wrapbool(_z_isnone(x))))


@ch(axiom=True, use_definition=False)
def listof(pred :isfunc) -> isfunc:
    return lambda t: istuple(t) and all(tmap(pred, t))
@ch(axiom=True, pattern=(lambda pred, x: listof(pred)(x)))
def _assert_listof_Definition(pred, x) -> istrue:
    return _z_wrapbool(_z_eq(listof(pred)(x), istuple(x) and all(tmap(pred, x))))

@ch(axiom=True, use_definition=False)
def reduce(f :isfunc, l :istuple, i):
    return functools.reduce(f, l, i)

@ch(axiom=True, use_definition=False)
def implies(x, y) -> isbool:
    return bool(y or not x)
@ch(axiom=True, pattern=(lambda x, y:implies(x, y)))
def _assert_implies_Z3Definition(x, y) -> istrue:
    return _z_wrapbool(_z_eq(implies(x, y), _z_wrapbool(_z_implies(_z_t(x), _z_t(y)))))
    #return _z_wrapbool(_z_eq(_z_t(implies(x, y)), _z_implies(_z_t(x), _z_t(y))))

'''

# def forall(f :isfunc) -> isbool:
#     raise RuntimeError('Unable to directly execute forall().')
# @ch(axiom=True)
# def _assert_forall(f :isfunc):
#     return _z_wrapbool(_z_eq(_z_t(forall(f)), _z_forall(f)))

#@ch(axiom=True, use_definition=False)
#def thereexists(f :isfunc) -> isbool:
#    raise RuntimeError('Unable to directly execute thereexists().')
#@ch(axiom=True)
#def _assert_thereexists_Z3Definition(f :isfunc) -> istrue:
#    return _z_wrapbool(_z_eq(_z_t(thereexists(f)), _z_thereexists(f)))


@ch(axiom=True)
def _assert_TruthyPredicateDefinition(x) -> istrue:
    '''List all possibilities for truthy values. '''
    return _z_wrapbool(_z_eq(_z_t(x), _z_or(
            _z_eq(x, True),
            _z_and(_z_isint(x), _z_neq(x, 0)),
            _z_and(_z_istuple(x), _z_neq(x, ())),
            _z_and(_z_isstring(x), _z_neq(x, "")),
            _z_isfunc(x),
        ))
    )

@ch(axiom=True)
def _assert_FalseyPredicateDefinition(x) -> istrue:
    '''List all possibilities for falsey values. '''
    return _z_wrapbool(_z_eq(_z_f(x), _z_or(
        _z_eq(x, False),
        _z_eq(x, 0),
        _z_eq(x, ()),
        _z_eq(x, ""),
        _z_eq(x, None))))


#def _assert_TruthyFalseOrUndef(x):
#    return _z_wrapbool(_z_or(_z_eq(x, _z_wrapbool(_z_isundefined(x))), _z_f(x), _z_t(x)))

# # Wonder whether axiomitizing with types as a first class value is easier...
# # type(x1) != type(x2) -> x1 != x2
# def _assert_(x): # seems unnecessary
#     # Basic types define disjoint sets.
#     return _z_wrapbool(_z_and(
#         _z_implies(_z_t(isbool(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(istuple(x)), _z_t(isfunc(x)), _z_t(isnone(x))))),
#         _z_implies(_z_t(isint(x)), _z_not(_z_or(_z_t(isbool(x)), _z_t(istuple(x)), _z_t(isfunc(x)), _z_t(isnone(x))))),
#         _z_implies(_z_t(istuple(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(isbool(x)), _z_t(isfunc(x)), _z_t(isnone(x))))),
#         _z_implies(_z_t(isfunc(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(istuple(x)), _z_t(isbool(x)), _z_t(isnone(x))))),
#         _z_implies(_z_t(isnone(x)), _z_not(_z_or(_z_t(isint(x)), _z_t(istuple(x)), _z_t(isfunc(x)), _z_t(isbool(x))))),
#     ))


'''


@ch(axiom=True, use_definition=False)
def _builtin_any(l:istuple) -> isbool: ...

@ch(axiom=True, use_definition=False)
def _builtin_all(t :istuple) -> isbool:
    return all(t)
@ch(axiom=True)
def _assert__builtin_all_IsTrueOnEmpty() -> istrue:
    return _z_wrapbool(_z_eq(all(()), True))
@ch(axiom=True, pattern=(lambda t, x:all((*t, x))))
def _assert__builtin_all_TruthOnDecomposition(t :istuple, x :isdefined) -> istrue:
    return _z_wrapbool(_z_eq(_z_t(all((*t, x))), _z_and(_z_t(x), _z_t(all(t)))))
@ch(axiom=True, pattern=(lambda t, x:all((x, *t))))
def _assert__builtin_all_TruthOnDecompositionFromLeft(t :istuple, x :isdefined) -> istrue:
    return _z_wrapbool(_z_eq(_z_t(all((x, *t))), _z_and(_z_t(x), _z_t(all(t)))))
@ch(axiom=True, pattern=(lambda t1, t2: all(t1 + t2)))
def _assert__builtin_all_DistributeOverConcatenation(t1 :istuple, t2 :istuple) -> istrue:
    return all(t1 + t2) == (all(t1) and all(t2))
@ch(axiom=True, pattern=(lambda t, f: all(tmap(f, t))))
def _assert__builtin_all_TrueForAnyInTuple(t :istuple, f :isfunc) -> istrue:
    return _z_wrapbool(_z_forall(lambda x:implies(all(tmap(f, t)) and x in t, f(x))))

@ch(axiom=True, use_definition=False)
def _op_In(x :isdefined, l :istuple) -> isbool:
    return x in l
@ch(axiom=True, pattern=lambda x, l: x in l)
def _assert__op_In_IsFalseOnEmptyContainer(x :isdefined, l :istuple) -> istrue:
    return implies(l == (), (x in l) == False)
@ch(axiom=True, pattern=lambda x, l: x in (l + (x,)))
def _assert__op_In_IsTrueOnMatchingSuffix(x :isdefined, l :istuple) -> istrue:
    return x in (l + (x,))
@ch(axiom=True, pattern=lambda x, l, y: x in (l + (y,)))
def _assert__op_In_IsEquivalentWhenRemovingUnequalElementsFromContainer(x  :isdefined, l :istuple, y :isdefined) -> istrue:
    return implies(y != x, x in (l + (y,)) == x in l)


@ch(axiom=True, use_definition=False)
def _builtin_len(l:istuple) -> isnat:
    return len(l)
@ch(axiom=True)
def _assert__builtin_len_IsZeroOnEmpty() -> istrue:
    return _z_wrapbool(_z_eq(len(()), 0))
@ch(axiom=True, pattern=(lambda x: len((x,))))
def _assert__builtin_len_IsOneOnSingleton(x:isdefined) -> istrue:
    return _z_wrapbool(_z_eq(len((x,)), 1))
@ch(axiom=True, pattern=(lambda x, t: len((*t, x))))
def _assert__builtin_len_ValueOnDecomposition(x:isdefined, t:istuple) -> istrue:
    return _z_wrapbool(_z_eq(len((*t, x)), len(t) + 1))
@ch(axiom=True, pattern=(lambda s: len(s)))
def _assert__builtin_len_Z3DefinitionOnStrings(s :isstring) -> istrue:
    return _z_wrapbool(_z_eq(len(s), _z_wrapint(_z_length(_z_string(s)))))

@ch(axiom=True, use_definition=False)
def tmap(f, l):
    return tuple(map(f, l))
@ch(axiom=True, pattern=(lambda f: tmap(f, ())))
def _assert_tmap_IsEmptyOnEmpty(f:isfunc) -> istrue:
    return tmap(f, ()) == ()
@ch(axiom=True, pattern=(lambda f, x: tmap(f, (x,))))
def _assert_tmap_ValueOnSingleton(f:isfunc, x:isdefined) -> istrue:
    return tmap(f, (x,)) == (f(x),)
@ch(axiom=True, pattern=(lambda f, t, x: tmap(f, (*t, x))))
def _assert_tmap_ValueOnDecomposition(f:isfunc, t:istuple, x:isdefined) -> istrue:
    return tmap(f, (*t, x)) == (*tmap(f, t), f(x))
@ch(axiom=True, pattern=(lambda f, t, x: tmap(f, (x, *t))))
def _assert_tmap_ValueOnDecompositionFromLeft(f:isfunc, t:istuple, x:isdefined) -> istrue:
    return tmap(f, (x, *t)) == (f(x), *tmap(f, t))
@ch(axiom=True, pattern=(lambda f, t1, t2: tmap(f, (t1 + t2))))
def _assert_tmap_DistributeOverConcatenation(f:isfunc, t1:istuple, t2:istuple) -> istrue:
    return _z_wrapbool(_z_eq(tmap(f, (t1 + t2)), tmap(f, t1) + tmap(f, t2)))
    
# TODO figure out definedness propagation for map()
#@ch(axiom=True, pattern=(lambda t, f: tmap(f, t))
#def _assert_tmap_DefinedWhen(t:istuple, f:isfunc):
#    return _z_wrapbool(_z_implies(
#        _z_forall(lambda x:implies(x in t, isdefined(f(x)))),
#        _z_t(istuple(tmap(f, t)))))

@ch(axiom=True, pattern=[(lambda t, f, g: tmap(g, t)), (lambda t,f,g: all(tmap(f, t)))])
def _assert_tmap_DefinedWhen(t:istuple, f:isfunc, g:isfunc) -> istrue:
    return _z_wrapbool(_z_implies(
        _z_and(_z_forall(lambda x:implies(isdefined(x) and f(x), isdefined(g(x)))),
               _z_t(all(tmap(f, t)))),
        _z_t(istuple(tmap(g, t)))))

@ch(axiom=True, pattern=[(lambda t, f, g,tx: tmap(f, t)), (lambda t,f,g,tx: tmap(g, tx))])
def _assert_tmap_ValuePreservesPredicate(t:istuple, f:isfunc, g:isfunc, tx) -> istrue:
    return _z_wrapbool(_z_implies(
        _z_forall(lambda x:implies(f(x),g(x))),
        _z_implies(_z_t(all(tmap(f, t))), _z_t(all(tmap(g, t))))))

# # all(map(P,t)) & forall(i, P(i) -> R(f(i))  ->  all(map(R,map(f(t))))
# # @ch(axiom=True, pattern=(lambda t, f, g,tx: tmap(f, t), lambda t,f,g,tx: tmap(g, tx))
# def _assert_tmap(t:istuple, f:isfunc, g:isfunc, tx):
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

# TODO: unclear whether range() needs a special variant!
@ch(axiom=True, use_definition=False)
def trange(x:isint) -> istuple:
    return tuple(range(x))
@ch(axiom=True, pattern=(lambda x:trange(x)))
def _assert_trange_HasTruthyElements(x:isint) -> istrue:
    return all(tmap(isnat, trange(x)))
@ch(axiom=True, pattern=(lambda x:trange(x)))
def _assert_trange_IsEmptyOnNegative(x) -> istrue:
    return implies(x <= 0, trange(x) == ())
@ch(axiom=True, pattern=(lambda x:trange(x)))
def _assert_trange_ValuesByInduction(x:isint) -> istrue:
    return implies(x > 0, trange(x) == trange(x-1) + (x-1,))
# def _builtin_trange(x:isint) -> (lambda l: all(tmap(isnat, l))) : ...

@ch(axiom=True, use_definition=False)
def _builtin_filter(f, l): ...
# def _assert__builtin_filter(f, l): # TODO f(i) must be defined for i in l
#     return all(tmap((lambda i: i in l), filter(f, l)))
# def _assert__builtin_filter(f, l, g):
#     return implies(all(tmap(g,l)), all(tmap(g, filter(f, l))))
'''
'''
@ch(axiom=True, use_definition=False)
def _builtin_tuple(*values): ...
#def _builtin_tuple(*values:lambda l:all(tmap(isdefined,l))) -> (istuple) : ...

@ch(axiom=True, use_definition=False)
def _op_Sub(a, b): ...
def _assert__op_Sub_IsIntOnInts(a, b) -> istrue:
  return isint(a - b) == (isint(a) and isint(b))
def _assert__op_Sub_Z3Definition(a :isint, b:isint) -> istrue:
    return _z_wrapbool(
        _z_eq(a - b, _z_wrapint(_z_sub(_z_int(a), _z_int(b)))))
'''
'''
@ch(axiom=True, use_definition=False)
def _op_Add(a, b): ...
def _assert__op_Add_IsIntOnInts(a, b) -> istrue:
    return isint(a + b) == (isint(a) and isint(b))
def _assert__op_Add_IsTupleOnTuples(a, b) -> istrue:
    return istuple(a + b) == (istuple(a) and istuple(b))
def _assert__op_Add_Z3DefinitionOnInts(a :isint, b :isint) -> istrue:
    return _z_wrapbool(_z_eq(a + b, _z_wrapint(_z_add(_z_int(a), _z_int(b)))))
def _assert__op_Add_Z3DefinitionOnStrings(a :isstring, b :isstring) -> istrue:
    return _z_wrapbool(_z_eq(a + b, _z_wrapstring(_z_add(_z_string(a), _z_string(b)))))
    #return _z_wrapbool(_z_eq(a + b, _z_wrapstring(_z_concat(_z_string(a), _z_string(b)))))

# TODO: We probably want an axiomization of concatenation that is More
# amenable to inductive proof (?)
def _assert__op_Add_ConcatenationPreservesContainment(a :istuple, b :istuple, x :isdefined) -> istrue:
    # Everything in a+b is in a or is in b (set usage)
    return (x in (a + b))  ==  (x in a or x in b)
def _assert__op_Add_Z3DefinitionOnTuples(a :istuple, b :istuple) -> istrue:
    # transition tuple addition to internal
    return a + b == _z_concat(a, b)
def _assert__op_Add_ConcatenationSize(a :istuple, b :istuple) -> istrue:
    # Size after concatenation (bag usage)
    return len(a + b)  ==  len(a) + len(b)
'''

@ch(axiom=True, use_definition=False)
def _op_Eq(x :isdefined,  y :isdefined) -> isbool: ...
@ch(axiom=True, pattern=(lambda x, y: x == y))
def _assert__op_Eq_Z3Definition(x, y) -> istrue:
    return _z_wrapbool(_z_eq(_z_t(x == y), _z_eq(x, y)))

@ch(axiom=True, use_definition=False)
def _op_NotEq(a :isdefined,  b :isdefined) -> isbool: ...
@ch(axiom=True, pattern=(lambda a, b: a != b))
def _assert__op_NotEq_Z3Definition(a :isdefined, b :isdefined) -> istrue:
    return _z_wrapbool(_z_eq(a != b, _z_wrapbool(_z_neq(a, b))))


'''
@ch(axiom=True, use_definition=False)
def _op_USub(a :isint) -> isint: ...
@ch(axiom=True, pattern=(lambda a: _op_USub(a)))
def _assert__op_USub_Z3Definition(a :isint) -> istrue:
    return _z_wrapbool(_z_eq(_z_wrapint(_z_negate(_z_int(a))), _op_USub(a)))

# TODO: tuple comparisons
@ch(axiom=True, use_definition=False)
def _op_Lt(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a < b))
def _assert__op_Lt_Z3Definition(a :isint, b :isint) -> istrue:
    return _z_wrapbool(_z_eq(a < b, _z_wrapbool(_z_lt(_z_int(a), _z_int(b)))))

@ch(axiom=True, use_definition=False)
def _op_Gt(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a > b))
def _assert__op_Gt_Z3Definition(a :isint, b :isint) -> istrue:
    return _z_wrapbool(_z_eq(a > b, _z_wrapbool(_z_gt(_z_int(a), _z_int(b)))))

@ch(axiom=True, use_definition=False)
def _op_LtE(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a <= b))
def _assert__op_LtE_Z3Definition(a :isint, b :isint) -> istrue:
    return _z_wrapbool(_z_eq(a <= b, _z_wrapbool(_z_lte(_z_int(a), _z_int(b)))))

@ch(axiom=True, use_definition=False)
def _op_GtE(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a >= b))
def _assert__op_GtE_Z3Definition(a :isint, b :isint) -> istrue:
    return _z_wrapbool(_z_eq(a >= b, _z_wrapbool(_z_gte(_z_int(a), _z_int(b)))))
'''
@ch(axiom=True, use_definition=False)
def _op_And(a :isdefined, b :isdefined) -> isbool: ...
@ch(axiom=True, pattern=(lambda a, b: a and b))
def _assert__op_And_Z3Definition(a, b) -> istrue:
    return _z_wrapbool(_z_eq(_z_t(a and b),        _z_and(_z_t(a), _z_t(b))))
@ch(axiom=True, pattern=(lambda a, b: a and b))
def _assert__op_And_Z3DefinitionWhenFalse(a, b) -> istrue:
    return _z_wrapbool(_z_eq(_z_f(a and b), _z_or(_z_f(a), _z_and(_z_isdefined(a), _z_f(b)))))
    # soundness failure with the following and 'DefinedWhen'
    #return _z_wrapbool(_z_eq(_z_f(a and b), _z_not(_z_and(_z_t(a), _z_t(b)))))
@ch(axiom=True, pattern=(lambda a, b: a and b))
def _assert__op_And_ShortCircuit(a, b) -> istrue:
    return _z_wrapbool(_z_implies(_z_f(a), _z_eq(a, a and b)))
@ch(axiom=True, pattern=(lambda a, b: a and b))
def _assert__op_And_DefinedWhen(a, b) -> istrue:
    return _z_wrapbool(_z_eq(_z_isdefined(a and b), _z_or(_z_f(a), _z_and(_z_isdefined(a), _z_isdefined(b)))))

@ch(axiom=True, use_definition=False)
def _op_Or(a :isdefined, b :isdefined) -> isbool: ...
@ch(axiom=True, pattern=(lambda a, b: a or b))
def _assert__op_Or_Z3Definition(a :isdefined, b :isdefined) -> istrue:
    return _z_wrapbool(_z_eq(_z_t(a or b),        _z_or(_z_t(a), _z_t(b))))
@ch(axiom=True, pattern=(lambda a, b: a or b))
def _assert__op_Or_Z3DefinitionWhenFalse(a :isdefined, b :isdefined) -> istrue:
    return _z_wrapbool(_z_eq(_z_f(a or b), _z_not(_z_or(_z_t(a), _z_t(b)))))
@ch(axiom=True, pattern=(lambda a, b: a or b))
def _assert__op_Or_ShortCircuit(a, b) -> istrue:
    return _z_wrapbool(_z_implies(_z_t(a), _z_eq(a, a or b)))

@ch(axiom=True, use_definition=False)
def _op_Not(x :isdefined) -> isbool: ...
@ch(axiom=True, pattern=(lambda x: not x))
def _assert__op_Not_Z3Definition(x :isdefined) -> istrue:
    return _z_wrapbool(_z_eq(not x, _z_wrapbool(_z_f(x))))

'''
@ch(axiom=True, use_definition=False)
def _op_Get(l, i):
    return l[i]
@ch(axiom=True, pattern=lambda l, i: isdefined(l[i]))
def _assert__op_Get_DefinedWhen(l :istuple, i :isint) -> istrue:
    return implies( 0 <= i < len(l), isdefined(l[i]))
@ch(axiom=True)
def _assert__op_Get_FirstOnTuple(x :isdefined, t :istuple) -> istrue:
    return (x, *t)[0] == x
@ch(axiom=True)
def _assert__op_Get_LastOnTuple(x :isdefined, t :istuple) -> istrue:
    return (*t, x)[-1] == x
@ch(axiom=True)
def _assert__op_Get_ShiftOutFirstOnTuple(x :isdefined, t :istuple, i :isint) -> istrue:
    return implies(i > 0, (x, *t)[i] == t[i - 1])
@ch(axiom=True)
def _assert__op_Get_NegativeOnTuple(t :istuple, i :isint) -> istrue:
    return implies(-len(t) <= i < 0, t[i] == t[len(t) + i])
@ch(axiom=True)
def _assert__op_Get_OnString(s :isstring, i :isint) -> istrue:
    return implies(0 <= i < len(s), s[i] == _z_wrapstring(_z_extract(_z_string(s), _z_int(i), _z_int(i+1))))
@ch(axiom=True)
def _assert__op_Get_NegativeOnString(s :isstring, i :isint) -> istrue:
    return implies(-len(s) <= i < 0, s[i] == s[len(s) + i])
#s[i] == _z_wrapstring(_z_extract(_z_string(s), _z_int(len(s) + i), _z_int(len(s) + i + 1))))

@ch(axiom=True, use_definition=False)
def _op_Sublist(t, start, end):
    return l[start:end]
@ch(axiom=True)
def _assert__op_Sublist_IsSameOnOpenRange(t :istuple) -> istrue:
    return t[0:] == t
@ch(axiom=True)
def _assert__op_Sublist_IsEmptyOnZeroMax(t :istuple) -> istrue:
    return t[:0] == ()
@ch(axiom=True)
def _assert__op_Sublist_IsEmptyOnRangeWithEqualMinAndMax(t :istuple, i :isint) -> istrue:
    return t[i:i] == ()
#def _assert__op_Sublist_PreservesInputWhenConcatenatingASplit(t :istuple, i :isint) -> istrue:
#    return t == t[:i] + t[i:]
@ch(axiom=True, pattern=(lambda s, i, j: s[i:j]))
def _assert__op_Sublist_Z3DefinitionOnStrings(s :isstring, i :isnat, j :isint) -> istrue:
    return _z_wrapbool(_z_eq(s[i:j], _z_wrapstring(_z_extract(_z_string(s), _z_int(i), _z_int(j)))))
'''


