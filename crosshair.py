import types
import functools

def ch(**kwargs):
    return lambda x:x

@ch(use_definition=False)
def istrue(x):
    return x
@ch(axiom=True, pattern=(lambda x:istrue(x)))
def istrue_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(istrue(x), x))

@ch(use_definition=False)
def isbool(x):
    return type(x) is bool
@ch(axiom=True, pattern=(lambda x:isbool(x)))
def isbool_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isbool(x), _z_wrapbool(_z_isbool(x))))

@ch(use_definition=False)
def isdefined(x):
    return True
@ch(axiom=True, pattern=(lambda x:isdefined(x)))
def isdefined_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isdefined(x), _z_wrapbool(_z_isdefined(x))))

@ch(use_definition=False)
def isfunc(x):
    return type(x) is types.LambdaType
@ch(axiom=True, pattern=(lambda x:isfunc(x)))
def isfunc_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isfunc(x), _z_wrapbool(_z_isfunc(x))))

@ch(use_definition=False)
def isint(x):
    return type(x) is int
@ch(axiom=True, pattern=(lambda x:isint(x)))
def isint_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isint(x), _z_wrapbool(_z_isint(x))))

@ch(use_definition=False)
def isnat(x):
    return isint(x) and x >= 0
@ch(axiom=True, patterns=[(lambda x:isint(x)), (lambda x:isnat(x))])
def isnat_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isnat(x), _z_wrapbool(_z_and(_z_isint(x), _z_gte(_z_int(x), _z_int(0))))))

@ch(use_definition=False)
def isstring(x):
    return type(x) == str
@ch(axiom=True, pattern=(lambda x:isstring(x)))
def isstring_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isstring(x), _z_wrapbool(_z_isstring(x))))
    
@ch(use_definition=False)
def istuple(x):
    return type(x) is tuple
@ch(axiom=True, pattern=(lambda x:istuple(x)))
def istuple_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(istuple(x), _z_wrapbool(_z_istuple(x))))

@ch(use_definition=False)
def isnone(x):
    return x is None
@ch(axiom=True, pattern=(lambda x:isnone(x)))
def isnone_Z3Definition(x) -> istrue:
    return _z_wrapbool(_z_eq(isnone(x), _z_wrapbool(_z_isnone(x))))


@ch(use_definition=False)
def implies(x, y):
    return bool(y or not x)
@ch(axiom=True, pattern=(lambda x, y:implies(x, y)))
def implies_Z3Definition(x, y) -> istrue:
    return _z_wrapbool(_z_eq(implies(x, y), _z_wrapbool(_z_implies(_z_t(x), _z_t(y)))))


@ch(axiom=True, pattern=(lambda x:_z_t(x)))
def TruthyPredicateDefinition(x) -> istrue:
    '''List all possibilities for truthy values.'''
    return _z_wrapbool(_z_eq(_z_t(x), _z_or(
            _z_eq(x, True),
            _z_and(_z_isint(x), _z_neq(x, 0)),
            _z_and(_z_istuple(x), _z_neq(x, ())),
            _z_and(_z_isstring(x), _z_neq(x, "")),
            _z_isfunc(x),
        ))
    )

@ch(axiom=True, pattern=(lambda x:_z_f(x)))
def FalseyPredicateDefinition(x) -> istrue:
    '''List all possibilities for falsey values.'''
    return _z_wrapbool(_z_eq(_z_f(x), _z_or(
        _z_eq(x, False),
        _z_eq(x, 0),
        _z_eq(x, ()),
        _z_eq(x, ""),
        _z_eq(x, None))))


@ch(use_definition=False)
def _op_Eq(x,  y): ...
@ch(axiom=True, pattern=(lambda x, y: x == y))
def _op_Eq_Z3Definition(x, y) -> istrue:
    # It would be cool if we could do without the isdefined preconditions, but
    # (5 / 0) == (5 / 0) is undef, not True. So we check for definedness in a few ways:
    return _z_wrapbool(_z_implies(_z_or(_z_isdefined(x == y), _z_and(_z_isdefined(x), _z_isdefined(y))),
                                  _z_eq(x == y, _z_wrapbool(_z_eq(x, y)))))

@ch(use_definition=False)
def _op_NotEq(x,  y): ...
@ch(axiom=True, pattern=(lambda x, y: x != y))
def _op_NotEq_Z3Definition(x :isdefined, y :isdefined) -> istrue:
    return _z_wrapbool(_z_implies(_z_or(_z_isdefined(x != y), _z_and(_z_isdefined(x), _z_isdefined(y))),
                                  _z_eq(x != y, _z_wrapbool(_z_neq(x, y)))))


@ch(use_definition=False)
def _op_And(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a and b))
def _op_And_Z3Definition(a, b) -> istrue:
    return _z_wrapbool(_z_eq(a and b, _z_ite(_z_f(a), a, b)))

@ch(use_definition=False)
def _op_Or(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a or b))
def _op_Or_Z3Definition(a :isdefined, b :isdefined) -> istrue:
    return _z_wrapbool(_z_eq(a or b, _z_ite(_z_t(a), a, b)))

@ch(use_definition=False)
def _op_Not(x): ...
@ch(axiom=True, pattern=(lambda x: not x))
def _op_Not_Z3Definition(x :isdefined) -> istrue:
    return _z_wrapbool(_z_eq(not x, _z_wrapbool(_z_f(x))))


@ch(use_definition=False)
def _op_USub(a): ...
@ch(axiom=True, pattern=(lambda a: -a))
def _op_USub_Z3Definition(a :isint) -> istrue:
    return _z_wrapbool(_z_eq(_z_wrapint(_z_negate(_z_int(a))), _op_USub(a)))

# TODO: tuple comparisons
@ch(use_definition=False)
def _op_Lt(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a < b))
def _op_Lt_Z3Definition(a :isint, b :isint) -> istrue:
    return _z_wrapbool(_z_eq(a < b, _z_wrapbool(_z_lt(_z_int(a), _z_int(b)))))

@ch(use_definition=False)
def _op_Gt(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a > b))
def _op_Gt_Z3Definition(a :isint, b :isint) -> istrue:
    return _z_wrapbool(_z_eq(a > b, _z_wrapbool(_z_gt(_z_int(a), _z_int(b)))))

@ch(use_definition=False)
def _op_LtE(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a <= b))
def _op_LtE_Z3Definition(a :isint, b :isint) -> istrue:
    return _z_wrapbool(_z_eq(a <= b, _z_wrapbool(_z_lte(_z_int(a), _z_int(b)))))

@ch(use_definition=False)
def _op_GtE(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a >= b))
def _op_GtE_Z3Definition(a :isint, b :isint) -> istrue:
    return _z_wrapbool(_z_eq(a >= b, _z_wrapbool(_z_gte(_z_int(a), _z_int(b)))))


@ch(axiom=True, pattern=lambda x,a: _z_concat(x, (a,)))
def _tuple_concat_extract_singleton_cons(x:istuple, a:isdefined) -> istrue:
    return _z_wrapbool(_z_eq(_z_concat(x, (a,)), _z_c(x, a)))

@ch(axiom=True, pattern=lambda x,y,a: _z_concat(x, _z_c(y, a)))
def _tuple_concat_extract_cons(x:istuple, y:istuple, a:isdefined) -> istrue:
    return _z_wrapbool(_z_eq(_z_concat(x, _z_c(y, a)), _z_c(_z_concat(x,y), a)))

@ch(axiom=True, use_definition=False, pattern=lambda x, l: x in l)
def _op_In(x :isdefined, l :istuple) -> isbool:
    return x in l
@ch(axiom=True, pattern=lambda x: x in ())
def _op_In_IsFalseOnEmptyContainer(x :isdefined) -> istrue:
    return (x in ()) == False
@ch(axiom=True, pattern=lambda x, l: x in (l + (x,)))
def _op_In_IsTrueOnMatchingSuffix(x :isdefined, l :istuple) -> istrue:
    return x in (l + (x,))
@ch(axiom=True, pattern=lambda x, l, y: x in (l + (y,)))
def _op_In_IsEquivalentWhenRemovingUnequalElementsFromContainer(x  :isdefined, l :istuple, y :isdefined) -> istrue:
    return implies(y != x, (x in (l + (y,))) == (x in l))


@ch(axiom=True, use_definition=False, pattern=(lambda l:len(l)))
def _builtin_len(l:istuple) -> isint:
    return len(l)
@ch(axiom=True)
def _builtin_len_IsZeroOnEmpty() -> istrue:
    return _z_wrapbool(_z_eq(len(()), 0))
@ch(axiom=True, pattern=(lambda x: len((x,))))
def _builtin_len_IsOneOnSingleton(x:isdefined) -> istrue:
    return _z_wrapbool(_z_eq(len((x,)), 1))
@ch(axiom=True, pattern=(lambda x, t: len((*t, x))))
def _builtin_len_ValueOnDecomposition(x:isdefined, t:istuple) -> istrue:
    return _z_wrapbool(_z_eq(len((*t, x)), len(t) + 1))
@ch(axiom=True, pattern=(lambda s: len(s)))
def _builtin_len_Z3DefinitionOnStrings(s :isstring) -> istrue:
    return _z_wrapbool(_z_eq(len(s), _z_wrapint(_z_length(_z_string(s)))))


@ch(use_definition=False)
def _op_Add(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a + b))
def _op_Add_Z3DefinitionOnInts(a :isint, b :isint) -> istrue:
    return _z_wrapbool(_z_eq(a + b, _z_wrapint(_z_add(_z_int(a), _z_int(b)))))
@ch(axiom=True, pattern=(lambda a, b: a + b))
def _op_Add_Z3DefinitionOnStrings(a :isstring, b :isstring) -> istrue:
    return _z_wrapbool(_z_eq(a + b, _z_wrapstring(_z_add(_z_string(a), _z_string(b)))))
@ch(axiom=True, pattern=(lambda a, b: a + b))
def _op_Add_Z3DefinitionOnTuples(a :istuple, b :istuple) -> istrue:
    return _z_wrapbool(_z_eq(a + b, _z_concat(a, b)))
@ch(axiom=True, pattern=(lambda a, b, x: x in (a + b)))
def _op_Add_ConcatenationPreservesContainment(a :istuple, b :istuple, x :isdefined) -> istrue:
    # Everything in a+b is in a or is in b (set usage)
    return (x in (a + b))  ==  (x in a or x in b)
@ch(axiom=True, patterns=[(lambda a, b: len(a + b)), (lambda a, b: len(a + b))])
def _op_Add_ConcatenationSize(a :istuple, b :istuple) -> istrue:
    # Size after concatenation (bag usage)
    return len(a + b)  ==  len(a) + len(b)

@ch(use_definition=False)
def _op_Sub(a, b): ...
@ch(axiom=True, pattern=(lambda a, b: a - b))
def _op_Sub_Z3Definition(a :isint, b:isint) -> istrue:
    return _z_wrapbool(
        _z_eq(a - b, _z_wrapint(_z_sub(_z_int(a), _z_int(b)))))

@ch(axiom=True, use_definition=False, pattern=lambda l:any(l))
def _builtin_any(l:istuple) -> isbool: ...

@ch(use_definition=False)
def _builtin_all(t):
    return all(t)
@ch(axiom=True, pattern=(lambda t: all(t)))
def all_Definition(t) -> istrue:
    return _z_wrapbool(_z_eq(all(t), _z_ite(_z_eq(t, ()), True, _z_wrapbool(_z_and(_z_t(_z_head(t)), _z_t(all(_z_tail(t))))))))
@ch(axiom=True, pattern=(lambda t1, t2: all(t1 + t2)))
def _builtin_all_DistributeOverConcatenation(t1 :istuple, t2 :istuple) -> istrue:
    return all(t1 + t2) == (all(t1) and all(t2))


@ch(axiom=True, use_definition=False, pattern=lambda x:trange(x))
def trange(x:isint) -> istuple:
    return tuple(range(x))
#@ch(axiom=True, pattern=(lambda x:trange(x)))
#def trange_GivesNaturalNumbers(x :isint) -> istrue:
#    return all(tmap(isnat, trange(x)))
@ch(axiom=True, pattern=(lambda x:trange(x)))
def trange_Definition(x :isint) -> istrue:
    return _z_wrapbool(_z_eq(trange(x), _z_ite(_z_lte(_z_int(x), _z_int(0)), (), (*trange(x-1), _z_wrapint(_z_sub(_z_int(x), _z_int(1)))))))

@ch(use_definition=False)
def _op_Get(l, i): ...
@ch(axiom=True, pattern=lambda l, i: isdefined(l[i]))
def _op_Get_DefinedWhen(l, i :isint) -> istrue:
    return implies((istuple(l) or isstring(l)) and (-len(l) <= i < len(l)), isdefined(l[i]))
@ch(axiom=True, pattern=lambda t, x, i: _z_c(t, x)[i])
def _op_Get_LastOnTuple(t, x, i:isint) -> istrue:
    return _z_c(t, x)[i] == (x if (i == len(t) or i == -1) else t[i if i>=0 else i + 1])


@ch(axiom=True, pattern=[(lambda s, i: s[i]), (lambda s, i: isstring(s))])
def _op_Get_OnString(s :isstring, i :isint) -> istrue:
    return implies(0 <= i < len(s), s[i] == _z_wrapstring(_z_extract(_z_string(s), _z_int(i), _z_int(i+1))))
@ch(axiom=True, pattern=[(lambda s, i: s[i]), (lambda s, i: isstring(s))])
def _op_Get_NegativeOnString(s :isstring, i :isint) -> istrue:
    return implies(-len(s) <= i < 0, s[i] == _z_wrapstring(_z_extract(_z_string(s), _z_int(len(s)+i), _z_int(len(s)+i+1))))



'''
@ch(axiom=True, use_definition=False)
def _op_Sublist(t, start, end):
    return l[start:end]
@ch(axiom=True)
def _op_Sublist_IsSameOnOpenRange(t :istuple) -> istrue:
    return t[0:] == t
@ch(axiom=True)
def _op_Sublist_IsEmptyOnZeroMax(t :istuple) -> istrue:
    return t[:0] == ()
@ch(axiom=True)
def _op_Sublist_IsEmptyOnRangeWithEqualMinAndMax(t :istuple, i :isint) -> istrue:
    return t[i:i] == ()
#def _op_Sublist_PreservesInputWhenConcatenatingASplit(t :istuple, i :isint) -> istrue:
#    return t == t[:i] + t[i:]
@ch(axiom=True, pattern=(lambda s, i, j: s[i:j]))
def _op_Sublist_Z3DefinitionOnStrings(s :isstring, i :isnat, j :isint) -> istrue:
    return _z_wrapbool(_z_eq(s[i:j], _z_wrapstring(_z_extract(_z_string(s), _z_int(i), _z_int(j)))))
'''


