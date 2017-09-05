import ast
import collections
import copy
import doctest
import functools
import inspect
import operator
import sys
import z3

import crosshair
from asthelpers import *

'''

Figure out how to apply rules to standard operators (like we do for builtins)
Determine what is and is not representable in the system
Deal with *args and **kwargs
Normalize: assignments, if statements, and returns
Deal with dependent type references as below.
Understand standalone assertions and rewriting.
Assertions and rewriting references must resolve to a specific method.
Understand user guided proof methods.
Handle variable renaming somehow

def longer(a:isseq, b:isseq) -> ((_ == a) or (_ == b)) and len(_) == max(len(a),len(b))

Lambda dropping:
(1) create a registry (enum) of all lambdas
(2) detecting which lambdas to handle is now about the constraints over the enum

Varying objectives:
(1) Correctness validation; typing
(2) Inlining/partial evaluation/Super optimization
(3) Compilation to LLVM / others
(4) integer range optimization
(5) datastructure selection
(6) JSON parse/serialize API optimization
(7) Memory management optimization

'''

class PureScopeTracker(ScopeTracker):
    def resolve(self, node):
        nodetype = type(node)
        if nodetype is ast.Name:
            refname = node.id
            # print('PureScopeTracker', refname, 'scopes', self.scopes)
            if refname[0] != '_':
                if hasattr(builtins, refname):
                    return _pure_defs.get_fn('_builtin_' + refname).get_definition()
                if hasattr(crosshair, refname):
                    return _pure_defs.get_fn(refname).get_definition()
            elif refname.startswith('_z_'):
                zname = refname[3].upper() + refname[4:]
                return getattr(Z, zname)
            return super().resolve(node)
        return node

_SYMCTR = 0
def gensym(name='sym'):
    global _SYMCTR
    ' TODO: How do we ensure symbols cannot possibly conflict? '
    _SYMCTR += 1
    return '{}#{:03d}'.format(name, _SYMCTR)


class PatternVarPreprocessor(ast.NodeTransformer):
    def __init__(self):
        self.mapping = {}
    def visit_Name(self, node):
        nodeid = node.id
        if not nodeid.isupper():
            return node
        else:
            mapping = self.mapping
            astvar = mapping.get(nodeid)
            if astvar is None:
                astvar = '$' + nodeid
                mapping[nodeid] = astvar
            return astcopy(node, id = astvar)

def preprocess_pattern_vars(*nodes):
    processor = PatternVarPreprocessor()
    if len(nodes) == 1:
        return processor.visit(nodes[0])
    else:
        return [processor.visit(node) for node in nodes]

class PatternVarReplacer(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping
    def visit_Name(self, node):
        nodeid = node.id
        if nodeid[0] == '$':
            return copy.deepcopy(self.mapping[nodeid[1:]])
        else:
            return node

def replace_pattern_vars(node, bindings):
    node = copy.deepcopy(node)
    return PatternVarReplacer(bindings).visit(node)

_MATCHER = {}
def matches(node, patt, bind):
    # print(('matched? node=', ast.dump(node)))
    # print(('matched? patt=', ast.dump(patt)))
    typ = type(patt)
    if typ is ast.Name:
        if patt.id[0] == '$':
            bind[patt.id[1:]] = node
            return True
    if typ is not type(node):
        bind.clear()
        return False
    cb = _MATCHER.get(typ)
    if cb:
        return cb(node, patt, bind)
    else:
        raise Exception('Unhandled node type: '+str(typ))
        bind.clear()
        return False

_MATCHER[ast.Call] = lambda n, p, b : (
    matches(n.func, p.func, b) and len(n.args) == len(p.args) and
    all(matches(ni, pi, b) for (ni, pi) in zip(n.args, p.args))
)
_MATCHER[ast.Name] = lambda n, p, b : (
    n.id == p.id
)
_MATCHER[ast.Module] = lambda n, p, b : (
    matches(n.body, p.body, b)
)
_MATCHER[ast.Expr] = lambda n, p, b : (
    matches(n.value, p.value, b)
)
_MATCHER[ast.BinOp] = lambda n, p, b : (
    type(n.op) is type(p.op) and matches(n.left, p.left, b) and matches(n.right, p.right, b)
)
_MATCHER[ast.BoolOp] = lambda n, p, b : (
    type(n.op) is type(p.op) and all(matches(ni, pi, b) for (ni, pi) in zip(n.values, p.values))
)
_MATCHER[ast.Num] = lambda n, p, b : n.n == p.n
_MATCHER[list] = lambda n, p, b: (
    all(matches(ni, pi, b) for (ni, pi) in zip(n,p))
)

_MATCHER[ast.arg] = lambda n, p, b: (
    n.arg == p.arg and n.annotation == p.annotation
)

def _test():
    '''
    >>> bindings = {}
    >>> patt, repl = preprocess_pattern_vars(astparse('0 + X'), astparse('X + 1'))
    >>> unparse(patt)
    '(0 + $X)'
    >>> matches(astparse('0 + 2'), patt, bindings)
    True
    >>> unparse(bindings['X'])
    '2'
    >>> unparse(replace_pattern_vars(repl, bindings))
    '(2 + 1)'
    '''
    pass

def ast_in(item, lst):
    print(unparse(lst), ast.In)
    return ast.Compare(left=item, ops=[ast.In()], comparators=[lst])

_AST_SUB_HASH = {
    ast.BinOp : lambda n: type(n.op),
    ast.Call : lambda n: n.func if isinstance(n.func, str) else 0,
    ast.BoolOp : lambda n: type(n.op)
}
def patthash(node):
    nodetype = type(node)
    return (hash(nodetype) << 8) + hash(_AST_SUB_HASH.get(nodetype, lambda n: 0)(node))

class Replacer(ast.NodeTransformer):
    '''
    Simple tranformer that just uses a lambda.
    '''
    def __init__(self, logic):
        self.logic = logic
    def __call__(self, node):
        return self.visit(node)
    def generic_visit(self, node):
        ret = self.logic(node)
        if (ret is node):
            return super().generic_visit(node)
        else:
            return ret

def patt_to_lambda(patt, argvars):
    if ( # check whether we can skip the lambda...
        type(patt) is ast.Call and
        type(patt.func) is ast.Name and
        len(patt.args) == len(argvars) and
        all(type(a) is ast.Name and
            a.id[0] == '$' and
            a.id[1:].text == v for (a,v) in zip(patt.args, argvars))
    ):
        return patt.func
    varmap = [(v,gensym(v)) for v in argvars]
    return ast.Lambda(
        args=ast.arguments(args=[ast.arg(arg=v2,annotation=None) for _,v2 in varmap],defaults=[],vararg='',kwarg=''),
        body=replace_pattern_vars(patt, {v1:ast.Name(id=v2) for v1,v2 in varmap})
    )

def beta_reduce(node):
    '''
    >>> unparse(beta_reduce(exprparse('(lambda x:x+1)(5)')))
    '(5 + 1)'
    '''
    if type(node) is not ast.Call:
        return node
    func = node.func
    if type(func) is ast.Name:
        return node
    if type(func) is not ast.Lambda:
        raise Exception()
    ret = inline(node, func)
    # print('beta reduce', unparse(node), unparse(ret))
    return ret

class AdvancedRewriter(PureScopeTracker):
    def __init__(self):
        super().__init__()
    def __call__(self, root):
        self.root = root
        self.result = None
        self.visit(root)
        return self.result if self.result else self.root

    def visit_Call(self, node):
        newnode = beta_reduce(node)
        if newnode is not node:
            return self.visit(newnode)
        node = newnode
        callfn = self.resolve(node.func)
        # print('in expr', ast.dump(node))
        # print('function getting called', ast.dump(callfn) if callfn else '')
        if callfn and getattr(callfn,'name',None) == 'reduce' and node is not self.root:
            reducefn = self.resolve(node.args[0])
            print('reduce callback',ast.dump(node.args[0]))
            if isinstance(reducefn, (ast.Lambda, ast.FunctionDef)):
                self.attempt_reduce_fn_xform(node, reducefn, node.args[1], node.args[2])
        return super().generic_visit(node)

    def attempt_reduce_fn_xform(self, callnode, reducefn, inputlist, initializer):
        ancestors = Replacer(lambda n: ast.Name(id='$R') if n is callnode else n)(self.root)
        # print('ancestors', unparse(ancestors))
        argnames = {a.arg for a in reducefn.args.args}
        inverse = lambda n: ast.Call(func=ast.Name(id='inverse*'),args=[n],keywords=[])
        body_with_inverses = Replacer(
            lambda n: inverse(n) if (type(n) is ast.Name and n.id in argnames) else n
        )(fn_expr(reducefn))
        # print('body_with_inverses', unparse(body_with_inverses))
        body_to_simplify = replace_pattern_vars(ancestors, {'R': body_with_inverses})
        # print('body_to_simplify', unparse(body_to_simplify))
        inverse_canceller = WrappedRewriteEngine(basic_simplifier)
        inverse_canceller.add(
            replace_pattern_vars(ancestors, {'R': inverse(ast.Name(id='I'))}),
            ast.Name(id='I'),
            always
        )
        simplified = inverse_canceller.rewrite(body_to_simplify)
        if matches(simplified, ast.Name(id='inverse*'), {}):
            return
        print('success reduce-fn transform:', unparse(simplified))
        # transformation OK
        reducefn.body = simplified
        new_inputlist = ast.Call(func=ast.Name(id='map'), args=[patt_to_lambda(ancestors, ['R']), inputlist], keywords={})
        new_initializer = replace_pattern_vars(ancestors, {'R': initializer})
        new_reduce = ast.Call(func=ast.Name(id='reduce'), args=[reducefn, new_inputlist, new_initializer], keywords={})

        self.result = new_reduce

class RewriteEngine(ast.NodeTransformer):
    def __init__(self):
        self._index = collections.defaultdict(list)
    def lookup(self, hsh):
        return self._index[hsh]
    def add(self, patt, repl, cond):
        patt, repl = preprocess_pattern_vars(patt, repl)
        self.lookup(patthash(patt)).append( (patt, repl, cond) )
    def generic_visit(self, node):
        while True:
            node = super().generic_visit(node)
            newnode = self.rewrite_top(node)
            if newnode is node:
                return node
            node = newnode
    def rewrite_top(self, node):
        while True:
            bind = {}
            matched = False
            for candidate in self.lookup(patthash(node)):
                patt, repl, cond = candidate
                if matches(node, patt, bind):
                    if not cond(bind):
                        continue
                    matched = True
                    newnode = replace_pattern_vars(repl, bind)
                    print('rewrite found ', unparse(patt))
                    print('rewrite', unparse(node), ' => ', unparse(newnode))
                    node = newnode
                    break
            if not matched:
                break
        return node
    def rewrite(self, node):
        return self.visit(node)

class WrappedRewriteEngine(RewriteEngine):
    def __init__(self, inner):
        self.inner = inner
        super().__init__()
    def lookup(self, hsh):
        r1 = self._index[hsh]
        r2 = self.inner.lookup(hsh)
        if not r1:
            return r2
        if not r2:
            return r1
        return r1 + r2

def normalize_binop(bindings):
    f = bindings['F']
    if type(f) is ast.Lambda:
        varnames = {a.arg for a in f.args.args}
        if type(f.body) is ast.BoolOp:
            boolop = f.body
            if (
                len(boolop.values) == 2 and
                all(type(v) is ast.Name for v in boolop.values) and
                varnames == {v.id for v in boolop.values}
            ):
                optype = type(boolop.op)
                if optype is ast.Or:
                    bindings['F'] = f = ast.Name(id='or*')
                elif optype is ast.And:
                    bindings['F'] = f = ast.Name(id='and*')
                else:
                    raise Exception()
                return True
    return False


basic_simplifier = RewriteEngine()
always = lambda x:True
# TODO: rewriting needs to resolve references -
# you cannot just rewrite anything named "isfunc"
# Or can you? Because pure must be imported as * and names cannot be reassigned?
for (patt, repl, condition) in [
    #('ToBool(isbool(X))', 'IsBool(X)', always),
    #('ToBool(isint(X))', 'IsInt(X)', always),
    #('ToBool(X and Y)', 'And(ToBool(X), ToBool(Y))', always),
    #('ToBool(isint(X))', 'IsInt(X)', always),
    # ('IsTruthy(X and Y)', 'And(IsTruthy(X), IsTruthy(Y))', always),
    #('IsTruthy(isint(X))', 'IsInt(X)', always),
    #('IsInt(X + Y)', 'And(IsInt(X), IsInt(Y))', always),
    #('IsBool(isbool(X))', 'Z3True', always),
    # ('IsBool(isnat(X))', 'Z3True', always), # do not need other types
    # ('IsBool(islist(X))', 'Z3True', always),
    # ('IsBool(isfunc(X))', 'Z3True', always),
    # ('isfunc(WrapFunc(X))', 'Z3True', always),
    # ('isbool(WrapBool(X))', 'Z3True', always),
    #('isint(X)', 'Z3True', lambda b:type(b['X']) is ast.Num),

    # ('isbool(isint(X))', 'True', always),
    # ('isnat(X + Y)', 'isnat(X) and isnat(Y)', always),
    # ('all(map(isnat, range(X)))', 'True', always),
    # ('isnat(X)', 'True', lambda b:type(b['X']) is ast.Num),
    # ('reduce(F,L,I)', 'reduce(F,L,I)', normalize_binop),
    # ('reduce(F,L,I)', 'all(L)', reduce_f_and_i('and*', [True])),
    # ('reduce(F,L,I)', 'any(L)', reduce_f_and_i('or*', [False, None])),
    # ('reduce(F,L,I)', 'False', reduce_f_and_i('and*', [False, None])),
    # ('reduce(F,L,I)', 'True', reduce_f_and_i('or*', [True])),
    # ('isbool(all(X))', 'all(map(isbool, X))', always),
    # ('map(F, map(G, L))', 'map(lambda x:F(G(x)), L)', always), # TODO integrate gensym() in here for lambda argument
    # ('all(map(F, L))', 'True', f_is_always_true),
    # # ('all(map(F,filter(G,L)))', 'all(map(GIMPLIESF,L))', mapfilter),
    # # if F(R(x,y)) rewrites to (some) R`(F(x), F(y))
    # # ('F(reduce(R, L, I))', 'reduce(R`, map(F, L), I)', reduce_fn_check),
]:
    basic_simplifier.add(exprparse(patt), exprparse(repl), condition)

# expr = exprparse('IsTruthy(isint(c) and isint(d))')
# rewritten = basic_simplifier.rewrite(expr)
# print('rewrite engine test', unparse(expr), unparse(rewritten))



PyFunc = z3.DeclareSort('PyFunc')
Unk = z3.Datatype('Unk')
Unk.declare('none')
Unk.declare('bool', ('tobool', z3.BoolSort()))
Unk.declare('int', ('toint', z3.IntSort()))
Unk.declare('func', ('tofunc', PyFunc))
Unk.declare('a', ('tl', Unk), ('hd', Unk))
Unk.declare('_') # empty tuple
Unk.declare('undef') # error value
(Unk,) = z3.CreateDatatypes(Unk)
App = z3.Function('.', Unk, Unk, Unk)


class ZHolder(): pass
Z = ZHolder()
Z.Wrapbool = Unk.bool # z3.Function('Wrapbool', z3.BoolSort(), Unk)
Z.Wrapint = Unk.int # z3.Function('Wrapint', z3.IntSort(), Unk)
Z.Wrapfunc = Unk.func # z3.Function('Wrapfunc', PyFunc, Unk)
Z.Bool = Unk.tobool # z3.Function('Bool', Unk, z3.BoolSort())
Z.Int = Unk.toint # z3.Function('Int', Unk, z3.IntSort())
Z.Func = Unk.tofunc # z3.Function('Func', Unk, PyFunc)
Z.Isbool = lambda x:Unk.is_bool(x)
Z.Isint = lambda x:Unk.is_int(x)
Z.Isfunc = lambda x:Unk.is_func(x)
Z.Istuple = lambda x:z3.Or(Unk.is_a(x), Unk.is__(x))
Z.Isnone = lambda x:Unk.is_none(x)
Z.Isdefined = lambda x:z3.Not(Unk.is_undef(x))
Z.Eq = lambda x,y: x == y
Z.Neq = lambda x,y: x != y
Z.Distinct = z3.Distinct
Z.T = z3.Function('T', Unk, z3.BoolSort())
Z.F = z3.Function('F', Unk, z3.BoolSort())
Z.N = Unk.none # z3.Const('None', Unk)
Z.Implies = z3.Implies
Z.And = z3.And
Z.Or = z3.Or
Z.Not = z3.Not
Z.Lt = lambda x,y: x < y
Z.Lte = lambda x,y: x <= y
Z.Gt = lambda x,y: x > y
Z.Gte = lambda x,y: x >= y
Z.Add = lambda x,y: x + y
Z.Sub = lambda x,y: x - y
Z.Concat = z3.Function('Concat', Unk, Unk, Unk)
# forall and exists are syntactically required to contain a lambda with one argument
Z.Forall = z3.ForAll
Z.Thereexists = z3.Exists

_z3_name_constants = {
    True: Z.Wrapbool(True),
    False: Z.Wrapbool(False),
    None: Z.N,
}

_fndef_to_moduleinfo = {}
def get_scope_for_def(fndef):
    return _fndef_to_moduleinfo[fndef].get_scope_for_def(fndef)

class ModuleInfo:
    def __init__(self, module, module_ast):
        self.module = module
        self.ast = module_ast
        self.functions = {'': FnInfo('', self)} # this is for global assertions, which have no corresponding definitional assertion
    def fn(self, name):
        if name not in self.functions:
            self.functions[name] = FnInfo(name, self)
        return self.functions[name]
    def get_fn(self, name):
        return self.functions[name]
    def get_scope_for_def(self, fndef):
        class FnFinder(PureScopeTracker):
            def __init__(self):
                super().__init__()
                self.hit = None
            def visit_FunctionDef(self, node):
                if node is fndef:
                    self.hit = [copy.deepcopy(s) for s in self.scopes]
                return node
        f = FnFinder()
        f.visit(self.ast)
        return f.hit

def fn_args(fn):
    '''
    Returns a list of positional arguments to be used in a z3 function.
    They are returned as ast.arg objects.
    *args and (later) **kwargs are encoded as single tuple and dict arguments.
    '''
    args = fn.args
    ret = []
    for arg in args.args:
        ret.append(arg)
    for arg in args.kwonlyargs:
        ret.append(arg)
    if args.vararg:
        ret.append(args.vararg)
    if args.kwarg:
        ret.append(args.kwarg)
    return ret

def argument_preconditions(args):
    preconditions = []
    for a in args:
        if not a.annotation: continue
        preconditions.append(astcall(a.annotation, ast.Name(id=a.arg)))
    return preconditions

def astcall(fn, *args):
    return ast.Call(func=fn, args=args, keywords=())

def astand(clauses):
    if len(clauses) == 1:
        return clauses[0]
    else:
        return ast.BoolOp(op=ast.And, values=clauses)

def fn_annotation_assertion(fn):
    args = fn_args(fn)
    preconditions = argument_preconditions(args)
    if not preconditions and not fn.returns:
        return None
    predicate = fn.returns if fn.returns else ast.Name(id='isdefined')
    varnamerefs = [ast.Name(id=a.arg) for a in args]
    expectation = astcall(predicate, astcall(ast.Name(id=fn.name), *varnamerefs))
    fdef = ast.FunctionDef(
        name = '_assertdef_'+fn.name,
        args=fn.args,
        body=[ast.Return(value=expectation)],
        decorator_list=[],
        returns=None
    )
    # print('expectation')
    # print(unparse(fn))
    # print(unparse(fdef))
    return fdef

class FnInfo:
    def __init__(self, name, moduleinfo):
        self.name = name
        self.moduleinfo = moduleinfo
        self.assertions = []
        self.definition = None
        self.definitional_assertion = None
    def add_assertion(self, assertion):
        self.assertions.append(assertion)
        _fndef_to_moduleinfo[assertion] = self.moduleinfo
    def set_definition(self, definition):
        if self.definition is not None:
            raise Exception('multiply defined function: '+str(self.name))
        self.definition = definition
        _fndef_to_moduleinfo[definition] = self.moduleinfo
        definitional_assertion = fn_annotation_assertion(definition)
        if definitional_assertion:
            self.definitional_assertion = definitional_assertion

    def get_definition(self):
        return self.definition
    def get_assertions(self):
        return self.assertions

def parse_pure(module):
    module_ast = ast.parse(open(module.__file__).read())
    ret = ModuleInfo(module, module_ast)
    for item in module_ast.body:
        itemtype = type(item)
        if itemtype == ast.FunctionDef:
            name = item.name
            if name.startswith('_assert_'):
                name = name[len('_assert_'):]
                ret.get_fn(name).add_assertion(item)
            else:
                ret.fn(name).set_definition(item)
    return ret

_pure_defs = parse_pure(crosshair)

def fn_for_op(optype):
    return _pure_defs.get_fn('_op_' + optype).get_definition()

_z3_fn_ids = set(id(x) for x in Z.__dict__.values())

def _merge_arg(accumulator, arg):
    if (type(arg) == ast.Starred):
        if accumulator == Unk._:
            return arg.value
        else:
            return Z.Concat(accumulator, arg.value)
    else:
        return Unk.a(accumulator, arg)

def z3apply(fnval, args):
    if id(fnval) in _z3_fn_ids:
        return fnval(*args)
    else:
        return App(fnval, functools.reduce(_merge_arg, args, Unk._))

class Z3BindingEnv(collections.namedtuple('Z3BindingEnv',['refs','support'])):
    def __new__(cls, refs=None):
        return super(Z3BindingEnv, cls).__new__(cls, refs if refs else {}, [])

class Z3Transformer(PureScopeTracker): #ast.NodeTransformer):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def transform(self, module, fnname):
        pass

    # def visit(self, node):
    #     print('visit', unparse(node))
    #     return super().visit(node)

    def generic_visit(self, node):
        raise Exception('Unhandled ast - to - z3 transformation: '+str(type(node)))

    def register(self, definition):
        # print('register?', definition)
        refs = self.env.refs
        if type(definition) == ast.Name:
            raise Exception('Undefined identifier: "{}" at line {}[:{}]'.format(
                definition.id,
                getattr(definition, 'lineno', ''),
                getattr(definition, 'col_offset', '')))
            return z3.Const(definition.id, Unk)
        if type(definition) == ast.arg:
            name = definition.arg
        elif type(definition) == ast.FunctionDef:
            name = definition.name
        else: # z3 function (z3 functions must be called immediately - they are not values)
            # print('register abort : ', repr(definition))
            return definition
            # if hasattr(definition, 'name'):
            #     name = definition.name()
            # else:
            #     name = definition.__name__
        if definition not in refs:
            # print('register done  : ', str(name))
            # print('register new Unk value', name, definition)
            # print('register.', name)
            refs[definition] = z3.Const(name, Unk)
        return refs[definition]

    def visit_Subscript(self, node):
        return self.handle_subscript(self.visit(node.value), node.slice)

    def handle_subscript(self, value, subscript):
        subscripttype = type(subscript)
        if subscripttype is ast.Index:
            self.env.ops.add(Get)
            return z3apply(fn_for_op('Get'), (value, self.visit(subscript.value)))
        elif subscripttype is ast.Slice:
            if subscript.step is None:
                self.env.ops.add(SubList)
                return z3apply(fn_for_op('SubList'), (value, self.visit(subscript.lower), self.visit(subscript.upper)))
            else:
                self.env.ops.add(SteppedSubList)
                return z3apply(fn_for_op('SteppedSubList'), (value, self.visit(subscript.lower), self.visit(subscript.upper), self.visit(subscript.step)))
        elif subscripttype is ast.ExtSlice:
            return functools.reduce(
                lambda a, b: z3apply(fn_for_op('Add'), (a, b)),
                (self.handle_subscript(value, dim) for dim in index.dims))

    def visit_NameConstant(self, node):
        return _z3_name_constants[node.value]

    def visit_Name(self, node):
        return self.register(self.resolve(node))

    def visit_Starred(self, node):
        # this node will get handled later, in z3apply()
        return ast.Starred(value=self.visit(node.value))

    def visit_BinOp(self, node):
        z3fn = self.register(fn_for_op(type(node.op).__name__))
        left, right = self.visit(node.left), self.visit(node.right)
        return z3apply(z3fn, (left, right))

    def visit_UnaryOp(self, node):
        z3fn = self.register(fn_for_op(type(node.op).__name__))
        val = self.visit(node.operand)
        return z3apply(z3fn, (val,))

    def visit_BoolOp(self, node):
        z3fn = self.register(fn_for_op(type(node.op).__name__))
        args = [self.visit(v) for v in node.values]
        return functools.reduce(lambda a,b:z3apply(z3fn,(a,b)), args)

    def visit_Compare(self, node):
        ret = None
        z3and = lambda : self.register(fn_for_op('And'))
        def add(expr, clause):
            return clause if expr is None else z3apply(z3and(), [clause, expr])
        lastval = self.visit(node.comparators[-1])
        for op, left in reversed(list(zip(node.ops[1:], node.comparators[:-1]))):
            z3fn = self.register(fn_for_op(type(op).__name__))
            left = self.visit(left)
            ret = add(ret, z3apply(z3fn, [left, lastval]))
            lastval = left

        z3fn = self.register(fn_for_op(type(node.ops[0]).__name__))
        ret = add(ret, z3apply(z3fn, [self.visit(node.left), lastval]))
        return ret

    def visit_Num(self, node):
        return Z.Wrapint(node.n)

    def function_body_to_z3(self, func):
        # argnames = [a.arg for a in f.args.args]
        # z3vars = [ast.Const(n, Unk) for n in argnames]
        # z3.ForAll([f.args.args[0].arg], to_z3(f.body,...))
        argnames = [a.arg for a in func.args.args]
        argvars = [z3.Const(name, Unk) for name in argnames]
        arg_name_to_var = dict(zip(argnames, argvars))
        arg_name_to_def = {name: ast.arg(arg=name) for name in argnames}
        self.scopes.append(arg_name_to_def)
        for name, definition in arg_name_to_def.items():
            self.env.refs[definition] = arg_name_to_var[name]
        z3body = self.visit(func.body)
        self.scopes.pop()
        return arg_name_to_var, z3body

    def visit_Lambda(self, node):
        name = 'lambda_{}_{}'.format(node.lineno, node.col_offset)
        funcval = Z.Wrapfunc(z3.Const(name, PyFunc))
        argpairs = [(a.value.arg,True) if type(a)==ast.Starred else (a.arg,False) for a in node.args.args]

        arg_name_to_var, z3body = self.function_body_to_z3(node)

        z3application = z3apply(funcval, [
            ast.Starred(value=arg_name_to_var[name]) if is_starred else arg_name_to_var[name]
            for name, is_starred in argpairs
        ])
        stmt = z3.ForAll([arg_name_to_var[v] for v,_ in argpairs], z3application == z3body)
        self.env.support.append(stmt)
        self.env.support.append(Unk.is_func(funcval))
        return funcval

    def visit_Tuple(self, node):
        #z3fn = self.register(_pure_defs.get_fn('_builtin_tuple').get_definition())
        if type(node.ctx) != ast.Load:
            raise Exception(ast.dump(node))
        params = [self.visit(a) for a in node.elts]
        # print('visit tuple ', *[p for p in params])
        return functools.reduce(_merge_arg, params, Unk._)
        # return functools.reduce(Unk.a, params, Unk._)
        # return z3apply(z3fn, params)

    def visit_Call(self, node):
        newnode = beta_reduce(node)
        if newnode is not node:
            return self.visit(newnode)
        z3fn = self.visit(node.func)
        # Special case for quantifiers:
        if z3fn is z3.ForAll or z3fn is z3.Exists:
            targetfn = node.args[0]
            if type(targetfn) == ast.Name:
                z3varargs = z3.Const(gensym('a'), Unk)
                targetfn = self.visit(targetfn)
                return z3fn([z3varargs], Z.T(App(targetfn, z3varargs)))
            else:
                arg_name_to_var, z3body = self.function_body_to_z3()
                return z3fn([n for n in arg_name_to_var.keys()], z3body)
        params = [self.visit(a) for a in node.args]
        # special case forall & thereexists
        return z3apply(z3fn, params)

def to_z3(node, env, initial_scopes=None):
    '''
    >>> to_z3(exprparse('False'), Z3BindingEnv())
    bool(False)
    >>> to_z3(exprparse('range(4)'), Z3BindingEnv())
    .(_builtin_range, a(_, int(4)))
    >>> to_z3(exprparse('(4,*())'), Z3BindingEnv())
    Concat(a(_, int(4)), _)
    >>> to_z3(exprparse('(*range,4)'), Z3BindingEnv())
    a(_builtin_range, int(4))
    >>> to_z3(exprparse('4 + 0'), Z3BindingEnv())
    .(_op_Add, a(a(_, int(4)), int(0)))
    >>> to_z3(exprparse('True and False'), Z3BindingEnv())
    .(_op_And, a(a(_, bool(True)), bool(False)))
    >>> to_z3(exprparse('(lambda x:True)(7)'), Z3BindingEnv())
    bool(True)
    >>> to_z3(exprparse('0 <= 5'), Z3BindingEnv())
    .(_op_LtE, a(a(_, int(0)), int(5)))
    >>> to_z3(exprparse('0 <= 5 < 9'), Z3BindingEnv()) # doctest: +NORMALIZE_WHITESPACE
    .(_op_And, a(a(_,
      .(_op_LtE, a(a(_, int(0)), int(5)))),
      .(_op_Lt,  a(a(_, int(5)), int(9)))))
    '''
    transformer = Z3Transformer(env)
    if initial_scopes:
        transformer.scopes = initial_scopes
    return transformer.visit(node)

def call_predicate(predicate, target):
    expr = ast.Call(func=predicate, args=[target], keywords=[])
    return basic_simplifier.rewrite(expr)

def to_assertion(expr, target, env, extra_args=()):
    call = call_predicate(expr, target)
    print(unparse(call))
    return to_z3(call, env)

def solve(assumptions, conclusion):
    make_repro_case = ('repro' in sys.argv)
    solver = z3.Solver()
    z3.set_param(
        verbose = 1,
        # mbqi=True,
    )
    solver.set(
        timeout=1000,
        unsat_core=True,
    )
    # solver.set('smt.mbqi', True)
    # solver.set('smt.timeout', 1000),
    # solver.set(auto_config=True)
    # solver.set('macro-finder', True)
    # solver.set('smt.pull-nested-quantifiers', True)

    required_assumptions = [a for a in assumptions if a.value is None]
    assumptions = [a for a in assumptions if a.value is not None]
    assumptions.sort(key=lambda a:a.value)
    assumptions = assumptions[:120]
    for l in assumptions:
        print('baseline:', l)
    assumptions = required_assumptions + assumptions
    # assumptions = [a.expr for a in assumptions]

    if make_repro_case:
        for assumption in assumptions:
            solver.add(assumption)
        solver.add(z3.Not(conclusion))
    else:
        for assumption in assumptions:
            solver.assert_and_track(assumption.expr,  str(assumption))#'assumption: '+str(assumption))
        solver.assert_and_track(z3.Not(conclusion), 'goal')

    core = None
    try:
        ret = solver.check()
        if ret == z3.unsat:
            if not make_repro_case:
                print ('BEGIN PROOF CORE ')
                core = solver.unsat_core()
                for stmt in core:
                    print(' '+str(stmt))
                print ('END PROOF CORE ')
            ret = True
        elif ret == z3.sat:
            print('Counterexample:')
            print(solver.model())
            ret = False
        else:
            ret = None
    except z3.z3types.Z3Exception as e:
        if e.value != b'canceled':
            raise e
        ret = None
    with open('repro.smt2', 'w') as fh:
        if make_repro_case:
            fh.write(solver.sexpr())
            fh.write("\n(check-sat)\n(get-model)\n")
            print('Wrote repro smt file.')
    return ret, core

def assertion_fn_to_z3(fn, env, scopes):
    args = fn_args(fn)
    scopes.append({a.arg:a for a in args})
    z3expr = to_z3(fn_expr(fn), env, scopes)
    if z3expr.decl() == Z.Wrapbool:
        z3expr = z3expr.arg(0)
    else:
        z3expr = Z.T(z3expr)
    preconditions = argument_preconditions(args)
    if preconditions:
        if len(preconditions) == 1:
            z3expr = Z.Implies(Z.T(to_z3(preconditions[0], env, scopes)), z3expr)
        else:
            z3expr = Z.Implies(Z.And([Z.T(to_z3(p, env, scopes)) for p in preconditions]), z3expr)
    z3arg_constants = [env.refs[a] for a in args if a in env.refs]
    if z3arg_constants:
        z3expr = z3.ForAll(z3arg_constants, z3expr)
    return z3expr

class Z3Statement:
    def __init__(self, first, env=None, scopes=None, goal=None):
        if env is None:
            self.src_ast = None
            self.semantic_hash = None
            self.value = None
            self.expr = first
        else:
            self.src_ast = first
            self.semantic_hash = semantic_hash(first)
            self.value = utility_score(self.semantic_hash, semantic_hash(goal))
            self.expr = assertion_fn_to_z3(first, env, scopes)
    def __str__(self):
        return 'Z3Statement(score={}, src={})'.format(
            self.value, unparse(self.src_ast) if self.src_ast else self.expr.sexpr())

def check_assertion_fn(conclusion_fn):
    ret, core, extraneous = prove_assertion_fn(conclusion_fn)
    print ('C A F', repr(extraneous))
    return ret

def prove_assertion_fn(conclusion_fn):
    env = Z3BindingEnv()

    print('Checking assertion:')
    print(' ', unparse(conclusion_fn))

    conclusion = assertion_fn_to_z3(conclusion_fn, env, [])

    print('Using support:')
    baseline = list(map(Z3Statement, env.support[:] + core_assertions(env)))
    # always-include assertions
    for a in _pure_defs.get_fn('').get_assertions():
        print(' ', unparse(a))
        baseline.append(Z3Statement(a, env, [], conclusion_fn))

    _MAX_DEPTH = 10 # TODO experiment with this
    handled = set()
    for iter in range(_MAX_DEPTH):
        borderset = set(env.refs.keys()) - handled
        addedone = False
        for name, fninfo in _pure_defs.functions.items():
            fn_def = fninfo.get_definition()
            if fn_def in borderset:
                addedone = True
                handled.add(fn_def)
                for assertion in fninfo.get_assertions():
                    print('.A. ', unparse(assertion))
                    scopes = get_scope_for_def(assertion)
                    baseline.append(Z3Statement(assertion, env, scopes, conclusion_fn))
                    baseline.append(Z3Statement(Unk.is_func(env.refs[fn_def])))
                if fninfo.definitional_assertion:
                    print('.D. ', unparse(fninfo.definitional_assertion))
                    scopes = get_scope_for_def(fn_def)
                    baseline.append(Z3Statement(fninfo.definitional_assertion, env, scopes, conclusion_fn))
        if not addedone:
            print('Completed knowledge expansion after {} iterations: {} functions.'.format(iter, len(handled)))
            break

    print ()
    print ('conclusion:',conclusion_fn)
    ret, core = solve(baseline, conclusion)
    extraneous = set(baseline) - set(core)
    return ret, core, extraneous

def core_assertions(env):
    refs = env.refs
    isint, isbool, _builtin_len, _builtin_tuple, _op_Add = [
        _pure_defs.get_fn(name).get_definition()
        for name in (
            'isint', 'isbool', '_builtin_len', '_builtin_tuple', '_op_Add')
    ]
    n = z3.Const('n', z3.IntSort())
    i = z3.Const('i', z3.IntSort())
    b = z3.Const('b', z3.BoolSort())
    r = z3.Const('r', Unk)
    g = z3.Const('g', Unk)
    x = z3.Const('x', Unk)
    baseline = [
        # z3.ForAll([n], Z.Int(Z.Wrapint(n)) == n),
        # z3.ForAll([n,i], (n == i) == (Z.Wrapint(n) == Z.Wrapint(i))), # implied by previous
        # z3.ForAll([x], Z.Not(Z.And(Z.F(x), Z.T(x)))), # TODO required? also expressable in pure.py
    ]

    # f(r, *(), ...) = f(r, ...)
    baseline.append(z3.ForAll([r],
        Z.Eq(Z.Concat(r, Unk._), r)
    ))

    # f(g, *(x,), ...) = f(g, x, ...)
    # TODO think this is derivable fromt he other two
    baseline.append(z3.ForAll([x, g],
        Z.Eq(
            Z.Concat(g, Unk.a(Unk._, x)),
            Unk.a(g, x)
        )
    ))

    # f(a, *(*r, x), ...) = f(a, *r, x, ...)
    baseline.append(z3.ForAll([x, g, r],
        Z.Eq(
            Z.Concat(g, Unk.a(r, x)),
            Unk.a(Z.Concat(g, r), x)
        )
    ))

    # if isint in refs:
    #     baseline.append(z3.ForAll([n], Z.T(App(refs[isint], Unk.a(Unk._, Z.Wrapint(n))))))
    # if isbool in refs:
    #     baseline.append(z3.ForAll([b], Z.T(App(refs[isbool], Unk.a(Unk._, Z.Wrapbool(b))))))
    # if _builtin_len in refs and _builtin_tuple in refs:
    #     baseline.append(App(refs[_builtin_len],
    #         Arg(ArgStart, App(refs[_builtin_tuple],
    #             ArgStart
    #         ))
    #     ) == Z.Wrapint(0))
    #     baseline.append(z3.ForAll([a, x],
    #         App(refs[_builtin_len],
    #             Arg(ArgStart, App(refs[_builtin_tuple], Arg(a, x)))
    #         ) ==
    #         App(refs[_op_Add], Arg(Arg(ArgStart,
    #             App(refs[_builtin_len], Arg(ArgStart, App(refs[_builtin_tuple], a)))),
    #             Z.Wrapint(1))
    #         )
    #     ))
    return baseline


class FunctionChecker(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        print('function def',
            node.name,
            [(a.arg,a.annotation) for a in node.args.args],
            node.returns,
            node.body)
        check_assertion_fn(node)
        return node

def pcompile(*functions):
    fn_to_module = {f:inspect.getmodule(f) for f in functions}
    for module in set(fn_to_module.values()):
        tree = ast.parse(inspect.getsource(module))
        x = FunctionChecker().visit(tree) # .body[0].body[0].value
    return functions if len(functions) > 1 else functions[0]

def utility_score(axiom, proof):
    # lower scores are better
    unrelated_axiom_bias = numpy.maximum(axiom - proof, 0.0)
    matching_bias = -(axiom * proof)
    print('utility ', sum(unrelated_axiom_bias), sum(matching_bias))
    return sum(unrelated_axiom_bias) + sum(matching_bias) * 3

def order_axioms(axioms, proof_obligation):
    proof_semantic = semantic_hash(proof_obligation)
    scores_and_axioms = sorted(
        ((utility_score(semantic_hash(axiom), proof_semantic), axiom)
        for axiom in axioms),
        key= lambda p:p[0]
    )
    return [a for (s, a) in scores_and_axioms]



if __name__ == "__main__":
    import doctest
    doctest.testmod()
