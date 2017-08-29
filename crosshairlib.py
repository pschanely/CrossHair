import ast
import builtins
import collections
import copy
import doctest
import functools
import inspect
import operator
import z3

import crosshair

try:
    import astunparse
    def unparse(e):
        return astunparse.unparse(e).strip()
except:
    unparse = ast.dump

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
(3)
'''

class ParamReplacer(ast.NodeTransformer):
    def __init__(self, param_mapping):
        self.mapping = param_mapping
    def visit_Name(self, node):
        return self.mapping.get(node.id, node) or node

def replace_params_with_objects(target_node, args, call_object):
    default_offset = len(args.args) - len(args.defaults)

    arg_mapping = {}
    for idx, arg in enumerate(a.arg for a in args.args):
        #print(('arg',dir(arg)))
        arg_mapping[arg] = None
        if idx >= default_offset:
            arg_mapping[arg] = args.defaults[idx - default_offset]

        if len(call_object.args) > idx:
            arg_mapping[arg] = call_object.args[idx]

    for keyword in call_object.keywords:
        arg_mapping[keyword.arg] = keyword.value

    return ParamReplacer(arg_mapping).visit(target_node)

def astparse(codestring):
    return ast.parse(codestring).body[0]
def exprparse(codestring):
    return ast.parse(codestring).body[0].value
def astcopy(node, **overrides):
    args = {k:getattr(node, k, None) for k in node._fields}
    args.update(overrides)
    return type(node)(**args)

def inline(call_node, func_to_inline):
    '''
    >>> foo = astparse('def foo(x): return x + 2')
    >>> expr = astparse('foo(42)').value
    >>> inlined = inline(expr, foo)
    >>> unparse(inlined)
    '(42 + 2)'
    >>> eval(compile(ast.Expression(inlined), 'f.py', 'eval'))
    44
    >>> unparse(inline(astparse('foo(*(42,))').value, foo))
    '(42 + 2)'
    '''
    if type(func_to_inline) is ast.Lambda:
        body = func_to_inline.body
    else:
        body = func_to_inline.body[0]
    if isinstance(body, ast.Return):
        body = body.value
    return replace_params_with_objects(body, func_to_inline.args, call_node)

class ScopeTracker(ast.NodeTransformer):
    def __init__(self):
        self.scopes = []
    def resolve(self, node):
        '''
        Finds source of value, whether defined elsewhere or just returns
        the original node.
        Different references that can be determined to be equal will be
        reference-equivalent.
        '''
        nodetype = type(node)
        if nodetype is ast.Name:
            refname = node.id
            # print('ScopeTracker', refname, 'scopes', self.scopes)
            if refname[0] != '_':
                if hasattr(builtins, refname):
                    return _pure_defs.get_fn('_builtin_' + refname).get_definition()
                if hasattr(crosshair, refname):
                    return _pure_defs.get_fn(refname).get_definition()
            elif refname.startswith('_z_'):
                zname = refname[3].upper() + refname[4:]
                return getattr(Z, zname)
            for bindings in reversed(self.scopes):
                if refname in bindings:
                    return self.resolve(bindings[refname])
        return node
    def preprocess_assign(self, node):
        nodetype = type(node)
        ret = {}
        if nodetype is ast.AnnAssign:
            if node.simple and type(node.target) is ast.Name:
                return {node.target.id: node.value}
        elif nodetype is ast.Assign:
            for target in node.targets:
                if type(target) is ast.Tuple:
                    raise Exception('Handle unpacking assignments')
                ret[target.id] = node.value
        elif nodetype is ast.FunctionDef:
            ret[node.name] = node
        else:
            raise Exception()
        return ret
    def statements(self, statements, idx):
        # print(repr(statements))
        processed = []
        while True:
            if idx >= len(statements):
                return processed
            statement = statements[idx]
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                pass # TODO handle imports
            is_assignment = isinstance(statement, (ast.Assign, ast.AnnAssign, ast.FunctionDef))
            predeclare = isinstance(statement, ast.FunctionDef)
            if is_assignment and predeclare: # must pre-declare in this case (for recursive functions)
                self.scopes.append(self.preprocess_assign(statement))
            statement = self.visit(statement)
            processed.append(statement)
            if is_assignment and not predeclare:
                self.scopes.append(self.preprocess_assign(statement))
            if is_assignment:
                remainder = self.statements(statements, idx + 1)
                self.scopes.pop()
                return processed + remainder
            else:
                idx += 1
    def visit_Module(self, node):
        return ast.Module(body=self.statements(node.body, 0))
    def visit_FunctionDef(self, node):
        self.scopes.append({a.arg: a for a in node.args.args})
        node = astcopy(node, body=self.statements(node.body, 0))
        self.scopes.pop()
        return node
    def visit_Lambda(self, node):
        self.scopes.append({a.arg: a for a in node.args.args})
        node = astcopy(node, body=self.visit(node.body))
        self.scopes.pop()
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

class AdvancedRewriter(ScopeTracker):
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



# TODO unclear whether something like this might be a better way:
# (I think not, because it confuses different levels of semantics?)
# Et = z3.Const('[]', Unk)
# Singleton = z3.Function('_', Unk, Unk)
# # Ed = z3.Const('{}', Unk) # TODO dicts, someday
# # Binding = z3.Function(':', Unk, Unk, Unk)

PyFunc = z3.DeclareSort('PyFunc')
# Tuple = z3.Datatype('Tuple')
Unk = z3.Datatype('Unk')
Unk.declare('none')
Unk.declare('bool', ('tobool', z3.BoolSort()))
Unk.declare('int', ('toint', z3.IntSort()))
Unk.declare('func', ('tofunc', PyFunc))
# Unk.declare('tuple', ('totuple', Tuple))
# Tuple.declare('t', ('tl', Tuple), ('hd', Unk))
# Tuple.declare('e')
# Tuple, Unk = z3.CreateDatatypes(Tuple, Unk)
# Star = z3.Function('s', Tuple, Unk, Tuple)
# App = z3.Function('.', Unk, Tuple, Unk)

Unk.declare('a', ('tl', Unk), ('hd', Unk))
Unk.declare('_') # empty tuple
(Unk,) = z3.CreateDatatypes(Unk)
#Star = z3.Function('s', Tuple, Unk, Tuple)
App = z3.Function('.', Unk, Unk, Unk)


# PyFunc = z3.DeclareSort('PyFunc')
# Unk = z3.DeclareSort('Unk')
#
# Args = z3.DeclareSort('Args')
# App = z3.Function('.', Unk, Args, Unk)
# Arg = z3.Function('a', Args, Unk, Args)
# StarArg = z3.Function('s', Args, Unk, Args)
# ArgStart = z3.Const('_', Args)



class ZHolder(): pass
Z = ZHolder()
Z.Wrapbool = Unk.bool # z3.Function('Wrapbool', z3.BoolSort(), Unk)
Z.Wrapint = Unk.int # z3.Function('Wrapint', z3.IntSort(), Unk)
Z.Wrapfunc = Unk.func # z3.Function('Wrapfunc', PyFunc, Unk)
Z.Bool = Unk.tobool # z3.Function('Bool', Unk, z3.BoolSort())
Z.Int = Unk.toint # z3.Function('Int', Unk, z3.IntSort())
Z.Func = Unk.tofunc # z3.Function('Func', Unk, PyFunc)
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
# Z.Forall = lambda f: z3.ForAll([f.args.args[0].arg], to_z3(f.body,...))

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
        class FnFinder(ScopeTracker):
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
    print('expectation')
    print(unparse(fn))
    print(unparse(fdef))
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

class Z3Transformer(ScopeTracker): #ast.NodeTransformer):
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
            # print('register abort : ', str(definition))
            return definition
            # if hasattr(definition, 'name'):
            #     name = definition.name()
            # else:
            #     name = definition.__name__
        if definition not in refs:
            # print('register done  : ', str(name))
            # print('register new Unk value', name, definition)
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
            if subscript.step == 1:
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

    def visit_Lambda(self, node):
        # num_args = len(node.args.args)
        # print('lambda numargs:', num_args, unparse(node))
        name = 'lambda_{}_{}'.format(node.lineno, node.col_offset)
        funcval = Z.Wrapfunc(z3.Const(name, PyFunc))

        args = gensym()
        QuantifiedVar = collections.namedtuple('QuantifiedVar', ['name'])
        self.scopes.append({args:QuantifiedVar(name=args)})
        pretend_call = astcall(None, ast.Starred(value=ast.Name(id=args)))
        inlined = inline(pretend_call, node) # TODO does not work yet: inlining needs to reconcile parameters and return error in appriate cases
        funcexpr = self.visit(inlined)
        self.scopes.pop()
        call_by_funcval = App(funcval, self.env.refs[args])
        stmt = z3.ForAll([z3.Const(args, Unk)], call_by_funcval == funcexpr)
        self.env.support.append(stmt)

        # # z3func = z3.Function(name, Unk, *[Unk for _ in range(num_args)])
        # argnames = [a.arg for a in node.args.args]
        # argvars = [z3.Const(name, Unk) for name in argnames]
        # # TODO binding env vals is a flat dictionary? that seems off
        # self.env.vals.update(dict(zip(argnames, argvars)))
        # z3body = to_z3(node.body, self.env)
        # z3application = z3apply(z3val, argvars)
        # # print(z3body.sexpr(), z3application.sexpr())
        # stmt = z3.ForAll(argvars, z3body == z3application)
        # self.env.support.append(stmt)
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
        # funcs = self.env.funcs
        # if name not in funcs:
        #     print('inferring uninterpreted fn', name, '#', len(node.args))
        #     funcs[name] = z3.Function(name, Unk, *(Unk for a in node.args))
        # z3fn = funcs[name]
        # if z3fn is ToInt: # TODO: necessary?
        #     cast_arg = node.args[0]
        #     if type(cast_arg) is ast.Num:
        #         return z3.Int(cast_arg.n)
        params = [self.visit(a) for a in node.args]
        # print(z3fn, params)
        # print(z3fn, [type(p) for p in params])
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

    # TODO: not wokring yet
    # >>> to_z3(exprparse('lambda x:x'), Z3BindingEnv())
    # func(lambda_1_0)
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

def fn_expr(fn):
    fntype = type(fn)
    if fntype == ast.FunctionDef:
        stmts = fn.body
        # filter out comments (or other "value" statements)
        stmts = [s for s in stmts if type(s) != ast.Expr]
        if len(stmts) > 1:
            raise Exception('More than one statement in function body:'+repr(stmts))
        if len(stmts) == 0:
            raise Exception('No statements in function body:'+repr(stmts))
        if type(stmts[0]) != ast.Return:
            raise Exception(type(stmts[0]))
        return stmts[0].value
    elif fntype == ast.Lambda:
        return fn.body
    else:
        raise Exception()

def solve(assumptions, conclusion, make_repro_case=False):
    solver = z3.Solver()
    z3.set_param(
        verbose = 10,
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

    if make_repro_case:
        for assumption in assumptions:
            solver.add(assumption)
        solver.add(z3.Not(conclusion))
    else:
        for assumption in assumptions:
            solver.assert_and_track(assumption, 'assumption: '+str(assumption))
        solver.assert_and_track(z3.Not(conclusion), 'goal: '+str(conclusion))

    try:
        ret = solver.check()
        if ret == z3.unsat:
            if not make_repro_case:
                print ('BEGIN PROOF CORE ')
                for stmt in solver.unsat_core():
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
    # if ret != True and not make_repro_case:
    #     ret2 = solve(assumptions, conclusion, make_repro_case=True)
    #     if ret != ret2:
    #         print('Differening results with repro and non-repro run!')
    #     else:
    #         print('FAILED; re-ran with repro case.')
    return ret

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

def check_assertion_fn(fn):
    env = Z3BindingEnv()

    print('Checking assertion:')
    print(' ', unparse(fn))
    conclusion = assertion_fn_to_z3(fn, env, [])

    print('Using support:')
    baseline = []
    # always-include assertions
    for a in _pure_defs.get_fn('').get_assertions():
        print(' ', unparse(a))
        expr = assertion_fn_to_z3(a, env, [])
        baseline.append(expr)
    baseline.extend(core_assertions(env))

    _MAX_DEPTH = 2 # TODO experiment with this
    handled = set()
    for _ in range(_MAX_DEPTH):
        borderset = set(env.refs.keys()) - handled
        for name, fninfo in _pure_defs.functions.items():
            fn_def = fninfo.get_definition()
            if fn_def in borderset:
                handled.add(fn_def)
                for assertion in fninfo.get_assertions():
                    # print('.A. ', unparse(assertion))
                    scopes = get_scope_for_def(assertion)
                    baseline.append(assertion_fn_to_z3(assertion, env, scopes))
                if fninfo.definitional_assertion:
                    # print('.D. ', unparse(fninfo.definitional_assertion))
                    scopes = get_scope_for_def(fn_def)
                    baseline.append(assertion_fn_to_z3(fninfo.definitional_assertion, env, scopes))

    print ()
    print ('conclusion:',conclusion)
    for l in baseline:
        print('baseline:', l)
    return solve(baseline, conclusion)

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
        z3.ForAll([n], Z.Int(Z.Wrapint(n)) == n),
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

    if isint in refs:
        baseline.append(z3.ForAll([n], Z.T(App(refs[isint], Unk.a(Unk._, Z.Wrapint(n))))))
    if isbool in refs:
        baseline.append(z3.ForAll([b], Z.T(App(refs[isbool], Unk.a(Unk._, Z.Wrapbool(b))))))
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
        z3check(node)

        return node

def pcompile(*functions):
    fn_to_module = {f:inspect.getmodule(f) for f in functions}
    #modules = set(inspect.getmodule(f) for f in functions)
    for module in set(fn_to_module.values()):
        tree = ast.parse(inspect.getsource(module))
        x = FunctionChecker().visit(tree) # .body[0].body[0].value
    return functions if len(functions) > 1 else functions[0]


'''
expr = ast.parse('def append(x,y): c(c(x, islist)+c(y, islist), islist)')
#print(unparse(expr))
x = AstVisitor().visit(expr).body[0].body[0].value
print(x, dir(x))


solver = z3.Solver()
solver.set("timeout", 600)

i = z3.Const('i', Unk)
j = z3.Const('j', Unk)
x = z3.Const('x', Unk)
y = z3.Const('y', Unk)


solver.add([
    # list append types
    z3.ForAll([i,j], z3.Implies(z3.And(IsList(i), IsList(j)), IsList(Add(i, j)))),
    # list append lengths
    z3.ForAll([i,j], z3.Implies(z3.And(IsList(i), IsList(j)), ToInt(Len(Add(i,j))) == ToInt(Len(i)) + ToInt(Len(j)))),
    # filter length is less than (or equal to) input length
    z3.ForAll([i], z3.Implies(z3.And(IsList(i), IsFunc(j)), ToInt(Len(Filter(j, i))) <= ToInt(Len(j)))),
    ])
'''



'''
Varying objectives:
(1) Correctness validation; typing
(2) Inlining/partial evaluation/Super optimization
(3) Compilation to LLVM / others
(4) integer range optimization
(5) datastructure selection
(6) JSON parse/serialize API optimization
(7) Memory management optimization
'''

'''
def r():
    eq(len([]), 0)
def r(X):
    implies(isint(X),
            eq(X + 0, X))
def r(X, Y):
    eq(X if True else Y, X)

# extrinsic invariant:

@pre(isnat)
@pre(isnat)
def add_is_greater(X, Y):
    X + Y > X

# rewrite rule:

@rewrite()
@pre(isint)
def add_zero(X):
    X + 0 == X

# regular function:

@pre(isint)
@post(isint, lambda o, i: abs(o) > abs(i))
def double(X):
    return X + c(X, isint)

#def top(things):
#    return functools.reduce(max, things)

@pre(lambda l: islist(l) or isstring(l) or isdict(l))
@post(isint, lambda r: r >= 0)
def len(x):
    pass

@pre(lambda x: isint(x) or islist(x) or isstring(x))
@pre(lambda y: isint(y) or islist(y) or isstring(y))
@post(lambda r, x, y: ((isint(r) and isint(x) and isint(y)) or
                       (islist(r) and islist(x) and islist(x)) or
                       (isstring(r) and isstring(x) and isstring(x))))
def operator_plus(x,y):
    pass

@pre(isfunc)
@pre()
@post()
def apply(f, x): # don't reason about "apply"; expand the dispatch table instead each time
    pass

@pre(isfunc)
@pre(islist)
@post(islist, lambda r, f, l: len(r) <= len(l) and all(apply(f,i) for i in r))
def filter(f, l):
    pass

@pre(islist, lambda l: all(map(isint, l)))
@post(isint, lambda r, l: 0 <= r <= len(l))
def count_sevens(l):
    return len(i for i in l if i == 7)
'''


'''
print(solver.check())

solver.add([
    IsList(x),
    IsList(y),
    ToInt(Len(y)) == 0,
])
print(solver)
theorem = ToInt(Len(Add(x, y))) == ToInt(Len(x))
theorem = ToInt(Len(y)) == ToInt(Len(y)) + ToInt(Len(x))
print('Theorem: ', theorem)
#IsList(Add(x,y))
solver.add(z3.Not(theorem))

try:
    ret = solver.check()
    if ret == z3.unsat:
        print('Proven.')
    elif ret == z3.sat:
        print('False.')
    else:
        print('Unsure.')
except z3.z3types.Z3Exception as e:
    if e.value == b'canceled':
        print('Timeout.')
    else:
        raise e
#print(ret)
'''


'''
    def visit_ImportFrom(self, node):
        node = self.generic_visit(node)
        print('from', node.module, 'import',
            [(n.name, n.asname) for n in node.names])
        return node

    def visit_Import(self, node):
        node = self.generic_visit(node)
        print('import', [(n.name, n.asname) for n in node.names])
        return node
'''
'''
ToBool(islist(filter(WrapFunc(lambda_9_18), range(x))))
[
ToBool(isnat(x)),
ForAll([f, l],
       Implies(And(ToBool(isfunc(f)), ToBool(islist(l))),
               ToBool(islist(filter(f, l))))),
ForAll(x, Implies(ToBool(isnat(x)), ToBool(islist(range(x))))),
ForAll(x, IsBool(isfunc(x))),
ForAll(x, IsBool(islist(x))),
ForAll(x, IsBool(isnat(x)))
]
'''

'''
def to_logic(node, env):
    typ = type(node)
    if typ is ast.BinOp or typ is ast.BoolOp:
        z3fn = _operator_z3_map[node.op.__class__]
        left, right = to_logic(node.left, env), to_logic(node.right, env)
        try:
            return z3fn(left, right)
        except z3.z3types.Z3Exception:
            print('bin node: ', node.op, z3fn)
            print('left: ', left, left.sort())
            print('right: ', right, right.sort())
            raise
    if typ is ast.Name:
        if node.id in env:
            return env[node.id]['z3']
        return _identifier_z3_map[node.id]
    if typ is ast.Call:
        z3fn = _identifier_z3_map[node.func.id]
        return z3fn(*(to_logic(a, env) for a in node.args))
    if typ is ast.Num:
        if type(node.n) != int:
            raise Exception('unsupported number: '+str(node.n))
        return int_const(node.n)#z3.IntVal(node.n)
    if typ is ast.Lambda:
        raise Exception()
    raise Exception('unsupported ast node: '+str(typ))
'''


'''
def lambdamatcher(n, p, b):
    if not matches(n.body, p.body, b):
        return False
    if len(p.args.args) != 1:
        raise Exception()
    varname = p.args.args[0].arg
    b[varname] = n.args
    return True
_MATCHER[ast.Lambda] = lambdamatcher
def should_inline(binds):
    foo = astparse('def foo(x): return x + 2')
    expr = astparse('foo(42)').value
    inlined = inline(expr, foo)
    binds['B'] = inline_lambda(binds['P'], binds['B'], binds['X'])
    return True
('(lambda P: B)(X)', 'B', should_inline)
rewritten = logical_rewriter.rewrite(exprparse('(lambda x:x+1)(6)'))
print(('rewrite engine', unparse(rewritten)))
sys.exit(0)
'''


'''
def test_solver():
    solver = z3.Solver()
    appl = z3.Function('apply', Unk, Unk, Unk)
    filtr = z3.Function('filtr', Unk, Unk, Unk)
    contains = z3.Function('contains', Unk, Unk, z3.BoolSort())
    checkerfn = z3.Const('checkerfn', Unk)
    inputlist = z3.Const('inputlist', Unk)
    X1 = z3.Const('X1', Unk)
    L1 = z3.Const('L1', Unk)
    X2 = z3.Const('X2', Unk)
    L2 = z3.Const('L2', Unk)
    X3 = z3.Const('X3', Unk)
    L3 = z3.Const('L3', Unk)
    c = z3.Const('c', Unk)
    e = z3.Const('e', Unk)
    # filtr(F, L) = if L == e: return e elif F(L[0]): [L[0]] + filtr(F, L[1:]) else: filtr(F, L[1:])
    assumptions = [
        z3.ForAll([X1], IsBool(appl(checkerfn, X1))),
        z3.ForAll([L1], z3.Or((L1 == e), z3.Exists([X2], contains(X2, L1)))),
        IsList(e),
        z3.ForAll([X3], e == filtr(X3, e)),
        IsList(c),
        z3.ForAll([X3,L2], z3.Implies(
            #IsBool(appl(checkerfn, X)),
            contains(X3, L2),
            z3.Implies(
                IsBool(appl(checkerfn, X3)),
                IsList(filtr(checkerfn, L2))
            )
        ))
    ]
    conclusion = IsList(filtr(checkerfn, c))
    #ret = z3.prove(z3.Implies(z3.And(*assumptions), conclusion))
    ret = solve(assumptions, conclusion)
    print('test solver', ret)

#test_solver()
#sys.exit(0)
'''


'''
# Is proving via recursion any harder?
_REDUCE_HEURISTIC_PATT = preprocess_pattern_vars(exprparse('reduce(F, L, I)'))

class ReduceHeuristic(ast.NodeTransformer):
    def __init__(self):
        self.bindings = {}
    def generic_visit(self, node):
        if self.bindings:
            return node
        nam = getattr(getattr(node, 'func', {}), 'id', None)
        if matches(node, _REDUCE_HEURISTIC_PATT, self.bindings):
            return ast.Name(id=AstVar('R'))
        else:
            self.bindings.clear()
            return super().generic_visit(node)
    def create_inference(self, assertion_node):
        print('create_inference', unparse(assertion_node))
        print('create_inference', ast.dump(assertion_node))
        predicate = self.visit(assertion_node)
        if not self.bindings:
            return []
        holds_for = lambda x : replace_pattern_vars(predicate, {'R':x})
        base_case = holds_for(self.bindings['I'])
        #N1 = z3.Const('N1', Unk)
        #N2 = z3.Const('N2', Unk)
        #list_contents = ast.Call(func=Name(id='all'), args=
        #induction = z3.Implies(z3.And(holds_for(N1), holds_for(N2)),
        #    holds_for(ast.Call(func=self.bindings['F'], args=[N1, N2], kwargs=[])))
        #final = z3.Implies(z3.And(base_case, list_contents, induction), fn_node)
        #return final
        N = gensym('N')
        LI = gensym('LI')
        env = {}
        holds_for_n = to_z3(holds_for(ast.Name(id=N)), env)
        print(ast.dump(ast_in(ast.Name(id=LI), self.bindings['L'])))
        n_in_list = to_z3(ast_in(ast.Name(id=LI), self.bindings['L']), env)
        holds_for_call = to_z3(holds_for(ast.Call(
            func=self.bindings['F'],
            args=[ast.Name(id=N), ast.Name(id=LI)],
            kwargs=[])), env)
        # true for a prefix of L, then true for F(reduce(L[:i]), L[i])
        induction = z3.Implies(
            z3.And(holds_for_n, n_in_list),
            holds_for_call)
        final = z3.Implies(z3.And(base_case, induction), fn_node)
        return [final]

'''


'''
        ToBool(all(map(isnat, x)))
Implies(ToBool(all(map(F,     L))), ToBool(all(map(F,     filter(G,                     L)))))
                                    ToBool(all(map(isnat, filter(WrapFunc(lambda_9_18), x))))
'''

# _builtin_stubs = {
#     any: astparse('def any(l:islist) -> isbool: pass'),
#     all: astparse('def all(l:islist) -> isbool: pass'),
#     len: astparse('def len(l:islist) -> isnat: pass'),
#     map: astparse('def map(f:isfunc, l:islist) -> islist: pass'),
#     range: astparse('def range(x:isint) -> (lambda l: all(map(isnat, l))): pass'),
#     filter: astparse('def filter(f:isfunc, l:islist) -> lambda r,f,l: all(map((lambda i: i in l), r)): pass'),
# }


# class LambdaLifter(ScopeTracker):
#     # When lambda escapes, mark for inlining
#     def __init__(self):
#         super().__init__()
#         self.topleveldefs = {}
#     def visit_FunctionDef(self, node):
#         return self.handle(node.name, node)
#     def visit_Lambda(self, node):
#         return self.handle('lambda', node)
#     def handle(self, name, node):
#         if not self.scopes:
#             return node # already at top level
#         args, body = node.args, node.body
#         # do stuff here!
#         super().visit_FunctionDef(node)
#         return self.visit_FunctionDef(node)
#         #args, body = node.args, node.body
#
# def lift_lambda(node):
#     LambdaLifter().visit(node)



    # A = z3.Const('A', Unk)
    # B = z3.Const('B', Unk)
    # F = z3.Const('F', Unk)
    # G = z3.Const('G', Unk)
    # L = z3.Const('L', Unk)
    # X = z3.Const('X', Unk)
    # N = z3.Const('N', z3.IntSort())
    # funcs = env.funcs
    # vals = env.vals
    # ops = env.ops
    # baseline = [
    #     #z3.ForAll([X], IsTruthy(X) == (X == WrapBool(True))),
    #     z3.ForAll([X], IsTruthy(_z3_fn_map[ast.Not](X)) == IsFalsey(X)),
    #     z3.ForAll([X], IsTruthy(X) == z3.Not(IsFalsey(X))),
    #     z3.ForAll([A,B], IsTruthy(_z3_fn_map[ast.And](A,B)) == z3.And(IsTruthy(A), IsTruthy(B))),
    #     z3.ForAll([A,B], IsFalsey(_z3_fn_map[ast.And](A,B)) == z3.And(IsFalsey(A), IsFalsey(B))),
    #     z3.ForAll([A,B], IsTruthy(_z3_fn_map[ast.Or](A,B)) == z3.Or(IsTruthy(A), z3.And(IsFalsey(A), IsTruthy(B)))),
    # ]
    # if 'isbool' in funcs:
    #     baseline += [
    #         IsTruthy(funcs['isbool'](WrapBool(True))),
    #         IsTruthy(funcs['isbool'](WrapBool(False))),
    #     ]
    # for common in set(funcs.keys()) & set(vals.keys()):
    #     print('common', common)
    #     func = funcs[common]
    #     argvars = [z3.Const('A{}'.format(i), Unk) for i in range(func.arity())]
    #     as_applied = z3apply(env, vals[common], argvars)
    #     baseline += [ z3.ForAll(argvars, func(*argvars) == as_applied) ]
    # if 'all' in funcs and 'map' in funcs:
    #     In = _z3_fn_map[ast.In]
    #     x_in_l = IsTruthy(In(X, L))
    #     all_l_has_f = IsTruthy(funcs['all'](funcs['map'](F, L)))
    #     all_l_has_g = IsTruthy(funcs['all'](funcs['map'](G, L)))
    #     x_has_f = IsTruthy(z3apply(env, F, [X]))
    #     f_implies_g = z3.ForAll([A], z3.Implies(
    #         IsTruthy(z3apply(env, F, [A])),
    #         IsTruthy(z3apply(env, G, [A]))))
    #     baseline += [
    #         z3.ForAll([L,F,X], z3.Implies(
    #             z3.And(x_in_l, all_l_has_f),
    #             x_has_f)),
    #         z3.ForAll([L,F,G,X], z3.Implies(
    #             z3.And(all_l_has_f, f_implies_g),
    #             all_l_has_g
    #         ))# TODO add? all(map(F, L)) and (F(I) -> G(I)) -> all(map(G,I))
    #     ]
    # if 'all' in funcs and 'reduce' in funcs and 'map' in funcs:
    #     z3apply(env,F,[A,B]) #hack to create apply#2
    #     app1, app2 = funcs['apply#1'], funcs['apply#2']
    #     f_holds = lambda x:IsTruthy(app1(F, x))
    #     l_has_f = IsTruthy(funcs['all'](funcs['map'](F, L)))
    #     x_has_f = IsTruthy(app1(F, X))
    #     g_preserves_f = z3.ForAll([A,B], z3.Implies(
    #         z3.And(f_holds(A), f_holds(B)), f_holds(app2(G, A, B))))
    #     baseline += [
    #         z3.ForAll([F,L,X,G],
    #             z3.Implies(z3.And(l_has_f, x_has_f, g_preserves_f),
    #                 f_holds(funcs['reduce'](G, L, X))))
    #     ]
    # if 'all' in funcs and 'filter' in funcs and 'map' in funcs:
    #     all_has_f = lambda l: IsTruthy(funcs['all'](funcs['map'](F, l)))
    #     baseline += [
    #         z3.ForAll([F,L,G],
    #             z3.Implies(all_has_f(L), all_has_f(funcs['filter'](G, L))))
    #     ]
    # if 'isfunc' in funcs:
    #     baseline += [
    #         # z3.ForAll([F], ToBool(funcs['isfunc'](F)))
    #     ]
    # if 'isbool' in funcs:
    #     baseline += [
    #         # z3.ForAll([X], IsTruthy(funcs['isbool'](X)))
    #     ]
    # if 'isnat' in funcs or 'isint' in funcs:
    #     # create isint
    #     if 'isint' not in funcs:
    #         funcs['isint'] = z3.Function('isint', Unk, Unk)
    #     if 'isnat' not in funcs:
    #         funcs['isnat'] = z3.Function('isnat', Unk, Unk)
    #     baseline += [
    #         z3.ForAll([X], z3.Implies(IsTruthy(funcs['isnat'](X)), IsTruthy(funcs['isint'](X)))),
    #         z3.ForAll([X], z3.Implies(IsTruthy(funcs['isnat'](X)), ToInt(X) >= 0)),
    #         z3.ForAll([X], z3.Implies(
    #             z3.And(IsTruthy(funcs['isint'](X)), ToInt(X) >= 0), IsTruthy(funcs['isnat'](X)))),
    #         z3.ForAll([N], z3.Implies(N >= 0, IsTruthy(funcs['isnat'](WrapInt(N))))),
    #         # TODO how to express these as invariants over the minus operator function?:
    #         z3.ForAll([A,B], z3.Implies(
    #             z3.And(IsTruthy(funcs['isint'](A)), IsTruthy(funcs['isint'](B))),
    #             IsTruthy(funcs['isint'](_z3_fn_map[ast.Sub](A,B))))),
    #         z3.ForAll([A,B], z3.Implies(
    #             z3.And(IsTruthy(funcs['isint'](A)), IsTruthy(funcs['isint'](B))),
    #             ToInt(_z3_fn_map[ast.Sub](A,B)) == ToInt(A) - ToInt(B))),
    #         z3.ForAll([N], IsTruthy(funcs['isint'](WrapInt(N)))),
    #         z3.ForAll([A,B], z3.Implies(
    #             z3.And(IsTruthy(funcs['isint'](A)), IsTruthy(funcs['isint'](B))),
    #             Add(A, B) == WrapInt(ToInt(A) + ToInt(B))))
    #     ]
    # if Get in ops:
    #     all_l_has_f = IsTruthy(funcs['all'](funcs['map'](F, L)))
    #     l_is_list = IsTruthy(funcs['islist'](L))
    #     x_indexes_l = And(IsTruthy(funcs['isint'](X)),
    #                       ToInt(X) >= 0,
    #                       ToInt(X) < ToInt(funcs['len'](L)))
    #     baseline +=  [
    #       z3.ForAll([F,X,L],
    #         z3.Implies(z3.And(all_l_has_f, l_is_list, x_indexes_l),
    #           IsTruthy(z3apply(env, F, [Get(L,X)]))))
    #     ]
    # if ast.Lt in ops or Gt in ops or LtE in ops or GtE in ops:
    #     if_a_and_b_are_ints = lambda e: z3.ForAll([A,B], z3.Implies(z3.And(IsTruthy(funcs['isint'](A)), IsTruthy(funcs['isint'](B))), e))
    #     baseline += [
    #         if_a_and_b_are_ints(IsTruthy(Lt(A, B)) == (ToInt(A) < ToInt(B))),
    #         if_a_and_b_are_ints(IsTruthy(Gt(A, B)) == ToInt(A) > ToInt(B)),
    #         if_a_and_b_are_ints(IsTruthy(LtE(A, B)) == ToInt(A) <= ToInt(B)),
    #         if_a_and_b_are_ints(IsTruthy(GtE(A, B)) == ToInt(A) >= ToInt(B)),
    #     ]



# def z3apply(env, wrapped_fn, args):
#     num_args = len(args)
#     funcs = env.funcs
#     fname = 'apply#{}'.format(num_args)
#     if fname not in funcs:
#         funcs[fname] = z3.Function(fname, Unk, Unk, *[Unk for _ in range(num_args)])
#     print('z3apply', funcs[fname], wrapped_fn, args)
#     return funcs[fname](wrapped_fn, *args)
#
#


# _const_cache = {}
# def int_const(v):
#     if v not in _const_cache:
#         _const_cache[v] = z3.Const('intconst_'+str(v), Unk)
#     return _const_cache[v]


# class RewriteRule(object):
#     def __init__(self, patt, repl):
#         self.patt, self.repl = preprocess_pattern_vars(patt, repl)
#     def tophash(self):
#         return patthash(self.patt)


# _z3_fn_map = {
#     'IsBool': IsBool,
#     'IsTruthy': IsTruthy,
#     'IsInt': IsInt,
#     'ToInt': ToInt,
#     'IsNat': IsNat,
#     'IsFunc': IsFunc,
#     'IsList': IsList,
#     'And': z3.And,
#     # ast.BinOp
#     ast.Add: z3.Function('Add', Unk, Unk, Unk),
#     ast.Sub: z3.Function('Sub', Unk, Unk, Unk),
#     ast.Mod: z3.Function('Mod', Unk, Unk, Unk),
#     # ast.BoolOp
#     ast.And: z3.Function('And', Unk, Unk, Unk),
#     ast.Or:  z3.Function('Or',  Unk, Unk, Unk),
#     # ast.UnaryOp
#     ast.Not: z3.Function('Not', Unk, Unk),
#     # ast.Compare:
#     ast.In:  z3.Function('In',  Unk, Unk, Unk),
#     ast.Eq:  z3.Function('Eq',  Unk, Unk, Unk),
#     ast.LtE: LtE,
#     ast.Lt:  Lt,
#     ast.GtE: GtE,
#     ast.Gt:  Gt,
# }


# def reduce_f_and_i(f, i):
#     def check(bindings):
#         function = bindings['F']
#         initializer = bindings['I']
#         return (
#             type(initializer) is ast.NameConstant and
#             initializer.value in i and
#             type(function) is ast.Name and
#             function.id == f
#             )
#     return check
#
# def f_is_always_true(bindings):
#     f = bindings['F']
#     return (type(f) == ast.Lambda and
#         type(f.body) == ast.NameConstant and
#         f.body.value == True)


# class SupportFinder(ScopeTracker):
#     def __init__(self, fn, env):
#         super().__init__()
#         self.env = env
#         self.support = []
#         self.supported = set()
#         self.nextiteration = [fn]
#     def visit_FunctionDef(self, node):
#         # print('FOUND FUNC ', [(a.arg, a.annotation.id) for a in node.args.args])
#         super().visit_FunctionDef(node)
#
#     # TODO: generate support statements for operators
#     # def visit_BinOp(self, node):
#     #     op = _builtin_stubs[node.op]
#
#     def visit_Name(self, node):
#         resolved = self.resolve(node)
#         if isinstance(resolved, ast.FunctionDef) and resolved not in self.supported:
#             print('support logic; resolved {} to {}'.format(node.id, resolved))
#             self.supported.add(resolved)
#             argnames = [a.arg for a in resolved.args.args]
#             # self.support.append(z3apply(resolved.name, argnames) ==
#             #    argnames)
#             if resolved.returns is None:
#                 return node
#                 # TODO implicitly, preconditions prevent error value return
#                 #    conclusion = (call != err)
#             self.nextiteration += [resolved.args, resolved.returns]
#             env = self.env # this actually matters! do not use empty [Z3BindingEnv()]
#             assumptions = [to_assertion(a.annotation, ast.Name(id=a.arg), env)
#                 for a in resolved.args.args if a.annotation]
#             call = ast.Call(
#                 func=ast.Name(id=resolved.name),
#                 args=[ast.Name(id=a) for a in argnames],
#                 keywords=[],
#             )
#             conclusion = to_assertion(resolved.returns, call, env)
#             if assumptions:
#                 if len(assumptions) == 1:
#                     statement = z3.Implies(assumptions[0], conclusion)
#                 else:
#                     statement = z3.Implies(z3.And(*assumptions), conclusion)
#             else:
#                 statement = conclusion
#             forall = [env.vals[a.arg] for a in resolved.args.args if a.arg in env.vals]
#             if forall:
#                 statement = z3.ForAll(forall, statement)
#             self.support.append(statement)
#             print('SUPPORT', self.support[-1])
#         return node
#
# def find_support(fn, env):
#     finder = SupportFinder(fn, env)
#     while finder.nextiteration:
#         values = finder.nextiteration[:]
#         finder.nextiteration[:] = []
#         for i in values:
#             finder.visit(i)
#     return finder.support
#


# def z3check(fn):
#     print('fn_expr(fn)',unparse(fn_expr(fn)))
#     result = ast.Call(func=fn.returns, args=[fn_expr(fn)], keywords=[])
#     result = basic_simplifier.rewrite(result)
#
#     env = Z3BindingEnv()
#     args = [ast.Name(id=a.arg) for a in fn.args.args]
#     z3_assumptions = [to_assertion(a.annotation, ast.Name(id=a.arg), env) for a in fn.args.args]
#     z3_conclusion = to_z3(ast.Call(func=ast.Name(id='IsTruthy'), args=[result], keywords=[]), env)
#     z3_assumptions += env.support
#     #z3_assumptions += find_support(fn, env)
#     # heuristics
#     #assumptions += ReduceHeuristic().create_inference(conclusion_expr)
#
#     print('==== conclusion ====')
#     print(z3_conclusion)
#     print('==== assumptions ====')
#     for a in z3_assumptions:
#         print(a)
#
#     # t = z3.Then('simplify', 'solve-eqs')#'propagate-values', 'ctx-simplify')
#     # print('simplify')
#     # for a in assumptions:
#     #     ts.add(a)
#     # ts.add(conclusion)
#     # print('cf',ts.check())
#     # print(z3.And(*(assumptions + [conclusion])))
#     # print(t(z3.And(*(assumptions + [conclusion]))))
#     # proven = False
#     proven = solve(z3_assumptions, z3_conclusion)
#     print('z3 proven? (T/F/None): ', proven)
#     if not proven:
#         print('Advanced input: ', unparse(result))
#         result = AdvancedRewriter()(result)
#         print('Advanced result: ', unparse(result))
#         print('Advanced, rewritten result: ', unparse(result))
#     return proven


# ToInt = z3.Function('ToInt', Unk, z3.IntSort())
# WrapInt = z3.Function('WrapInt', z3.IntSort(), Unk)
# IsTruthy = z3.Function('IsTruthy', Unk, z3.BoolSort())
# IsFalsey = z3.Function('IsFalsey', Unk, z3.BoolSort())
# WrapBool = z3.Function('WrapBool', z3.BoolSort(), Unk)
# WrapFunc = z3.Function('WrapFunc', PyFunc, Unk)

# IsInt = z3.Function('IsInt', Unk, z3.BoolSort())
# IsBool = z3.Function('IsBool', Unk, z3.BoolSort())
# IsNat = z3.Function('IsNat', Unk, z3.BoolSort())
# IsList = z3.Function('IsList', Unk, z3.BoolSort())
# IsFunc = z3.Function('IsFunc', Unk, z3.BoolSort())
# Add = z3.Function('Add', Unk, Unk, Unk)
# Get = z3.Function('Get', Unk, Unk, Unk)
# SteppedSubList = z3.Function('SteppedSubList', Unk, Unk, Unk, Unk, Unk)
# SubList = z3.Function('SubList', Unk, Unk, Unk, Unk)
# And = z3.And
# Len = z3.Function('Len', Unk, Unk)
# Filter = z3.Function('Filter', Unk, Unk, Unk)
# Map = z3.Function('Map', Unk, Unk, Unk)
# Reduce = z3.Function('Reduce', Unk, Unk, Unk, Unk)
# Range = z3.Function('range', Unk, Unk)
# LtE = z3.Function('LtE', Unk, Unk, Unk)
# Lt =  z3.Function('Lt', Unk, Unk, Unk)
# GtE = z3.Function('GtE', Unk, Unk, Unk)
# Gt =  z3.Function('Gt', Unk, Unk, Unk)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
