import ast
import collections
import copy
import doctest
import functools
import inspect
import operator
import os
import sys
import z3

import crosshair
from asthelpers import *
import prooforacle

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
                if refname[3].isupper():
                    raise Exception('Invalid Z3 intrinsic: "'+refname+'"')
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
    return ast.Compare(left=item, ops=[ast.In()], comparators=[lst])

def ast_eq(left, right):
    return ast.Compare(left=left, ops=[ast.Eq()], comparators=[right])

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
        return node # raise Exception()
    else:
        ret = inline(node, func)
    # print('beta reduce', unparse(node), unparse(ret))
    return ret

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
    # # ('F(reduce(R, L, I))', 'reduce(R`, map(F, L), I)', reduce_fn_check),
]:
    basic_simplifier.add(exprparse(patt), exprparse(repl), condition)

# expr = exprparse('IsTruthy(isint(c) and isint(d))')
# rewritten = basic_simplifier.rewrite(expr)
# print('rewrite engine test', unparse(expr), unparse(rewritten))



PyFunc = z3.DeclareSort('PyFunc')
# We parameterize undef with an uninterpreted value so that undefs aren't known to be equal to each other:
PyUndef = z3.DeclareSort('PyUndef')
Unk = z3.Datatype('Unk')
Unk.declare('none')
Unk.declare('bool', ('tobool', z3.BoolSort()))
Unk.declare('int', ('toint', z3.IntSort()))
Unk.declare('string', ('tostring', z3.StringSort()))
Unk.declare('func', ('tofunc', PyFunc))
Unk.declare('c', ('tl', Unk), ('hd', Unk))
Unk.declare('_') # empty tuple
Unk.declare('undef', ('toundef', PyUndef))
Unk = Unk.create()
App = z3.Function('.', Unk, Unk, Unk)


class ZHolder(): pass
Z = ZHolder()
Z.Wrapbool = Unk.bool
Z.Wrapint = Unk.int
Z.Wrapstring = Unk.string
Z.Wrapfunc = Unk.func
Z.Wrapnone = Unk.none
Z.C = Unk.c
Z.Bool = Unk.tobool
Z.Int = Unk.toint
Z.String = Unk.tostring
Z.Func = Unk.tofunc
Z.Head = Unk.hd
Z.Tail = Unk.tl
Z.Isbool = lambda x:Unk.is_bool(x)
Z.Isint = lambda x:Unk.is_int(x)
Z.Isstring = lambda x:Unk.is_string(x)
Z.Isfunc = lambda x:Unk.is_func(x)
Z.Istuple = lambda x:z3.Or(Unk.is_c(x), Unk.is__(x))
Z.Isnone = lambda x:Unk.is_none(x)
Z.Isdefined = lambda x:z3.Not(Unk.is_undef(x))
Z.Isundefined = lambda x:Unk.is_undef(x)
Z.Eq = lambda x,y: x == y
Z.Neq = lambda x,y: x != y
Z.Distinct = z3.Distinct
Z.T = z3.Function('T', Unk, z3.BoolSort())
Z.F = z3.Function('F', Unk, z3.BoolSort())
Z.N = Unk.none
Z.Length = z3.Length
Z.Implies = z3.Implies
Z.And = z3.And
Z.Or = z3.Or
Z.Not = z3.Not
Z.Negate = lambda x: -x
Z.Lt = lambda x,y: x < y
Z.Lte = lambda x,y: x <= y
Z.Gt = lambda x,y: x > y
Z.Gte = lambda x,y: x >= y
Z.Ite = z3.If
Z.Add = lambda x,y: x + y
Z.Sub = lambda x,y: x - y
Z.Extract = z3.Extract#lambda s,o,l: z3.Extract(Unk.tostring(s), o, l)
Z.Seqconcat = z3.Concat
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
def get_scopes_for_def(fndef):
    return _fndef_to_moduleinfo[fndef].get_scopes_for_def(fndef)
def get_fninfo_for_def(fndef):
    if fndef not in _fndef_to_moduleinfo:
        raise Exception('unknown function definition: '+unparse(fndef))
    for fninfo in _fndef_to_moduleinfo[fndef].functions.values():
        if fninfo.get_definition() == fndef:
            return fninfo
    raise Exception('fninfo not found')

class ModuleInfo:
    def __init__(self, module, module_ast):
        self.module = module
        self.ast = module_ast
        self.functions = {}#'': FnInfo('', self)} # this is for global assertions, which have no corresponding definitional assertion
        self.function_order = []
    def fn(self, name):
        if name not in self.functions:
            self.functions[name] = FnInfo(name, self)
            self.function_order.append(name)
        return self.functions[name]
    def get_fn(self, name):
        return self.functions[name]
    def get_scopes_for_def(self, fndef):
        class FnFinder(PureScopeTracker):
            def __init__(self):
                super().__init__()
                self.hit = None
            def visit_FunctionDef(self, node):
                if node is fndef:
                    # TODO: deepcopy was here before, but causes problems for
                    # borderset stuff. Consider why I thought we needed it.
                    #self.hit = [copy.deepcopy(s) for s in self.scopes]
                    self.hit = [s.copy() for s in self.scopes]
                return node
        f = FnFinder()
        f.visit(self.ast)
        return f.hit

def astand(clauses):
    if len(clauses) == 1:
        return clauses[0]
    else:
        return ast.BoolOp(op=ast.And, values=clauses)

def fn_returns_istrue(fn_ast):
    if fn_ast.returns:
        predicate = fn_ast.returns
        return isinstance(predicate, ast.Name) and predicate.id == 'istrue'
    return False

def fn_annotation_assertion(fn, assert_by_name):
    args = fn_args(fn)
    preconditions = argument_preconditions(args)
    if not preconditions and not fn.returns:
        return None
    predicate = fn.returns if fn.returns else ast.Name(id='isdefined')
    if assert_by_name:
        varnamerefs = [ast.Name(id=a.arg) for a in args]
        expectation = astcall(predicate, astcall(ast.Name(id=fn.name), *varnamerefs))
    else:
        expectation = fn_expr(fn)
        if not fn_returns_istrue(fn):
            expectation = astcall(predicate, expectation)
    fdef = ast.FunctionDef(
        name = fn.name,
        args=fn.args,
        body=[ast.Return(value=expectation)],
        decorator_list=fn.decorator_list,
        returns=None
    )
    # print('expectation')
    # print(unparse(expectation))
    # print(unparse(fdef))
    return fdef

def calls_name(expr):
    if type(expr) == ast.Call and type(expr.func) == ast.Name:
        return expr.func.id

def ch_option_true(expr, key, default):
    astval = find_ch_options(expr).get(key)
    if astval is None:
        return default
    return astval.value

def find_ch_options(expr):
    if type(expr) == ast.FunctionDef:
        for dec in expr.decorator_list:
            if calls_name(dec) == 'ch':
                return dict((keyword.arg, keyword.value) for keyword in dec.keywords)
    return {}

def fn_defining_assertions(fn, suppress_definition=False):
    #print('def expectation', fn)
    #print(unparse(fn))
    args = fn_args(fn)
    assertions = [
        ast.FunctionDef(
            name = fn.name + '_isfunc',
            args=ast.arguments(args=[],defaults=[],vararg='',kwarg='',kwonlyargs=[]),
            body=[ast.Return(value=astcall(ast.Name(id='isfunc'), ast.Name(id=fn.name)))],
            decorator_list=[],
            returns=None
        ),
    ]
    if ch_option_true(fn, 'use_definition', True) and not suppress_definition:
        varnamerefs = [ast.Name(id=a.arg) for a in args]
        call_by_name = astcall(ast.Name(id=fn.name), *varnamerefs)
        eq_expr = astcall(ast.Name(id='_z_eq'), call_by_name, fn_expr(fn))
        assertions.append(ast.FunctionDef(
            name = fn.name+'_definition',
            args=remove_annotations(fn.args),
            body=[ast.Return(value=astcall(ast.Name(id='_z_wrapbool'), eq_expr))],
            decorator_list=[],
            returns=astcall(ast.Name(id='istrue')),
        ))
    return assertions

def compile_ast(ast_definition, containing_module):
    fn_name = ast_definition.name
    ast_definition = ast.Module(body=[ast_definition])
    code_object = compile(ast_definition, containing_module.__file__, 'exec')
    locals_dict = {}
    globals_dict = {}
    locals_dict.update(containing_module.__dict__)
    exec(code_object, locals_dict, globals_dict)
    return locals_dict[fn_name]

class FnInfo:
    def __init__(self, name, moduleinfo):
        self.name = name
        self.moduleinfo = moduleinfo
        self.assertions = []
        self.definition = None
        self.definitional_assertion = None # assertion implied by the types

    def add_assertion(self, assertion, containing_module):
        self.assertions.append( (assertion, compile_ast(assertion, containing_module)) )
        _fndef_to_moduleinfo[assertion] = self.moduleinfo # TODO is this used/useful?

    def set_definition(self, definition):
        if self.definition is not None:
            raise RedefinitionError(definition, self.name)
        self.definition = definition
        _fndef_to_moduleinfo[definition] = self.moduleinfo

        assert_by_name = not ch_option_true(definition, 'use_definition', True)
        definitional_assertion = fn_annotation_assertion(definition, assert_by_name)
        if definitional_assertion:
            self.definitional_assertion = definitional_assertion
    def get_defining_assertions(self, suppress_definition=False):
        # not the "definitional" assertion: this asserts that the function is equal to its body
        # and that the object in question is a function
        return fn_defining_assertions(self.definition, suppress_definition=suppress_definition)

    def get_definition(self):
        return self.definition
    def get_assertions(self):
        return self.assertions

_fninfo_by_def = {}
def parse_pure(module):
    with open(module.__file__) as fh:
        module_ast = ast.parse(fh.read())
    ret = ModuleInfo(module, module_ast)
    # TODO : parse imports; track scope for duplicate definitions
    for item in module_ast.body:
        itemtype = type(item)
        if itemtype == ast.FunctionDef:
            try:
                name = item.name
                ret.fn(name).set_definition(item)
                _fninfo_by_def[item] = ret.get_fn(name)
            except FnExprError as e:
                raise LocalizedError(item, str(e))
        elif itemtype == ast.Import:
            pass
    return ret

_parsed_modules = {} # module object to ModuleInfo
def get_module_info(module):
    if module not in _parsed_modules:
        _parsed_modules[module] = parse_pure(module)
    return _parsed_modules[module]

def get_all_dependent_fninfos(fn_info):
    for moduleinfo in _parsed_modules.values():
        names = moduleinfo.function_order
        isimported = fn_info.name not in names
        if not isimported:
            names = names[:names.index(fn_info.name)]
        for fn_name in names:
            yield moduleinfo.functions[fn_name], isimported

_pure_defs = get_module_info(crosshair)

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
        return Unk.c(accumulator, arg)

def make_args(args):
    accumulator = Unk._
    encountered_star = False
    for arg in args:
        if (type(arg) == ast.Starred):
            encountered_star = True
            if accumulator == Unk._:
                accumulator = arg.value
            else:
                accumulator = Z.Concat(accumulator, arg.value)
        else:
            if encountered_star:
                accumulator = Z.Concat(accumulator, Unk.c(Unk._, arg))
            else:
                accumulator = Unk.c(accumulator, arg)
    return accumulator

def z3apply(fnval, args):
    if id(fnval) in _z3_fn_ids:
        return fnval(*args)
    else:
        return App(fnval, make_args(args))

class Z3BindingEnv(collections.namedtuple('Z3BindingEnv',['refs','support'])):
    def __new__(cls, refs=None):
        return super(Z3BindingEnv, cls).__new__(cls, refs if refs else {}, [])

class ClientError(Exception):
    pass

class LocalizedError(ClientError):
    def __init__(self, node, message):
        self.filename = ''
        self.node = node
        self.line = node.lineno
        self.col = getattr(node, 'col_offset', 0)
        ClientError.__init__(self, message)
        
class ResolutionError(LocalizedError):
    def __init__(self, node):
        LocalizedError.__init__(self, node, 'Undefined identifier: ' + node.id)

class RedefinitionError(LocalizedError):
    def __init__(self, node, name):
        LocalizedError.__init__(self, node, name + ' redefined')

class UnsoundnessError(ClientError):
    def __init__(self, unsat_core):
        self.unsat_core = unsat_core
        ClientError.__init__(self, 'Prior unsoundness detected: '+
                    ', '.join(sorted(unsat_core)))


class Z3Transformer(PureScopeTracker):
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
            raise ResolutionError(definition)
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
            #print('register new ref: ', name, id(definition), " at ",
            #      getattr(definition, 'lineno', ''), ":",
            #      getattr(definition, 'col_offset', '')
            #)
            refs[definition] = z3.Const(name, Unk)
        return refs[definition]

    def visit_IfExp(self, node):
        test, body, orelse = map(self.visit, (node.test, node.body, node.orelse))
        return z3.If(Z.T(test), body, orelse)
    
    def visit_Subscript(self, node):
        return self.handle_subscript(self.visit(node.value), node.slice)

    def handle_subscript(self, value, subscript):
        subscripttype = type(subscript)
        if subscripttype is ast.Index:
            z3fn = self.register(fn_for_op('Get'))
            return z3apply(z3fn, (value, self.visit(subscript.value)))
        elif subscripttype is ast.Slice:
            if subscript.step is None:
                z3fn = self.register(fn_for_op('Sublist'))
                lower = self.visit(subscript.lower) if subscript.lower is not None else Z.Wrapnone
                upper = self.visit(subscript.upper) if subscript.upper is not None else Z.Wrapnone
                return z3apply(z3fn, (value, lower, upper))
            else:
                return z3apply(fn_for_op('SteppedSublist'), (value, self.visit(subscript.lower), self.visit(subscript.upper), self.visit(subscript.step)))
        elif subscripttype is ast.ExtSlice:
            return functools.reduce(
                lambda a, b: z3apply(fn_for_op('Add'), (a, b)),
                (self.handle_subscript(value, dim) for dim in index.dims))

    def visit_NameConstant(self, node):
        return _z3_name_constants[node.value]

    def visit_Name(self, node):
        ret = self.register(self.resolve(node))
        return ret

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

    def visit_None(self, node):
        return Z.Wrapnone
    
    def visit_Num(self, node):
        return Z.Wrapint(node.n)

    def visit_Str(self, node):
        return Z.Wrapstring(z3.StringVal(node.s))

    def function_body_to_z3(self, func):
        argpairs = [(a.value.arg,True) if type(a)==ast.Starred else (a.arg,False) for a in func.args.args]

        argnames = [a.arg for a in func.args.args]
        argvars = [z3.Const(name, Unk) for name in argnames]
        arg_name_to_var = dict(zip(argnames, argvars))
        arg_name_to_def = {name: ast.arg(arg=name) for name in argnames}
        self.scopes.append(arg_name_to_def)
        for name, definition in arg_name_to_def.items():
            self.env.refs[definition] = arg_name_to_var[name]
        z3body = self.visit(func.body)
        self.scopes.pop()

        z3vars = [
            ast.Starred(value=arg_name_to_var[n]) if is_starred else arg_name_to_var[n]
            for n, is_starred in argpairs
        ]

        return z3body, z3vars

    def visit_Lambda(self, node):
        name = 'lambda_{}_{}'.format(node.lineno, node.col_offset)
        funcval = Z.Wrapfunc(z3.Const(name, PyFunc))

        z3body, z3vars = self.function_body_to_z3(node)
        z3application = z3apply(funcval, z3vars)
        stmt = z3.ForAll(z3vars, z3application == z3body, patterns=[z3application])
        self.env.support.append(stmt)
        self.env.support.append(Unk.is_func(funcval))
        return funcval

    def visit_Tuple(self, node):
        if type(node.ctx) != ast.Load:
            raise Exception(ast.dump(node))
        params = [self.visit(a) for a in node.elts]
        return make_args(params)

    def visit_Call(self, node):
        newnode = beta_reduce(node)
        if newnode is not node:
            return self.visit(newnode)
        z3fn = self.visit(node.func)
        # Special case for quantifiers:
        if z3fn is z3.ForAll or z3fn is z3.Exists:
            targetfn = node.args[0]
            if type(targetfn) != ast.Lambda:
                raise LocalizedError(targetfn, 'Quantifier argument must be a lambda')
            z3body, z3vars = self.function_body_to_z3(targetfn)
            return z3fn(z3vars, Z.T(z3body))

        params = [self.visit(a) for a in node.args]
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

def solve(assumptions, conclusion, oracle, options=None):
    make_repro_case = os.environ.get('CH_REPRO') in ('1','true')
    #solver = z3.Then('macro-finder', 'qe','smt').solver()
    #solver = z3.Then(z3.Tactic('skip'), z3.Tactic('smt')).solver()
    solver = z3.Solver()
    #applier = z3.Tactic('skip').apply
    #applier = z3.Tactic('macro-finder').apply
    #applier = z3.Tactic('simplify').apply
    #applier = z3.Tactic('qe2').apply
    #applier = z3.Tactic('solve-eqs').apply
    z3.set_param(
        verbose = 0,
    )
    opts = {
        'timeout': 1000,
        'unsat_core': True,
        'core.minimize': True,
        'macro-finder': True,
        #'candidate_models': True,
        'smt.mbqi': False,
        #'mbqi.force_template': 0,
        'smt.pull-nested-quantifiers': True,
    }
    if options:
        opts.update(options.get('z3_opts', {}))
    solver.set(**opts)

    #required_assumptions = [a for a in assumptions if a.score is None]
    #assumptions = [a for a in assumptions if a.score is not None]
    #assumptions.sort(key=lambda a:a.score)
    #assumptions = assumptions[:120]
    for l in assumptions:
        print('baseline:', l)
    print('conclusion (in z3):', conclusion)
    #assumptions = required_assumptions + assumptions

    stmt_index = {}
    if make_repro_case:
        for assumption in assumptions:
            solver.add(assumption.expr)
        solver.add(z3.Not(conclusion))
    else:
        #print('**************')
        #goal = z3.Goal()
        #for a in assumptions:
        #    goal.add(a.expr)
        #goal.add(z3.Not(conclusion))
        #print(applier(goal))#z3.Goal(conclusion))#*assumptions)# + [
        #print('**************')
        for idx, assumption in enumerate(assumptions):
            label = 'assumption{}'.format(idx)
            stmt_index[label] = assumption
            solver.assert_and_track(assumption.expr, label)
        solver.assert_and_track(z3.Not(conclusion), 'conclusion')

    core = set()
    try:
        ret = solver.check()
        if ret == z3.unsat:
            if not make_repro_case:
                core = set(map(str, solver.unsat_core()))
                #for stmt in core:
                #    if stmt == 'conclusion': continue
                #    ast = stmt_index[stmt].src_ast
                #    if ast:
                #        print(unparse(ast))
                #        #print(' '+stmt_index[stmt].expr.sexpr())
                if 'conclusion' not in core:
                    raise UnsoundnessError(
                        [stmt_index[stmt].src_ast.name for stmt in core if stmt_index[stmt].src_ast])
            ret = True
        elif ret == z3.sat:
            print('BEGIN COUNTER EXAMPLE')
            print(solver.model())
            print('END COUNTER EXAMPLE')
            ret = False
        else:
            print('NOT SAT OR UNSAT: ', ret, ' ', solver.reason_unknown())
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
    report = { 'statements': [{
        'body':unparse(a.src_ast), 'used':k in core, 'name':a.src_ast.name
    } for k, a in stmt_index.items() if a.src_ast]}
    return ret, report

def find_weight(expr):
    weightast = find_ch_options(expr).get('weight')
    return ast.literal_eval(weightast) if weightast else 10

def find_patterns(expr, found_implies=False):
    #print('find_patterns ', unparse(expr))
    if type(expr) == ast.FunctionDef:
        declared_patterns = []
        options = find_ch_options(expr)
        patterns = [options['pattern']] if 'pattern' in options else []
        if 'patterns' in options:
            patterns.extend(options['patterns'].elts)
        for pattern in patterns:
            if isinstance(pattern, ast.Lambda):
                pattern = ast.List(elts=(pattern,))
            multipattern_contents = []
            for lam in pattern.elts:
                if type(lam) != ast.Lambda:
                    raise LocalizedError(lam, 'Non-lambda pattern')
                if tuple(a.arg for a in lam.args.args) != tuple(a.arg for a in expr.args.args):
                    print('pattern mismatch: ', unparse(lam.args), unparse(expr.args))
                    raise LocalizedError(lam, 'Pattern arguments do not match function arguments')
                #print('DECL ', unparse(lam.body))
                multipattern_contents.append(lam.body)
            declared_patterns.append(multipattern_contents)
        if declared_patterns:
            return declared_patterns
        found_patterns = find_patterns(fn_expr(expr))
        #print('Inferred pattern(s) ', [[unparse(p) for p in patts] for patts in found_patterns])
        return found_patterns
    if type(expr) == ast.Lambda:
        return find_patterns(fn_expr(expr))
    fnname = calls_name(expr)
    if fnname == '_z_wrapbool':
        return find_patterns(expr.args[0], found_implies=found_implies)
    elif fnname == 'implies' or fnname == '_z_implies':
        return find_patterns(expr.args[1], found_implies=True)
    else:
        # check for equals in various forms
        if fnname == '_z_eq':
            return find_patterns(expr.args[0], found_implies=found_implies)
            # return find_patterns(expr.args[1], found_implies=found_implies)
        if type(expr) == ast.Compare:
            if len(expr.ops) == 1 and type(expr.ops[0]) == ast.Eq:
                # return find_patterns(expr.comparators[0], found_implies=found_implies)
                return find_patterns(expr.left, found_implies=found_implies)
    return [[expr]]

_Z3_implies_decl = z3.Implies(False, True).decl()
_Z3_eq_decl = (z3.Int('x') == 0).decl()
def z3_implication_equality_form(expr, found_implies=False):
    if expr.decl() == _Z3_implies_decl:
        return z3_implication_equality_form(expr.arg(1), found_implies=True)
    elif expr.decl() == _Z3_eq_decl:
        return expr.arg(0), expr.arg(1)
    if found_implies:
        return (expr,)
    return None

_NO_VAL = object()
_DISABLE = object()
def assertion_fn_to_z3(fn, env, scopes, weight=_NO_VAL, wrap_in_forall=True, options=None):
    args = fn_args(fn)
    expr = fn_expr(fn)
    scopes.append({a.arg:a for a in args})
    z3expr = to_z3(expr, env, scopes)
    #print('  assertion_fn_to_z3 ', getattr(fn,'name'), ' : ', z3expr.sexpr())
    if z3expr.decl() == Z.Wrapbool:
        z3expr = z3expr.arg(0)
    else:
        z3expr = Z.T(z3expr)

    #print(unparse(fn))
    forall_kw = {}
    # last problem with weight: the Concat axioms had different weights, causing tuple-proof-troubles
    #forall_kw['weight'] = find_weight(fn) if weight is _NO_VAL else weight
    if args:
        if (not options) or options.get('prove_with_patterns',True):
            multipatterns = find_patterns(fn)
            if multipatterns:
                # TODO check that the pattern expression covers the bound variables
                forall_kw['patterns'] = [
                    to_z3(m[0], env, scopes) if len(m) == 1 else z3.MultiPattern(*[to_z3(p, env, scopes) for p in m])
                    for m in multipatterns
                ]

    preconditions = argument_preconditions(args)
    if preconditions:
        if len(preconditions) == 1:
            z3expr = Z.Implies(Z.T(to_z3(preconditions[0], env, scopes)), z3expr)
        else:
            z3expr = Z.Implies(Z.And([Z.T(to_z3(p, env, scopes)) for p in preconditions]), z3expr)

    z3arg_constants = [env.refs[a] for a in args if a in env.refs]
    if wrap_in_forall and z3arg_constants:
        try:
            #print(z3arg_constants, z3expr, forall_kw)
            z3expr = z3.ForAll(z3arg_constants, z3expr, **forall_kw)
        except z3.z3types.Z3Exception as e:
            if 'invalid pattern' in str(e):
                raise LocalizedError(expr, 'Unable to infer pattern')
            raise e

    #print('  ', 'forallkw', forall_kw, z3expr.sexpr())
    scopes.pop()
    return z3expr

def make_statement(fn, env, scopes, weight_overrides, options=None):
    weight = weight_overrides(fn.name)
    if weight is None:
        weight = find_weight(fn)
    #print('find weight ', weight, ' in ', fn.name)
    if weight is _DISABLE:
        return None
    #print(unparse(fn))
    return Z3Statement(fn, assertion_fn_to_z3(fn, env, scopes, options=options))

class Z3Statement:
    def __init__(self, src_ast, expr):
        self.score = None
        self.src_ast = src_ast
        self.expr = expr
        #print('Z3: ', expr)
    def set_score(self, score):
        self.score = score
    def __str__(self):
        return 'Z3Statement(score={}, src={})'.format(
            self.score, unparse(self.src_ast) if self.src_ast else self.expr.sexpr())

def ast_for_function(fn):
    conclusion_src = inspect.getsource(fn).strip()
    tree = ast.parse(conclusion_src)
    if type(tree) == ast.Module:
        tree = tree.body[0]
    return tree

def check_assertion_fn(fn_definition, compiled_fn, oracle=None, extra_support=(), weight_overrides=lambda n:None):
    options = {
        'prove_with_patterns': ch_option_true(fn_definition, 'prove_with_patterns', True)
    }
    prove_with = find_ch_options(fn_definition).get('prove_with')
    if prove_with:
        names = set(ast.literal_eval(prove_with))
        weight_overrides = lambda n: 10 if n in names else _DISABLE

    fn_info = get_fninfo_for_def(fn_definition)
    counterexample = None
    #if compiled_fn:
    #    counterexample = prooforacle.find_counterexample(compiled_fn)
    #tree = ast_for_function(conclusion_fn)
    ret, report = prove_assertion_fn(fn_info, oracle=oracle, extra_support=extra_support, weight_overrides=weight_overrides, options=options)

    if counterexample is not None:
        print('Counterexample found: ', counterexample)
        if ret is True:
            raise Exception('Counterexample conflicts with proof')
        report['counterexample'] = counterexample
        return False, report
    else:
        if ret is False:
            print('Cannot prove, but cannot find counterexample')

    return ret, report

def prove_assertion_fn(fn_info, oracle=None, extra_support=(), weight_overrides=lambda n:None, options=None):
    env = Z3BindingEnv()

    conclusion_fn = fn_info.definitional_assertion
    print('Checking assertion:')
    print(unparse(conclusion_fn))
    print()

    scopes = get_scopes_for_def(fn_info.definition)
    conclusion = assertion_fn_to_z3(conclusion_fn, env, scopes, wrap_in_forall=False, options=options)

    baseline = []

    for fninfo, isimported in get_all_dependent_fninfos(fn_info):
        try:
            fn_def = fninfo.get_definition()
            subscopes = get_scopes_for_def(fn_def)
            if fninfo.definitional_assertion:
                baseline.append(make_statement(fninfo.definitional_assertion, env, subscopes, weight_overrides, options=options))
            if not fn_returns_istrue(fninfo.definition):
                for a in fninfo.get_defining_assertions(suppress_definition=isimported):
                    baseline.append(make_statement(a, env, subscopes, weight_overrides, options=options))
        except LocalizedError as e:
            e.filename = fninfo.moduleinfo.module.__file__
            raise e
    
    baseline.extend([Z3Statement(None, a) for a in core_assertions(env)])

    baseline = [s for s in baseline if s is not None]
    if oracle is None:
        oracle = prooforacle.TrivialProofOracle()
    # print ()
    print ('conclusion (in python):', unparse(conclusion_fn))
    oracle.score_axioms(baseline, conclusion_fn)
    return solve(baseline, conclusion, oracle, options=options)

def core_assertions(env):
    r = z3.Const('r', Unk)
    g = z3.Const('g', Unk)
    x = z3.Const('x', Unk)
    baseline = []

    # r + () = r
    baseline.append(z3.ForAll([r],
        Z.Eq(Z.Concat(r, Unk._), r),
        patterns=[Z.Concat(r, Unk._)]
    ))

    # () + r = r
    baseline.append(z3.ForAll([r],
        Z.Eq(Z.Concat(Unk._, r), r),
        patterns=[Z.Concat(Unk._, r)]
    ))

    # f(g, *(x,), ...) = f(g, x, ...)
    # invalid for definedness reasons:
    # istuple((g, *(undef,))) is not true!
    #baseline.append(z3.ForAll([x, g],
    #    Z.Eq(
    #        Z.Concat(g, Unk.c(Unk._, x)),
    #        Unk.c(g, x)
    #    ),
    #    patterns=[Z.Concat(g, Unk.c(Unk._, x))]
    #))

    # (g, *(*r, x), ...) = (g, *r, x, ...)
    # invalid for definedness reasons:
    # istuple((g, *(*undef, x))) is not true!
    #baseline.append(z3.ForAll([x, g, r],
    #    Z.Eq(
    #        Z.Concat(g, Unk.c(r, x)),
    #        Unk.c(Z.Concat(g, r), x)
    #    ),
    #    patterns=[Z.Concat(g, Unk.c(r, x))]
    #))

    return baseline



if __name__ == "__main__":
    import doctest
    doctest.testmod()
