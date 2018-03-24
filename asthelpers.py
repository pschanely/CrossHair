import ast
import builtins
import doctest
import functools
import hashlib
import numpy

try:
    import astunparse
    def unparse(e):
        return astunparse.unparse(e).strip()
except:
    raise
    unparse = ast.dump

def astparse(codestring):
    return ast.parse(codestring).body[0]

def exprparse(codestring):
    return ast.parse(codestring).body[0].value

def astcopy(node, **overrides):
    ''' Shallow copies an ast node, possibly including some kw changes. '''
    args = {k:getattr(node, k, None) for k in node._fields}
    args.update(overrides)
    return type(node)(**args)

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

def astcall(fn, *args):
    return ast.Call(func=fn, args=args, keywords=())

class PureNodeTransformer:
    ''' Like ast.NodeTransformer, but does not modify the original AST. '''

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        newfields = {}
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                newfields[field] = [self.visit(item) if isinstance(item, ast.AST) else item for item in value]
            elif isinstance(value, ast.AST):
                newfields[field] = self.visit(value)
            else:
                newfields[field] = value
        ret = type(node)(**newfields)
        return ret

def argument_preconditions(args):
    preconditions = []
    for a in args:
        if not a.annotation: continue
        preconditions.append(astcall(a.annotation, ast.Name(id=a.arg)))
    return preconditions

def remove_annotations(node):
    '''
    >>> unparse(remove_annotations(astparse('def foo(x:isint, y=None)->2: pass')))
    'def foo(x, y=None):\\n    pass'
    '''
    class AnnotationRemover(PureNodeTransformer):
        def visit_arg(self, arg):
            return astcopy(arg, annotation=None)
        def visit_FunctionDef(self, fdef):
            return astcopy(self.generic_visit(fdef), returns=None)
    return AnnotationRemover().visit(node)

def apply_ast_template(template, **mapping):
    '''
    Replaces Name nodes in an ast template with the given mapping.
    >>> unparse(apply_ast_template(exprparse('2'), X=4))
    '2'
    >>> unparse(apply_ast_template(exprparse('2 > 4'), X=4))
    '(2 > 4)'
    '''
    class ParamReplacer(PureNodeTransformer):
        def visit_Name(self, node):
            return mapping.get(node.id, node)
    return ParamReplacer().visit(template)

class FnExprError(Exception):
    pass

def fn_expr(fn):
    '''
    Given either a Lambda or FunctionDef, returns appropriate body expression.
    Fails is the function is not expression-based.
    '''
    fntype = type(fn)
    # if fntype == ast.Expr:
    #     fn = fn.value
    #     fntype = type(fn)
    if fntype == ast.FunctionDef:
        stmts = fn.body
        # filter out comments (or other "value" statements)
        stmts = [s for s in stmts if type(s) != ast.Expr]
        if len(stmts) > 1:
            raise FnExprError('More than one statement in function body')
        if len(stmts) == 0:
            raise FnExprError('No statements in function body')
        if type(stmts[0]) != ast.Return:
            raise Exception(type(stmts[0]))
        return stmts[0].value
    elif fntype == ast.Lambda:
        return fn.body
    else:
        raise Exception('Unable to find function in '+str(type(fn)))

def _isstarred(node):
    return type(node) == ast.Starred

def arguments_positional_minmax(arguments):
    '''
    Given an arguments node (from a Lambda or FunctionDef), returns the
    minimum and (possibly None) maximum number of positional arguments that
    are allowed.
    >>> arguments_positional_minmax(exprparse('lambda a,b,c=4:...').args)
    (2, 3)
    >>> arguments_positional_minmax(exprparse('lambda *,c=4:...').args)
    (0, 0)
    >>> arguments_positional_minmax(exprparse('lambda a,*b,c=4:...').args)
    (1, None)
    '''
    maxargs = None if arguments.vararg else len(arguments.args)
    minargs = len(arguments.args) - len(arguments.defaults)
    return (minargs, maxargs)

def call_positional_minmax(call_node):
    '''
    Given a Call node, returns the minimum and maximum number of positional
    arguments that could be provided.
    >>> call_positional_minmax(exprparse('f(1,*x,2,*y)'))
    (2, None)
    >>> call_positional_minmax(exprparse('f(*x)'))
    (0, None)
    >>> call_positional_minmax(exprparse('f(x, y)'))
    (2, 2)
    '''
    positional = call_node.args
    maxparams = None if any(_isstarred(p) for p in positional) else len(positional)
    minparams = sum(1 for p in positional if not _isstarred(p))
    return (minparams, maxparams)

def call_positional_expr(call_node, idx, *extra):
    params = call_node.args
    while idx > 0 and not _isstarred(params[0]):
        idx -= 1
        params = params[1:]
        if not params: # return default
            return extra[0]
    if idx == 0 and not _isstarred(params[0]): # the easy case
        return params[0]
    params = [p.value if _isstarred(p) else ast.Tuple(elts=[p]) for p in params]
    concatenated = functools.reduce(lambda x,y:ast.BinOp(x,ast.Add,y), params)
    ret = ast.Subscript(
        value=concatenated,
        slice=ast.Index(value=ast.Num(n=idx)))
    if extra:
        default = extra[0]
        ret = apply_ast_template(
            exprparse('R if I < len(R) else D'),
            R=ret, I=idx, D=default)
    return ret

def inline(call_node, func_to_inline):
    '''
    Attempts to fill in arguments of a function using a call node.
    Returns a new (specialized) function body.

    >>> foo = astparse('def foo(x): return x + 2')
    >>> expr = astparse('foo(42)').value
    >>> inlined = inline(expr, foo)
    >>> unparse(inlined)
    '(42 + 2)'
    >>> unparse(foo)
    'def foo(x):\\n    return (x + 2)'
    >>> unparse(inline(astparse('foo(*(100,))').value, foo))
    '((100,)[0] + 2)'
    >>> unparse(inline(astparse('foo(*bar)').value, foo))
    '(bar[0] + 2)'
    '''
    args = func_to_inline.args

    minp, maxp = call_positional_minmax(call_node)
    mina, maxa = arguments_positional_minmax(func_to_inline.args)
    if ((maxp is not None and maxp < mina) or
        (maxa is not None and maxa < minp)):
        return exprparse('raise TypeError("incorrect number of arguments")')
    guaranteed_ok = mina <= minp and (maxa is None or maxp <= maxa)
    mapping = {}
    non_defaulted_count = len(args.args) - len(args.defaults)
    for idx, argname in enumerate(a.arg for a in args.args):
        if idx < non_defaulted_count:
            callexpr = call_positional_expr(call_node, idx)
        else:
            default = args.defaults[idx - non_defaulted_count]
            callexpr = call_positional_expr(call_node, idx, default)
        mapping[argname] = callexpr
    # TODO keyword argement handling
    # for keyword in call_node.keywords:
    #     arg_mapping[keyword.arg] = keyword.value

    body = fn_expr(func_to_inline)
    ret = apply_ast_template(body, **mapping)
    if not guaranteed_ok:
        # wrap result in run-time argument check
        pass
    return ret

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

class ScopeTracker(PureNodeTransformer):
    def __init__(self):
        self.scopes = []
    def resolve(self, node):
        '''
        Finds source of value, whether defined elsewhere or just returns
        the original node.
        Different references that can be determined to be equal will be
        reference-equivalent.
        '''
        # print('resolve ', type(node), self.scopes)
        nodetype = type(node)
        if nodetype is ast.Name:
            refname = node.id
            for bindings in reversed(self.scopes):
                if refname in bindings:
                    # recursive in case the binding is to another name:
                    # TODO: but this doesn't work because the scopes won't be correct on next call!?
                    # disabling recursive call for now.
                    # return self.resolve(bindings[refname]) 
                    return bindings[refname]
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
        # print('lambda scopes {', self.scopes)
        node = astcopy(node, body=self.visit(node.body))
        # print('lambda scopes }', self.scopes)
        self.scopes.pop()
        return node

if __name__ == "__main__":
    import doctest
    doctest.testmod()
