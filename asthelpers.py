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

def fn_expr(fn):
    '''
    Given either a Lambda or FunctionDef, returns appropriate body expression.
    Fails is the function is not expression-based.
    '''
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
        # print('lambda scopes {', self.scopes)
        node = astcopy(node, body=self.visit(node.body))
        # print('lambda scopes }', self.scopes)
        self.scopes.pop()
        return node

def hash_(s):
    h = hashlib.sha256()
    h.update(s.encode())
    return h

def _deterministic_hash(stringdatums):
    s = ''.join(stringdatums)
    return int.from_bytes(hash_(s).digest(), byteorder='big')

def _fn_object_name(fn):
    fntype = type(fn)
    if fntype == ast.Name:
        return fn.id
    elif fntype == ast.FunctionDef:
        return fn.name
    else:
        return type(fn).__name__

_bin_ops = [
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult
]
class SemanticAstHasher(ScopeTracker):
    def __init__(self, num_buckets=211):
        super().__init__()
        self.buckets = numpy.zeros((num_buckets,))
        self._stoptypes = set([ast.Num, ast.Name, ast.NameConstant])
        self._skiptypes = set([ast.Load] + _bin_ops)
    def root_label(self, node):
        nodetype = type(node)
        if nodetype == ast.BinOp:
            return [node.op.__class__.__name__]
        elif nodetype == ast.Call:
            return self.root_label(node.func)
            target = self.resolve(node.func)
            # print('call target', unparse(target))
            return [_fn_object_name(target)]
        elif nodetype == ast.Name:
            target = self.resolve(node)
            # print(node.id,' resolve to ',type(target), target)
            if target is not node:
                return ['uuu']
            return [node.id]
        else:
            return [node.__class__.__name__]
    # def visit_BinOp(self, node):
    #     return self.generic_visit(node, [node.op.__class__.__name__])
    # def visit_Call(self, node):
    #     target = self.resolve(node.func)
    #     name = target.name if type(target) == ast.FunctionDef else target.__class__.__name__
    #     return self.generic_visit(node, [name])
    # def visit_Lambda(self, node):
    #     return super().generic_visit(node)
    def generic_visit(self, node):
        if type(node) in self._skiptypes: return node
        hashsrc = self.root_label(node)
        self.add(hashsrc) # for the unary registration
        if type(node) in self._stoptypes:
            return node
        for field, value in ast.iter_fields(node):
            if not isinstance(value, list):
                value = (value,)
            for (idx, item) in enumerate(value):
                if isinstance(item, ast.AST):
                    if type(item) in self._skiptypes: continue
                    self.visit(item)
                    self.add(hashsrc + [field, str(idx)] + self.root_label(item))
        return node
    def add(self, hashsrc):
        val = _deterministic_hash(hashsrc)
        # print('  ', val % len(self.buckets), ' <= ', hashsrc)
        self.buckets[val % len(self.buckets)] = 1.0

def semantic_hash(astobject):
    # print('semantic_hash', unparse(astobject))
    h = SemanticAstHasher()
    h.visit(astobject)
    return h.buckets

if __name__ == "__main__":
    import doctest
    doctest.testmod()
