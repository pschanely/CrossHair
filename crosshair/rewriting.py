import ast
import collections
import copy
import mypy.nodes
from typing import *

from asthelpers import astcopy, astparse, PureNodeTransformer

try:
    import astunparse  # type: ignore

    def unparse(e):
        return astunparse.unparse(e).strip()
except Exception:
    raise
    unparse = ast.dump


def isvarnode(node):
    return type(node) is mypy.nodes.NameExpr and node.name[0] == '$'


class PatternVarPreprocessor(PureNodeTransformer):
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
            return astcopy(node, id=astvar)

    def visit_Expr(self, node):  # promote variable Expr statements
        node = super().generic_visit(node)
        if isvarnode(node.value):
            return node.value
        return node


def preprocess_pattern(*nodes):
    processor = PatternVarPreprocessor()
    if len(nodes) == 1:
        return processor.visit(nodes[0])
    else:
        return [processor.visit(node) for node in nodes]


class PatternVarReplacer(PureNodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        nodeid = node.id
        if nodeid[0] == '$':
            return copy.deepcopy(self.mapping[nodeid[1:]])
        else:
            return node

    def generic_visit(self, node):
        result = super().generic_visit(node)
        # flatten any nested lists
        if hasattr(result, 'body'):
            newbody = []
            for bodyitem in result.body:
                if isinstance(bodyitem, list):
                    newbody.extend(bodyitem)
                else:
                    newbody.append(bodyitem)
            result.body = newbody
        return result


def replace_pattern_vars(node, bindings):
    node = copy.deepcopy(node)
    return PatternVarReplacer(bindings).visit(node)


Asts = Union[mypy.nodes.Node, List[mypy.nodes.Node]]
AstBindings = Dict[str, Asts]
_MATCHER: Dict[type, Callable] = {}


def matches(
        node: Asts, patt: Asts, bind: AstBindings) -> bool:
    # print(' -- matches() -- ')
    # print(('  node=', unparse(node) if isinstance(node,list) else ast.dump(node)))
    # print(('  patt=', unparse(patt) if isinstance(patt,list) else ast.dump(patt)))
    typ = type(patt)
    if isvarnode(patt):
        bind[cast(ast.Name, patt).name[1:]] = node
        return True
    if typ is not type(node):
        bind.clear()
        return False
    cb = _MATCHER.get(typ)
    if cb:
        return cb(node, patt, bind)
    else:
        raise Exception('Unhandled node type: ' + str(typ))
        bind.clear()
        return False


def list_matches(
        node: List[mypy.nodes.Node], patt: List[mypy.nodes.Node], bind: AstBindings) -> bool:
    var_indices = [idx for idx, part in enumerate(patt) if isvarnode(part)]
    if len(var_indices) > 1:
        raise Exception('Cannot have multiple variables in a statement list')
    if var_indices:
        lidx = var_indices[0]
        patt_ridx = (var_indices[0] + 1)
        node_ridx = patt_ridx + len(node) - len(patt)
        # print('lidx', lidx, 'ridx', ridx)
        # print('node left ', list(map(unparse,node[:lidx])))
        # print('node right ', list(map(unparse,node[ridx:])))
        if (list_matches(node[:lidx], patt[:lidx], bind) and
            list_matches(node[node_ridx:], patt[patt_ridx:], bind)):
            var_node = cast(mypy.nodes.NameExpr, patt[lidx])
            bind[var_node.name[1:]] = node[lidx:node_ridx]
            return True
        return False
    else:
        # print('raw list compare node ', unparse(node))
        # print('raw list compare patt ', unparse(patt))
        return (len(node) == len(patt) and
                all(matches(ni, pi, bind) for (ni, pi) in zip(node, patt)))

def name_matches(
        node: mypy.nodes.NameExpr, patt: mypy.nodes.NameExpr, bind: AstBindings) -> bool:
    if hasattr(node, 'node') and hasattr(patt, 'node'):
        return getattr(node, 'node') is getattr(patt, 'node')
    return node.name == patt.name

_MATCHER[mypy.nodes.NameExpr] = name_matches
_MATCHER[mypy.nodes.CallExpr] = lambda n, p, b : (
    matches(n.func, p.func, b) and len(n.args) == len(p.args) and
    all(matches(ni, pi, b) for (ni, pi) in zip(n.args, p.args))
)
#_MATCHER[mypy.nodes.ast.Module] = lambda n, p, b : (
#    matches(n.body, p.body, b)
#)
#_MATCHER[ast.Expression] = lambda n, p, b : (
#    matches(n.value, p.value, b)
#)
_MATCHER[mypy.nodes.IfStmt] = lambda n, p, b : (
    matches(n.expr, p.expr, b) and matches(n.body, p.body, b) and matches(n.elsebody, p.elsebody, b)
)
_MATCHER[mypy.nodes.OpExpr] = lambda n, p, b : (
    type(n.op) is type(p.op) and matches(n.left, p.left, b) and matches(n.right, p.right, b)
)
_MATCHER[mypy.nodes.ComparisonExpr] = lambda n, p, b : (
    type(n.operators) is type(p.operators) and all(matches(ni, pi, b) for (ni, pi) in zip(n.operands, p.operands))
)
_MATCHER[ast.arg] = lambda n, p, b: (
    n.arg == p.arg and n.annotation == p.annotation
)
_MATCHER[mypy.nodes.IntExpr] = lambda n, p, b : n.value == p.value
#_MATCHER[ast.Num] = lambda n, p, b : n.n == p.n
_MATCHER[list] = list_matches



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


class RewriteEngine(ast.NodeTransformer):
    def __init__(self):
        self._index = collections.defaultdict(list)

    def lookup(self, hsh):
        return self._index[hsh]

    def add(self, patt, repl, cond):
        patt, repl = preprocess_pattern(patt, repl)
        self.lookup(patthash(patt)).append((patt, repl, cond))

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
