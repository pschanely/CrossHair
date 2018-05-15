import ast
import enum
import inspect
import re
import traceback
from typing import *

import z3  # type: ignore

from crosshair.util import debug
from crosshair.asthelpers import astcall
from crosshair.asthelpers import astparse
from crosshair.asthelpers import exprparse
from crosshair.asthelpers import apply_ast_template
from crosshair.asthelpers import unparse
from crosshair.asthelpers import PureNodeTransformer
from crosshair.type_handlers import make_reader
from crosshair.type_handlers import unpack_signature
from crosshair.type_handlers import InputNotUnpackableError
from crosshair.type_handlers import make_z3_var
from crosshair.type_handlers import ExpressionNotSmtable
from crosshair.type_handlers import simplify_value
from crosshair.typed_inspect import signature


def strip_comment_line(line: str) -> str:
    line = line.strip()
    if line.startswith("'''") or line.startswith('"""'):
        line = line[3:]
    if line.endswith("'''") or line.endswith('"""'):
        line = line[:-3]
    return line.strip()


def get_doc_lines(thing: object) -> Iterable[Tuple[int, str]]:
    doc = inspect.getdoc(thing)
    if doc is None:
        return
    lines, line_num = inspect.getsourcelines(thing)
    line_num += len(lines) - 1
    line_numbers = {}
    for line in reversed(lines):
        line_numbers[strip_comment_line(line)] = line_num
        line_num -= 1
    for line in doc.split('\n'):
        line = strip_comment_line(line)
        try:
            lineno = line_numbers[line]
        except KeyError:
            continue
        yield (lineno, line)


class CheckStatus(enum.IntEnum):
    Ok = 0
    OkThrows = 1
    InputNotUnpackable = 2
    PreFalse = 3
    PreFail = 4
    RunFail = 5
    PostFail = 6
    PostFalse = 7

    def is_ok(self):
        return self in (CheckStatus.Ok, CheckStatus.OkThrows)

    def is_failure(self):
        return self in (CheckStatus.RunFail, CheckStatus.PostFail,
                        CheckStatus.PostFalse)


CheckResult = Tuple[CheckStatus, Any, BaseException]
CheckFn = Callable[..., CheckResult]


class Condition:
    def __init__(self, fn: Callable, checker: CheckFn, z3expr,
                 throws: Set[str], src_info: Tuple[str, int]) -> None:
        (self.fn, self.checker, self.z3expr, self.throws, self.src_info) = (
            fn, checker, z3expr, throws, src_info)

    def has_args(self):
        return bool(inspect.signature(self.fn).parameters)

    def unpack_args(self, buf: bytearray) -> Tuple[List[object],
                                                   Mapping[str, object]]:
        return unpack_signature(signature(self.fn), make_reader(buf))

    def check_buffer(self, buf: bytearray) -> CheckResult:
        try:
            args, kwargs = self.unpack_args(buf)
        except InputNotUnpackableError as e:
            return CheckStatus.InputNotUnpackable, None, None
        return self.checker(*args, **kwargs)

    def fails_on_args(self, args: List[object],
                      kwargs: Mapping[str, object]) -> str:
        status, result, exc = self.checker(*args, **kwargs)
        if status == CheckStatus.RunFail:
            msg = type(exc).__name__ + ': ' + str(exc)
        elif status == CheckStatus.PostFail:
            msg = (type(exc).__name__ + ': ' + str(exc) +
                   ' while checking result (' + repr(result) + ')')
        elif status == CheckStatus.PostFalse:
            msg = 'Not true on result (' + repr(result) + ')'
        else:
            return None
        msg += ' when calling ' + self.format_call(args, kwargs)
        return msg

    def format_call(self, args: List[object],
                    kwargs: Mapping[str, object]) -> str:
        boundargs = signature(self.fn).bind(*args, **kwargs)
        argdescs = (list(map(repr, cast(tuple, boundargs.args))) +
                    [k + '=' + repr(v) for k, v in
                     boundargs.kwargs.items()])  # type: ignore
        return self.fn.__name__ + '(' + ', '.join(argdescs) + ')'

    def simplify_args(self, input_args: List[object],
                      input_kwargs: Mapping[str, object]
                      ) -> Tuple[List[object], Mapping[str, object]]:
        '''
        post: len(return[0]) == len(input_args)
        post: len(return[1]) == len(input_kwargs)
        '''
        args = input_args[:]
        kwargs = dict(input_kwargs)
        any_change = True
        while any_change:
            any_change = False
            for i, a in enumerate(args):
                for candidate in simplify_value(a):
                    spliced_args = args[:i] + [candidate] + args[i + 1:]
                    if self.fails_on_args(spliced_args, kwargs):
                        args[i] = candidate
                        any_change = True
                        break
            for k, v in kwargs.items():
                for candidate in simplify_value(v):
                    if self.fails_on_args(args, {**kwargs, k: candidate}):
                        kwargs[k] = candidate
                        any_change = True
                        break
        return args, kwargs


_ALONE_RETURN = re.compile(r'\breturn\b')


class Z3Transformer(PureNodeTransformer):
    def visit_IfExp(self, node):
        return astcall(ast.Name(id='__z3__.If'), self.visit(node.test),
                       self.visit(node.body), self.visit(node.orelse))

    # def visit_Call(self, node):

    def visit_BoolOp(self, node):
        if type(node.op) == ast.And:
            z3fn = 'And'
        elif type(node.op) == ast.Or:
            z3fn = 'Or'
        else:
            raise Exception()
        args = [self.visit(v) for v in node.values]
        return astcall(ast.Name(id='__z3__.' + z3fn), *args)

    # def visit_Compare(self, node):
    #     if len(node.ops) == 1 and type(node.ops[0]) == ast.Eq:
    #         return astcall(ast.Name(id='__z3__.Eq'), node.left,
    #                        node.comparators[0])
    #     else:
    #         print(unparse(node))
    #        raise Exception()
    # def visit_NameConstant(self, node):
    #     return node.value


def make_z3_expr(fn: Callable[..., Any], pre: List[str], post: str) -> object:
    '''
    >>> def foo(x: int) -> int:
    ...   return 2 * x + x

    Solving the z3 expression generates counterexamples:
    >>> z3.solve(make_z3_expr(foo, ['x > 0'], 'return == 3'))
    [x = 2]

    Or "no solution", if the post condition logically follows:
    >>> z3.solve(make_z3_expr(foo, ['x > 0', 'x < 2'], 'return == 3'))
    no solution

    None is returned if the expression cannot be modeled in z3:
    >>> make_z3_expr(lambda a:a.foobar(), ['x > 0'], 'return == 3')
    '''
    sig = inspect.signature(fn)
    try:
        z3env = fn.__globals__.copy()  # type: ignore
        z3env.update({p.name: make_z3_var(p.annotation, p.name)
                      for p in sig.parameters.values()})
    except ExpressionNotSmtable as e:
        debug('ExpressionNotSmtable:', e)
        return None
    z3env['__z3__'] = z3

    fnast = cast(ast.FunctionDef, astparse(inspect.getsource(fn)))
    last_stmt = fnast.body[-1]
    if isinstance(last_stmt, ast.Return):
        fnast.body[-1] = ast.Assign(targets=[ast.Name(id='__return__')],
                                    value=last_stmt.value)
    precondition = exprparse(' and '.join(pre) or 'True')
    post = _ALONE_RETURN.sub('__return__', post)
    postcondition = exprparse(post)
    full_ast_expr = exprparse('(__pre__) and __z3__.Not(__post__)')
    full_expr = apply_ast_template(full_ast_expr, __pre__=precondition,
                                   __post__=postcondition)
    fnast.body.append(ast.Assign(targets=[ast.Name(id='__return__')],
                                 value=full_expr))
    fnast = cast(ast.FunctionDef, Z3Transformer().visit(fnast))
    debug(unparse(fnast))
    codeobj = compile(unparse(fnast.body), '<string>', 'exec')
    try:
        exec(codeobj, z3env)
    except Exception:
        debug(traceback.format_exc())
        return None

    return z3env['__return__']


def make_checker(fn: Callable[..., Any], pre: List[str], post: str,
                 throws: Set[str]) -> CheckFn:
    sig = inspect.signature(fn)
    # strip annotations (not required, and can mess with eval below)
    sig = inspect.Signature([p.replace(annotation=None)
                             for p in sig.parameters.values()],
                            return_annotation=None)

    precondition = ' and '.join('(' + p + ')' for p in pre) or 'True'
    name = fn.__qualname__
    args = ','.join([p.name for p in sig.parameters.values()])
    call = '{}({})'.format(name, args)
    post = _ALONE_RETURN.sub('__return__', post)
    body = '''def c{}:
  pre_satisfied = False
  try:
    pre_satisfied = {}
  except BaseException as e:
    return __chkstatus__.PreFail, None, e
  if not pre_satisfied:
    return __chkstatus__.PreFalse, None, None
  try:
    __return__ = {}
  except BaseException as e:
    return __chkstatus__.RunFail, None, e
  try:
    _post_result = {}
  except BaseException as e:
    return __chkstatus__.PostFail, __return__, e
  return ((__chkstatus__.Ok if _post_result else __chkstatus__.PostFalse),
          __return__, None)'''.format(sig, precondition, call, post)
    print(body)
    gbls = fn.__globals__.copy()  # type: ignore
    gbls[fn.__name__] = fn
    gbls['__chkstatus__'] = CheckStatus
    lcls: Mapping[str, object] = {}
    exec(body, gbls, lcls)
    checker = cast(Callable[..., None], lcls['c'])

    def check_wrapper(*a, **kw):
        try:
            return checker(*a, **kw)
        except BaseException as e:
            if type(e).__name__ in throws:
                return (CheckStatus.OkThrows, None, None)
            raise e

    return check_wrapper


def get_conditions(fn: Callable) -> List[Condition]:
    '''
    Searches for crosshair directives in the docstring for the
    given callable. Returns a Condition object for each postcondition.
    '''
    lines = list(get_doc_lines(fn))
    pre = []
    throws: Set[str] = set()
    for _line_num, line in lines:
        if line.startswith('pre:'):
            pre.append(line[len('pre:'):].strip())
        if line.startswith('throws:'):
            for ex in line[len('throws:'):].split(','):
                throws.add(ex.strip())
    conditions = []
    for line_num, line in lines:
        if line.startswith('post:'):
            post = line[len('post:'):].strip()
            src_info = (inspect.getsourcefile(fn), line_num)
            checker = make_checker(fn, pre, post, throws)
            z3expr = make_z3_expr(fn, pre, post)
            conditions.append(Condition(fn, checker, z3expr, throws, src_info))
    return conditions
