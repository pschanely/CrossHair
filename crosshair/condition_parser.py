import builtins
import inspect
import re
import types
from dataclasses import dataclass
from typing import *

from crosshair.util import CrosshairInternal


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


@dataclass(frozen=True)
class ConditionExpr():
    expr: types.CodeType
    filename: str
    line: int
    addl_context: str
    expr_source: str = ''

@dataclass(frozen=True)
class Conditions():
    pre: List[ConditionExpr]
    post: List[ConditionExpr]
    raises: Set[str]
    sig: inspect.Signature
    mutable_args: Set[str]
    def has_any(self) -> bool:
        return bool(self.pre or self.post)

@dataclass(frozen=True)
class ClassConditions():
    inv: List[ConditionExpr]
    methods: List[Tuple[Callable, Conditions]]
    def has_any(self) -> bool:
        return bool(self.inv) or any(c.has_any() for m,c in self.methods)

def compile_expr(expr:str) -> types.CodeType:
    return compile(expr, '<string>', 'eval')

def fn_globals(fn:Callable) -> Dict[str, object]:
    if hasattr(fn, '__wrapped__'):
        return fn_globals(fn.__wrapped__)  # type: ignore
    elif hasattr(fn, '__globals__'):
        return fn.__globals__ # type:ignore
    else:
        return builtins.__dict__

def resolve_signature(fn:Callable, self_type:Optional[type]=None) -> Optional[inspect.Signature]:
    ''' Resolve type annotations with get_type_hints, and adds a type for self. '''
    try:
        sig = inspect.signature(fn)
    except ValueError:
        return None
    type_hints = get_type_hints(fn, fn_globals(fn))
    params = sig.parameters.values()
    if (self_type and
        len(params) > 0 and
        next(iter(params)).name == 'self' and
        'self' not in type_hints):
        type_hints['self'] = self_type
    newparams = []
    for name, param in sig.parameters.items():
        if name in type_hints:
            param = param.replace(annotation=type_hints[name])
        newparams.append(param)
    newreturn = type_hints.get('return', sig.return_annotation)
    return inspect.Signature(newparams, return_annotation=newreturn)


_ALONE_RETURN = re.compile(r'\breturn\b')
def sub_return_as_var(expr_string):
    return _ALONE_RETURN.sub('__return__', expr_string)
    
_POST_LINE = re.compile(r'^\s*post(?:\[([\w\s\,\.]*)\])?\:(.*)$')

def get_fn_conditions(fn: Callable, self_type:Optional[type] = None) -> Conditions:
    sig = resolve_signature(fn, self_type=self_type)
    if sig is None:
        raise CrosshairInternal('Unable to determine signature of function: ' + str(fn))
    if isinstance(fn, types.BuiltinFunctionType):
        return Conditions([], [], set(), sig, set())
    filename = inspect.getsourcefile(fn)
    lines = list(get_doc_lines(fn))
    pre = []
    raises: Set[str] = set()
    for line_num, line in lines:
        if line.startswith('pre:'):
            src = line[len('pre:'):].strip()
            expr = compile_expr(src)
            pre.append(ConditionExpr(expr, filename, line_num, '', src))
        if line.startswith('raises:'):
            for ex in line[len('raises:'):].split(','):
                raises.add(ex.strip())
    post_conditions = []
    mutable_args = set()
    for line_num, line in lines:
        match = _POST_LINE.match(line)
        if match:
            (cur_mutable, expr_string) = match.groups()
            if cur_mutable is not None:
                for m in cur_mutable.split(','):
                    mutable_args.add(m.strip())
            src = expr_string.strip()
            post = compile_expr(sub_return_as_var(src))
            post_conditions.append(ConditionExpr(post, filename, line_num, '', src))
    return Conditions(pre, post_conditions, raises, sig, mutable_args)

def get_class_conditions(cls: type) -> ClassConditions:
    try:
        filename = inspect.getsourcefile(cls)
    except TypeError: # raises TypeError for builtins
        return ClassConditions([], [])
    lines = list(get_doc_lines(cls))
    inv = []
    for line_num, line in lines:
        if line.startswith('inv:'):
            src = line[len('inv:'):].strip()
            expr = compile_expr(src)
            inv.append(ConditionExpr(expr, filename, line_num, '', src))

    methods = []
    for method_name, method in cls.__dict__.items():
        if not inspect.isfunction(method):
            continue
        conditions = get_fn_conditions(method, self_type=cls)
        context_string = 'when calling ' + method_name
        local_inv = []
        for cond in inv:
            local_inv.append(ConditionExpr(cond.expr, cond.filename, cond.line, context_string, cond.expr_source))

        if method_name == '__new__':
            use_pre, use_post = False, False
        elif method_name == '__del__':
            use_pre, use_post = True, False
        elif method_name == '__init__':
            use_pre, use_post = False, True
        elif method_name.startswith('__') and method_name.endswith('__'):
            use_pre, use_post = True, True
        elif method_name.startswith('_'):
            use_pre, use_post = False, False
        else:
            use_pre, use_post = True, True
        if use_pre:
            conditions.pre.extend(local_inv)
        if use_post:
            if method_name == '__init__':
                conditions.mutable_args.add(
                    next(iter(inspect.signature(method).parameters.keys())))
            conditions.post.extend(local_inv)
        if conditions.has_any():
            methods.append((method, conditions))

    return ClassConditions(inv, methods)
