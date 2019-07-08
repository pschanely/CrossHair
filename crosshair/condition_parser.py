import inspect
import re
import types
from dataclasses import dataclass
from typing import *

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

@dataclass(frozen=True)
class Conditions():
    pre: List[ConditionExpr]
    post: List[ConditionExpr]
    raises: Set[str]
    def has_any(self):
        return self.pre or self.post

@dataclass(frozen=True)
class ClassConditions():
    inv: List[ConditionExpr]

def compile_expr(expr:str) -> types.CodeType:
    return compile(expr, '<string>', 'eval')

_ALONE_RETURN = re.compile(r'\breturn\b')
def sub_return_as_var(expr_string):
    return _ALONE_RETURN.sub('__return__', expr_string)
    
def get_fn_conditions(fn: Callable) -> Conditions:
    filename = inspect.getsourcefile(fn)
    lines = list(get_doc_lines(fn))
    pre = []
    raises: Set[str] = set()
    for line_num, line in lines:
        if line.startswith('pre:'):
            expr = compile_expr(line[len('pre:'):].strip())
            pre.append(ConditionExpr(expr, filename, line_num, ''))
        if line.startswith('raises:'):
            for ex in line[len('raises:'):].split(','):
                raises.add(ex.strip())
    post_conditions = []
    for line_num, line in lines:
        if line.startswith('post:'):
            post = compile_expr(sub_return_as_var(line[len('post:'):].strip()))
            post_conditions.append(ConditionExpr(post, filename, line_num, ''))
    return Conditions(pre, post_conditions, raises)

def get_class_conditions(cls: type) -> ClassConditions:
    filename = inspect.getsourcefile(cls)
    lines = list(get_doc_lines(cls))
    inv = []
    for line_num, line in lines:
        if line.startswith('inv:'):
            expr = compile_expr(line[len('inv:'):].strip())
            inv.append(ConditionExpr(expr, filename, line_num, ''))
    return ClassConditions(inv)
