import builtins
import collections
import inspect
import re
import sys
import types
from dataclasses import dataclass, replace
from typing import *

from crosshair.util import debug, memo


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
    try:
        lines, line_num = inspect.getsourcelines(thing)
    except OSError:
        return
    line_num += len(lines) - 1
    line_numbers = {}
    for line in reversed(lines):
        line_numbers[strip_comment_line(line)] = line_num
        line_num -= 1
    for line in doc.split('\n'):
        l = strip_comment_line(line)
        try:
            lineno = line_numbers[l]
        except KeyError:
            continue
        yield (lineno, line)


@dataclass()
class ConditionSyntaxMessage:
    filename: str
    line_num: int
    message: str
        
@dataclass(init=False)
class ConditionExpr():
    expr: Optional[types.CodeType]
    filename: str
    line: int
    expr_source: str
    addl_context: str = ''
    compile_err: Optional[ConditionSyntaxMessage] = None
    def __init__(self, filename:str, line: int, expr_source: str, addl_context:str=''):
        self.filename = filename
        self.line = line
        self.expr_source = expr_source
        self.addl_context = addl_context
        self.expr = None
        try:
            self.expr = compile(expr_source, '<string>', 'eval')
        except:
            e = sys.exc_info()[1]
            self.compile_err = ConditionSyntaxMessage(filename, line, str(e))

@dataclass(frozen=True)
class Conditions():
    pre: List[ConditionExpr]
    post: List[ConditionExpr]
    raises: Set[str]
    sig: inspect.Signature
    mutable_args: Set[str]
    fn_syntax_messages: List[ConditionSyntaxMessage]
    def has_any(self) -> bool:
        return bool(self.pre or self.post)
    def syntax_messages(self) -> Iterator[ConditionSyntaxMessage]:
        for cond in (self.pre + self.post):
            if cond.compile_err is not None:
                yield cond.compile_err
        yield from self.fn_syntax_messages
    def compilable(self) -> 'Conditions':
        return replace(self,
                pre=[c for c in self.pre if c.expr is not None],
                post=[c for c in self.post if c.expr is not None],
        )

@dataclass(frozen=True)
class ClassConditions():
    inv: List[ConditionExpr]
    methods: Mapping[str, Conditions]
    def has_any(self) -> bool:
        return bool(self.inv) or any(c.has_any() for c in self.methods.values())

def merge_fn_conditions(sub_conditions: Conditions, super_conditions: Conditions) -> Conditions:
    
    # TODO: resolve the warning below:
    #   (1) the type of self always changes
    #   (2) paramter renames (or *a, **kws) could result in varied bindings
    if sub_conditions.sig is not None and sub_conditions.sig != super_conditions.sig:
        debug('WARNING: inconsistent signatures', sub_conditions.sig, super_conditions.sig)
    
    pre = sub_conditions.pre if sub_conditions.pre else super_conditions.pre
    post = super_conditions.post + sub_conditions.post
    raises = sub_conditions.raises | super_conditions.raises
    mutable_args = sub_conditions.mutable_args if sub_conditions.mutable_args else super_conditions.mutable_args
    return Conditions(pre,
                      post,
                      raises,
                      sub_conditions.sig,
                      mutable_args,
                      sub_conditions.fn_syntax_messages)
    
def merge_class_conditions(class_conditions: List[ClassConditions]) -> ClassConditions:
    inv: List[ConditionExpr] = []
    methods: Dict[str, Conditions] = {}
    for class_condition in reversed(class_conditions): # reverse because mro searches left side first
        inv.extend(class_condition.inv)
        methods.update(class_condition.methods)
    return ClassConditions(inv, methods)
    
    
def fn_globals(fn:Callable) -> Dict[str, object]:
    if hasattr(fn, '__wrapped__'):
        return fn_globals(fn.__wrapped__)  # type: ignore
    if inspect.isfunction(fn): # excludes built-ins, which don't have closurevars
        closure_vars = inspect.getclosurevars(fn)
        if closure_vars.nonlocals:
            return {**closure_vars.nonlocals, **closure_vars.globals}
    if hasattr(fn, '__globals__'):
        return fn.__globals__ # type:ignore
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


_HEADER_LINE = re.compile(r'^(\s*)((?:post)|(?:pre)|(?:raises)|(?:inv))(?:\[([\w\s\,\.]*)\])?\:\:?\s*(.*?)\s*$')
_SECTION_LINE = re.compile(r'^(\s*)(.*?)\s*$')

@dataclass(init=False)
class SectionParse:
    syntax_messages: List[ConditionSyntaxMessage]
    sections: Dict[str, List[Tuple[int, str]]]
    mutable_expr: Optional[str] = None
    def __init__(self):
        self.sections = collections.defaultdict(list)
        self.syntax_messages = []

def has_expr(line: str) -> bool:
    line = line.strip()
    return bool(line) and not line.startswith('#')

def parse_sections(lines: List[Tuple[int, str]], sections: Tuple[str, ...], filename: str) -> SectionParse:
    parse = SectionParse()
    cur_section: Optional[Tuple[str, int]] = None
    for line_num, line in lines:
        if line.strip() == '':
            continue
        if cur_section:
            section, indent = cur_section
            match = _SECTION_LINE.match(line)
            if match:
                this_indent = len(match.groups()[0])
                if this_indent > indent:
                    if has_expr(match.groups()[1]):
                        parse.sections[section].append((line_num, match.groups()[1]))
                    # Still in the current section; continue:
                    continue
            cur_section = None
        match = _HEADER_LINE.match(line)
        if match:
            indentstr, section, bracketed, inline_expr = match.groups()
            if section not in sections:
                continue
            if bracketed:
                if section != 'post':
                    parse.syntax_messages.append(ConditionSyntaxMessage(
                        filename, line_num, f'brackets not allowed in {section} section'))
                    continue
                if parse.mutable_expr:
                    parse.syntax_messages.append(ConditionSyntaxMessage(
                        filename, line_num, f'duplicate post section'))
                    continue
                else:
                    parse.mutable_expr = bracketed
            if has_expr(inline_expr):
                parse.sections[section].append((line_num, inline_expr))
                continue
            else:
                cur_section = (section, len(indentstr))
    return parse

    

def get_fn_conditions(fn: Callable, self_type:Optional[type] = None) -> Optional[Conditions]:
    sig = resolve_signature(fn, self_type=self_type)
    if sig is None:
        return None
    if isinstance(fn, types.BuiltinFunctionType):
        return Conditions([], [], set(), sig, set(), [])
    filename = inspect.getsourcefile(fn)
    lines = list(get_doc_lines(fn))
    parse = parse_sections(lines, ('pre', 'post', 'raises'), filename)
    pre = []
    raises: Set[str] = set()
    post_conditions = []
    mutable_args: Set[str] = set()
    if parse.mutable_expr:
        for expr in parse.mutable_expr.split(','):
            mutable_args.add(expr.strip().split('.')[0])
    
    for line_num, expr in parse.sections['pre']:
        pre.append(ConditionExpr(filename, line_num, expr))
    for line_num, expr in parse.sections['raises']:
        raises.add(expr)
    for line_num, expr in parse.sections['post']:
        post_conditions.append(ConditionExpr(filename, line_num, expr))

    return Conditions(pre, post_conditions, raises, sig, mutable_args, parse.syntax_messages)

@memo
def get_class_conditions(cls: type) -> ClassConditions:
    try:
        filename = inspect.getsourcefile(cls)
    except TypeError: # raises TypeError for builtins
        return ClassConditions([], {})
    
    super_conditions = merge_class_conditions([get_class_conditions(base) for base in cls.__bases__])
    super_methods = super_conditions.methods
    inv = super_conditions.inv[:]
    parse = parse_sections(list(get_doc_lines(cls)), ('inv',), filename)
    for line_num, line in parse.sections['inv']:
        inv.append(ConditionExpr(filename, line_num, line))

    methods = {}
    for method_name, method in cls.__dict__.items():
        if not inspect.isfunction(method):
            continue
        conditions = get_fn_conditions(method, self_type=cls)
        if conditions is None:
            debug('Skipping ', str(method), ': Unable to determine the function signature.')
            continue
        if method_name in super_methods:
            conditions = merge_fn_conditions(conditions, super_methods[method_name])
        context_string = 'when calling ' + method_name
        local_inv = []
        for cond in inv:
            local_inv.append(ConditionExpr(cond.filename, cond.line, cond.expr_source, context_string))

        if method_name == '__new__':
            use_pre, use_post = False, False # invariants don't apply (__new__ isn't passed a concrete instance)
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
            methods[method_name] = conditions

    return ClassConditions(inv, methods)
