import ast
import builtins
import collections
import functools
import inspect
import re
import sys
import types
from dataclasses import dataclass, replace
from typing import *

from crosshair.util import debug
from crosshair.util import memo
from crosshair.util import source_position


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
        lines, line_num = inspect.getsourcelines(thing)  # type:ignore
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

class ImpliesTransformer(ast.NodeTransformer):
    '''
    pre- and post- conditions commonly want an implies(X, Y) operation.
    But it's important to only evaluate Y when X is true; so we rewrite
    this function into "Y if X else True"
    '''
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'implies':
            if len(node.args) != 2:
                raise SyntaxError('implies() must have exactly two arguments')
            condition, implication = node.args
            pos = {'lineno': node.lineno, 'col_offset': node.col_offset}
            return ast.IfExp(condition, implication, ast.Constant(True, **pos), **pos)
        return node

def compile_expr(src: str) -> types.CodeType:
    parsed = ast.parse(src, '<string>', 'eval')
    parsed = ImpliesTransformer().visit(parsed)
    return compile(parsed, '<string>', 'eval')

@dataclass()
class ConditionSyntaxMessage:
    filename: str
    line_num: int
    message: str


@dataclass
class ConditionExpr():
    evaluate: Optional[Callable[[Mapping[str, object]], object]]
    filename: str
    line: int
    expr_source: str
    addl_context: str = ''
    compile_err: Optional[ConditionSyntaxMessage] = None

    def __repr__(self):
        return f'ConditionExpr(filename={self.filename!r}, '\
            f'line={self.line!r}, '\
            f'expr_source={self.expr_source!r}, '\
            f'addl_context={self.addl_context!r}, '\
            f'compile_err={self.compile_err!r})'


@dataclass(frozen=True)
class Conditions:
    '''
    Describes the contract of a function.
    '''

    fn: Callable
    ''' The body of the function to analyze. '''

    pre: List[ConditionExpr]
    ''' The preconditions of the function. '''

    post: List[ConditionExpr]
    ''' The postconditions of the function. '''

    raises: FrozenSet[Type[BaseException]]
    '''
    A set of expection types that are expected.
    Subtypes of expected exceptions are also considered to be expected.
    CrossHair will attempt to report when this function raises an
    unexpected exception.
    '''

    sig: inspect.Signature
    '''
    The signature of the funtion. Argument and return type
    annotations should be resolved to real python types when possible.
    '''

    mutable_args: Optional[FrozenSet[str]]
    '''
    A set of arguments that are deeply immutable.
    When None, no assertion about mutability is provided.
    OTOH, an empty set asserts that the function does not mutate any argument.
    '''

    fn_syntax_messages: List[ConditionSyntaxMessage]
    '''
    A list of errors resulting from the parsing of the contract.
    In general, conditions should not be checked when such messages exist.
    '''

    def has_any(self) -> bool:
        return bool(self.pre or self.post)

    def syntax_messages(self) -> Iterator[ConditionSyntaxMessage]:
        for cond in (self.pre + self.post):
            if cond.compile_err is not None:
                yield cond.compile_err
        yield from self.fn_syntax_messages


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
        debug('WARNING: inconsistent signatures',
              sub_conditions.sig, super_conditions.sig)

    pre = sub_conditions.pre if sub_conditions.pre else super_conditions.pre
    post = super_conditions.post + sub_conditions.post
    raises = sub_conditions.raises | super_conditions.raises
    mutable_args = (sub_conditions.mutable_args
                    if sub_conditions.mutable_args is not None
                    else super_conditions.mutable_args)
    return Conditions(sub_conditions.fn,
                      pre,
                      post,
                      raises,
                      sub_conditions.sig,
                      mutable_args,
                      sub_conditions.fn_syntax_messages)


def merge_class_conditions(class_conditions: List[ClassConditions]) -> ClassConditions:
    inv: List[ConditionExpr] = []
    methods: Dict[str, Conditions] = {}
    # reverse because mro searches left side first
    for class_condition in reversed(class_conditions):
        inv.extend(class_condition.inv)
        methods.update(class_condition.methods)
    return ClassConditions(inv, methods)


def fn_globals(fn: Callable) -> Dict[str, object]:
    if hasattr(fn, '__wrapped__'):
        return fn_globals(fn.__wrapped__)  # type: ignore
    if inspect.isfunction(fn):  # excludes built-ins, which don't have closurevars
        closure_vars = inspect.getclosurevars(fn)
        if closure_vars.nonlocals:
            return {**closure_vars.nonlocals, **getattr(fn, '__globals__', {})}
    if hasattr(fn, '__globals__'):
        return fn.__globals__  # type:ignore
    return builtins.__dict__


def resolve_signature(fn: Callable) -> Tuple[Optional[inspect.Signature], Optional[ConditionSyntaxMessage]]:
    '''
    Get signature and resolve type annotations with get_type_hints.
    Returns a pair of Nones if no signature is available for the function.
    (e.g. it's implemented in C)
    Returns an unresolved signature and an error message if the type resultion errors.
    (e.g. the annotation references a type name that isn't dfined)
    '''
    # TODO: Test resolution with members at multiple places in the hierarchy.
    # e.g. https://bugs.python.org/issue29966
    try:
        sig = inspect.signature(fn)
    except ValueError:
        return (None, None)
    try:
        type_hints = get_type_hints(fn, fn_globals(fn))
    except NameError as name_error:
        filename, lineno = source_position(fn)
        return (sig, ConditionSyntaxMessage(filename, lineno, str(name_error)))
    params = sig.parameters.values()
    newparams = []
    for name, param in sig.parameters.items():
        if name in type_hints:
            param = param.replace(annotation=type_hints[name])
        newparams.append(param)
    newreturn = type_hints.get('return', sig.return_annotation)
    return (inspect.Signature(newparams, return_annotation=newreturn), None)

def set_first_arg_type(sig: inspect.Signature, first_arg_type: type) -> inspect.Signature:
    newparams = list(sig.parameters.values())
    newparams[0] = newparams[0].replace(annotation=first_arg_type)
    return inspect.Signature(newparams, return_annotation=sig.return_annotation)

_HEADER_LINE = re.compile(
    r'^(\s*)((?:post)|(?:pre)|(?:raises)|(?:inv))(?:\[([\w\s\,\.]*)\])?\:\:?\s*(.*?)\s*$')
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
                        parse.sections[section].append(
                            (line_num, match.groups()[1]))
                    # Still in the current section; continue:
                    continue
            cur_section = None
        match = _HEADER_LINE.match(line)
        if match:
            indentstr, section, bracketed, inline_expr = match.groups()
            if section not in sections:
                continue
            if bracketed is not None:
                if section != 'post':
                    parse.syntax_messages.append(ConditionSyntaxMessage(
                        filename, line_num, f'brackets not allowed in {section} section'))
                    continue
                if parse.mutable_expr is not None:
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

class ConditionParser:
    def get_fn_conditions(self, fn: Callable, first_arg_type: Optional[type] = None) -> Optional[Conditions]:
        raise NotImplemented
    def get_class_conditions(self, cls: type) -> ClassConditions:
        raise NotImplemented

class CompositeConditionParser(ConditionParser):
    def __init__(self):
        self.parsers = []
        self.class_cache: Dict[type, ClassConditions] = {}

    def get_fn_conditions(self, fn: Callable, first_arg_type: Optional[type] = None) -> Optional[Conditions]:
        for parser in self.parsers:
            conditions = parser.get_fn_conditions(fn, first_arg_type)
            if conditions is not None:
                return conditions
        return None

    def get_class_conditions(self, cls: type) -> ClassConditions:
        cached_ret = self.class_cache.get(cls)
        if cached_ret is not None:
            return cached_ret
        ret = ClassConditions([], {})
        for parser in self.parsers:
            conditions = parser.get_class_conditions(cls)
            if conditions.has_any():
                ret = conditions
                break
        self.class_cache[cls] = ret
        return ret

def condition_from_source_text(
        filename: str, line: int, expr_source: str,
        namespace: Dict[str, object], addl_context: str = '') -> ConditionExpr:
    evaluate, compile_err = None, None
    try:
        compiled = compile_expr(expr_source)
        def evaluatefn(bindings: Mapping[str, object]) -> object:
            return eval(compiled, {**namespace, **bindings})
        evaluate = evaluatefn
    except:
        e = sys.exc_info()[1]
        compile_err = ConditionSyntaxMessage(filename, line, str(e))
    return ConditionExpr(
        filename = filename,
        line = line,
        expr_source = expr_source,
        addl_context = addl_context,
        evaluate = evaluate,
        compile_err = compile_err,
    )

class Pep316Parser(ConditionParser):
    def __init__(self, global_parser: ConditionParser = None):
        if global_parser is None:
            global_parser = self
        self._global_parser = global_parser

    def get_fn_conditions(self, fn: Callable, first_arg_type: Optional[type] = None) -> Optional[Conditions]:
        filename, first_line = source_position(fn)
        sig, resolution_err = resolve_signature(fn)
        if sig is None:
            return None
        if resolution_err:
            return Conditions(fn, [], [], frozenset(), sig, None, [resolution_err])
        if first_arg_type:
            sig = set_first_arg_type(sig, first_arg_type)
        if isinstance(fn, types.BuiltinFunctionType):
            return Conditions(fn, [], [], frozenset(), sig, frozenset(), [])
        lines = list(get_doc_lines(fn))
        parse = parse_sections(lines, ('pre', 'post', 'raises'), filename)
        pre: List[ConditionExpr] = []
        raises: Set[Type[BaseException]] = set()
        post_conditions: List[ConditionExpr] = []
        mutable_args: Optional[FrozenSet[str]] = None
        if parse.mutable_expr is not None:
            mutable_args = frozenset(expr.strip().split('.')[0]
                                     for expr in parse.mutable_expr.split(',')
                                     if expr != '')
        for line_num, expr in parse.sections['pre']:
            pre.append(condition_from_source_text(filename, line_num, expr, fn_globals(fn)))
        for line_num, expr in parse.sections['raises']:
            if '#' in expr:
                expr = expr.split('#')[0]
            for exc_source in expr.split(','):
                try:
                    exc_type = eval(exc_source)
                except:
                    e = sys.exc_info()[1]
                    parse.syntax_messages.append(ConditionSyntaxMessage(
                        filename, line_num, str(e)))
                    continue
                if not issubclass(exc_type, BaseException):
                    parse.syntax_messages.append(ConditionSyntaxMessage(
                        filename, line_num, f'"{exc_type}" is not an exception class'))
                    continue
                raises.add(exc_type)
        for line_num, expr in parse.sections['post']:
            post_conditions.append(condition_from_source_text(filename, line_num, expr, fn_globals(fn)))

        return Conditions(fn, pre, post_conditions, frozenset(raises), sig,
                          mutable_args, parse.syntax_messages)

    def get_class_conditions(self, cls: type) -> ClassConditions:
        global_parser = self._global_parser
        try:
            filename = inspect.getsourcefile(cls)
        except TypeError:  # raises TypeError for builtins
            filename = None
        if filename is None:
            return ClassConditions([], {})
        namespace = sys.modules[cls.__module__].__dict__

        super_conditions = merge_class_conditions(
            [global_parser.get_class_conditions(base) for base in cls.__bases__])
        super_methods = super_conditions.methods
        inv = super_conditions.inv[:]
        parse = parse_sections(list(get_doc_lines(cls)), ('inv',), filename)
        for line_num, line in parse.sections['inv']:
            inv.append(condition_from_source_text(filename, line_num, line, namespace))

        methods = {}
        method_names = set(cls.__dict__.keys()) | super_methods.keys()
        for method_name in method_names:
            method = cls.__dict__.get(method_name, None)
            super_method_conditions = super_methods.get(method_name)
            if super_method_conditions is not None:
                revised_sig = set_first_arg_type(super_method_conditions.sig, cls)
                super_method_conditions = replace(super_method_conditions, sig=revised_sig)
            if method is None:
                if super_method_conditions is None:
                    continue
                else:
                    conditions: Conditions = super_method_conditions
            else:
                if inspect.isfunction(method):
                    parsed_conditions = global_parser.get_fn_conditions(method, first_arg_type=cls)
                elif isinstance(method, classmethod):
                    method = method.__get__(cls).__func__ # type: ignore
                    parsed_conditions = global_parser.get_fn_conditions(method, first_arg_type=type(cls))
                elif isinstance(method, staticmethod):
                    parsed_conditions = global_parser.get_fn_conditions(method.__get__(cls), first_arg_type=None)
                else:
                    #debug('Skipping unhandled member type ', type(method), ': ', method_name)
                    continue

                if parsed_conditions is None:
                    debug('Skipping ', str(method),
                          ': Unable to determine the function signature.')
                    continue
                if super_method_conditions is None:
                    conditions = parsed_conditions
                else:
                    conditions = merge_fn_conditions(
                        parsed_conditions, super_method_conditions)
            context_string = 'when calling ' + method_name
            local_inv = []
            for cond in inv:
                local_inv.append(condition_from_source_text(
                    cond.filename, cond.line, cond.expr_source, namespace, context_string))

            if method_name == '__new__':
                # invariants don't apply (__new__ isn't passed a concrete instance)
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
                conditions.post.extend(local_inv)
            if conditions.has_any():
                methods[method_name] = conditions

        return ClassConditions(inv, methods)


class IcontractParser(ConditionParser):
    def __init__(self, global_parser: ConditionParser = None):
        import icontract
        self.icontract = icontract
        if global_parser is None:
            global_parser = self
        self._global_parser = global_parser

    def get_fn_conditions(self, fn: Callable, first_arg_type: Optional[type] = None) -> Optional[Conditions]:
        icontract = self.icontract
        checker = icontract._checkers.find_checker(func=fn)  # type: ignore
        if checker is None:
            return None
        contractless_fn = fn.__wrapped__  # type: ignore
        sig, resolution_err = resolve_signature(fn)
        if sig is None:
            return None
        if resolution_err:
            return Conditions(contractless_fn, [], [], frozenset(), sig, None, [resolution_err])
        if first_arg_type:
            sig = set_first_arg_type(sig, first_arg_type)

        pre: List[ConditionExpr] = []
        post: List[ConditionExpr] = []

        def eval_contract(contract, kwargs):
            condition_kwargs = icontract._checkers.select_condition_kwargs(
                contract=contract, resolved_kwargs=kwargs)
            return contract.condition(**condition_kwargs)

        disjunction = checker.__preconditions__
        if len(disjunction) == 0:
            pass
        elif len(disjunction) == 1:
            for contract in disjunction[0]:
                evalfn = functools.partial(eval_contract, contract)
                filename, line_num = source_position(contract.condition)
                pre.append(ConditionExpr(evalfn, filename, line_num, ''))
        else:
            def eval_disjunction(disjunction, kwargs) -> bool:
                for conjunction in disjunction:
                    ok = True
                    for contract in conjunction:
                        if not eval_contract(contract, kwargs):
                            ok = False
                            break
                    if ok:
                        return True
                return False
            evalfn = functools.partial(eval_disjunction, disjunction)
            filename, line_num = source_position(contractless_fn)
            pre.append(ConditionExpr(evalfn, filename, line_num, ''))

        # TODO handle snapshots
        #snapshots = checker.__postcondition_snapshots__  # type: ignore

        def post_eval(contract, kwargs):
            _old = kwargs.pop('__old__')
            kwargs['result'] = kwargs.pop('__return__')
            del kwargs['_']
            condition_kwargs = icontract._checkers.select_condition_kwargs(
                contract=contract, resolved_kwargs=kwargs)
            return contract.condition(**condition_kwargs)
        for postcondition in checker.__postconditions__:
            evalfn = functools.partial(post_eval, postcondition)
            filename, line_num = source_position(postcondition.condition)
            post.append(ConditionExpr(evalfn, filename, line_num, ''))
        return Conditions(contractless_fn,
                          pre,
                          post,
                          raises=frozenset((AttributeError, IndexError,)),  # TODO all exceptions are OK?
                          sig=sig,
                          mutable_args=None,
                          fn_syntax_messages=[])

    def get_class_conditions(self, cls: type) -> ClassConditions:
        global_parser = self._global_parser
        methods = {}
        method_names = set(cls.__dict__.keys())
        for method_name in method_names:
            method = cls.__dict__.get(method_name, None)
            if inspect.isfunction(method):
                parsed_conditions = global_parser.get_fn_conditions(method, first_arg_type=cls)
            elif isinstance(method, classmethod):
                method = method.__get__(cls).__func__ # type: ignore
                parsed_conditions = global_parser.get_fn_conditions(method, first_arg_type=type(cls))
            elif isinstance(method, staticmethod):
                parsed_conditions = global_parser.get_fn_conditions(method.__get__(cls), first_arg_type=None)
            else:
                #debug('Skipping unhandled member type ', type(method), ': ', method_name)
                continue

            if parsed_conditions is None:
                debug('Skipping ', str(method),
                      ': Unable to determine the function signature.')
                continue
            if parsed_conditions.has_any():
                methods[method_name] = parsed_conditions
        return ClassConditions([], methods)
