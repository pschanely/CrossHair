import ast
import collections
import contextlib
import enum
from functools import partial
from functools import wraps
import inspect
from inspect import BoundArguments
import re
import sys
import textwrap
import traceback
import types
from dataclasses import dataclass
from dataclasses import replace
from typing import *

try:
    import icontract
except ModuleNotFoundError:
    icontract = None  # type: ignore


try:
    import hypothesis
    from hypothesis import strategies as st
    from hypothesis.internal.conjecture.data import ConjectureData
except ModuleNotFoundError:
    hypothesis = None  # type: ignore

from crosshair.fnutil import fn_globals
from crosshair.fnutil import set_first_arg_type
from crosshair.fnutil import FunctionInfo
from crosshair.options import AnalysisKind
from crosshair.util import CrosshairInternal, IgnoreAttempt, UnexploredPath
from crosshair.util import debug
from crosshair.util import eval_friendly_repr
from crosshair.util import frame_summary_for_fn
from crosshair.util import is_pure_python
from crosshair.util import sourcelines
from crosshair.util import DynamicScopeVar


class ConditionExprType(enum.Enum):
    INVARIANT = "invariant"
    PRECONDIITON = "precondition"
    POSTCONDIITON = "postcondition"

    def __str__(self):
        return self.value


# For convience
INVARIANT = ConditionExprType.INVARIANT
PRECONDIITON = ConditionExprType.PRECONDIITON
POSTCONDIITON = ConditionExprType.POSTCONDIITON


class NoEnforce:
    """
    Signal to suppress contract enforcement.

    This function wrapper does nothing on its own. But the enforcement tracer
    looks for it and will skip conditions on `fn` when this wrapper is detected.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *a, **kw) -> object:
        return self.fn(*a, **kw)


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
    _filename, line_num, lines = sourcelines(thing)  # type:ignore
    line_num += len(lines) - 1
    line_numbers = {}
    for line in reversed(lines):
        line_numbers[strip_comment_line(line)] = line_num
        line_num -= 1
    for line in doc.split("\n"):
        l = strip_comment_line(line)
        try:
            lineno = line_numbers[l]
        except KeyError:
            continue
        yield (lineno, line)


class ImpliesTransformer(ast.NodeTransformer):
    """
    Transform AST to rewrite implies operation.

    Pre- and post-conditions commonly want an implies(X, Y) operation.
    But it's important to only evaluate Y when X is true; so we rewrite
    this function into "Y if X else True"
    """

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "implies":
            if len(node.args) != 2:
                raise SyntaxError("implies() must have exactly two arguments")
            condition, implication = node.args
            pos = {"lineno": node.lineno, "col_offset": node.col_offset}
            return ast.IfExp(condition, implication, ast.Constant(True, **pos), **pos)
        return node


def compile_expr(src: str) -> types.CodeType:
    parsed = ast.parse(src, "<string>", "eval")
    parsed = ImpliesTransformer().visit(parsed)
    return compile(parsed, "<string>", "eval")


_NO_RETURN = object()
UNABLE_TO_REPR = "<unable to repr>"


def default_counterexample(
    fn_name: str,
    bound_args: BoundArguments,
    return_val: object = _NO_RETURN,
) -> str:
    with eval_friendly_repr():
        call_desc = ""
        if return_val is not _NO_RETURN:
            try:
                repr_str = repr(return_val)
            except Exception as e:
                if isinstance(e, (IgnoreAttempt, UnexploredPath)):
                    raise
                debug(
                    f"Exception attempting to repr function output: ",
                    traceback.format_exc(),
                )
                repr_str = UNABLE_TO_REPR
            if repr_str != "None":
                call_desc = call_desc + " (which returns " + repr_str + ")"
        messages: List[str] = []
        for argname, argval in list(bound_args.arguments.items()):
            try:
                repr_str = repr(argval)
            except Exception as e:
                if isinstance(e, (IgnoreAttempt, UnexploredPath)):
                    raise
                debug(
                    f'Exception attempting to repr input "{argname}": ',
                    traceback.format_exc(),
                )
                repr_str = UNABLE_TO_REPR
            messages.append(argname + " = " + repr_str)
        call_desc = fn_name + "(" + ", ".join(messages) + ")" + call_desc

        if messages:
            return "when calling " + call_desc
        else:
            return "for any input"


@dataclass()
class ConditionSyntaxMessage:
    filename: str
    line_num: int
    message: str


@dataclass
class ConditionExpr:
    condition_type: ConditionExprType
    evaluate: Optional[Callable[[Mapping[str, object]], bool]]
    filename: str
    line: int
    expr_source: str
    compile_err: Optional[ConditionSyntaxMessage] = None

    def __repr__(self):
        return (
            f"ConditionExpr(filename={self.filename!r}, "
            f"line={self.line!r}, "
            f"expr_source={self.expr_source!r}, "
            f"compile_err={self.compile_err!r})"
        )


@dataclass(frozen=True)
class Conditions:
    """Describe the contract of a function."""

    fn: Callable
    """
    The body of the function to analyze.
    Ideally, this is just the body of the function and does not include checking
    pre- or post-conditions. (though this is not always possible)
    """

    src_fn: Callable
    """
    The body of the function to use for error reporting. Usually the same as
    `fn`, but sometimes the original is wrapped in shell for exception handling
    or other reasons.
    """

    pre: List[ConditionExpr]
    """ The preconditions of the function. """

    post: List[ConditionExpr]
    """ The postconditions of the function. """

    raises: FrozenSet[Type[BaseException]]
    """
    A set of expection types that are expected.
    Subtypes of expected exceptions are also considered to be expected.
    CrossHair will attempt to report when this function raises an
    unexpected exception.
    """

    sig: inspect.Signature
    """
    The signature of the funtion. Argument and return type
    annotations should be resolved to real python types when possible.
    """

    # TODO: can mutation checking be implemented as just another kind of postcondition?
    mutable_args: Optional[FrozenSet[str]]
    """
    A set of arguments that are deeply immutable.
    When None, no assertion about mutability is provided.
    OTOH, an empty set asserts that the function does not mutate any argument.
    """

    fn_syntax_messages: List[ConditionSyntaxMessage]
    """
    A list of errors resulting from the parsing of the contract.
    In general, conditions should not be checked when such messages exist.
    """

    counterexample_description_maker: Optional[
        Callable[[BoundArguments, object], str]
    ] = None
    """
    An optional callback that formats a counterexample invocation as text.
    It takes the example arguments and the returned value
    (or the senitnel value _MISSING if function did not complete).
    """

    def has_any(self) -> bool:
        return bool(self.pre or self.post or self.fn_syntax_messages)

    def syntax_messages(self) -> Iterator[ConditionSyntaxMessage]:
        for cond in self.pre + self.post:
            if cond.compile_err is not None:
                yield cond.compile_err
        yield from self.fn_syntax_messages

    def format_counterexample(
        self, args: BoundArguments, return_val: object = _NO_RETURN
    ) -> str:
        if self.counterexample_description_maker is not None:
            return self.counterexample_description_maker(args, return_val)
        return default_counterexample(self.src_fn.__name__, args, return_val)


@dataclass(frozen=True)
class ClassConditions:
    inv: List[ConditionExpr]
    methods: Mapping[str, Conditions]

    def has_any(self) -> bool:
        return bool(self.inv) or any(c.has_any() for c in self.methods.values())


def add_completion_conditions(conditions: Conditions):
    """Assume a trivial postcondition of "True" when preconditions exist."""
    post = conditions.post
    if not post and conditions.pre:
        filename, line, _lines = sourcelines(conditions.src_fn)
        post.append(ConditionExpr(POSTCONDIITON, lambda vars: True, filename, line, ""))


def add_completion_conditions_to_class(class_conditions: ClassConditions):
    for method in class_conditions.methods.values():
        add_completion_conditions(method)


def merge_fn_conditions(
    sub_conditions: Conditions, super_conditions: Conditions
) -> Conditions:

    # TODO: resolve the warning below:
    #   (1) the type of self always changes
    #   (2) paramter renames (or *a, **kws) could result in varied bindings
    if sub_conditions.sig is not None and sub_conditions.sig != super_conditions.sig:
        debug(
            "WARNING: inconsistent signatures", sub_conditions.sig, super_conditions.sig
        )

    pre = sub_conditions.pre if sub_conditions.pre else super_conditions.pre
    post = super_conditions.post + sub_conditions.post
    raises = sub_conditions.raises | super_conditions.raises
    mutable_args = (
        sub_conditions.mutable_args
        if sub_conditions.mutable_args is not None
        else super_conditions.mutable_args
    )
    fn = sub_conditions.fn
    return Conditions(
        fn,
        fn,
        pre,
        post,
        raises,
        sub_conditions.sig,
        mutable_args,
        sub_conditions.fn_syntax_messages,
    )


def merge_class_conditions(class_conditions: List[ClassConditions]) -> ClassConditions:
    inv: List[ConditionExpr] = []
    methods: Dict[str, Conditions] = {}
    # reverse because mro searches left side first
    for class_condition in reversed(class_conditions):
        inv.extend(class_condition.inv)
        methods.update(class_condition.methods)
    return ClassConditions(inv, methods)


_HEADER_LINE = re.compile(
    r"""^(\s*)\:?  # whitespace with optional leading colon
         ((?:post)|(?:pre)|(?:raises)|(?:inv))  # noncapturing keywords
         (?:\[([\w\s\,\.]*)\])?  # optional params in square brackets
         \:\:?\s*  # single or double colons
         (.*?)  # The (non-greedy) content
         \s*$""",
    re.VERBOSE,
)
_SECTION_LINE = re.compile(r"^(\s*)(.*?)\s*$")


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
    return bool(line) and not line.startswith("#")


def parse_sections(
    lines: List[Tuple[int, str]], sections: Tuple[str, ...], filename: str
) -> SectionParse:
    parse = SectionParse()
    cur_section: Optional[Tuple[str, int]] = None
    for line_num, line in lines:
        if line.strip() == "":
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
            if bracketed is not None:
                if section != "post":
                    parse.syntax_messages.append(
                        ConditionSyntaxMessage(
                            filename,
                            line_num,
                            f"brackets not allowed in {section} section",
                        )
                    )
                    continue
                if parse.mutable_expr is not None:
                    parse.syntax_messages.append(
                        ConditionSyntaxMessage(
                            filename, line_num, f"duplicate post section"
                        )
                    )
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
    def get_fn_conditions(self, fn: FunctionInfo) -> Optional[Conditions]:
        raise NotImplementedError

    def get_class_conditions(self, cls: type) -> ClassConditions:
        raise NotImplementedError


class ConcreteConditionParser(ConditionParser):
    def __init__(self, toplevel_parser: ConditionParser = None):
        if toplevel_parser is None:
            toplevel_parser = self
        self._toplevel_parser = toplevel_parser

    def get_toplevel_parser(self):
        return self._toplevel_parser

    def get_class_invariants(self, cls: type) -> List[ConditionExpr]:
        raise NotImplementedError

    def get_class_conditions(self, cls: type) -> ClassConditions:
        if not is_pure_python(cls):
            # We can't get conditions/line numbers for classes written in C.
            return ClassConditions([], {})

        toplevel_parser = self.get_toplevel_parser()
        methods = {}
        super_conditions = merge_class_conditions(
            [toplevel_parser.get_class_conditions(base) for base in cls.__bases__]
        )
        inv = self.get_class_invariants(cls)
        super_methods = super_conditions.methods
        method_names = set(cls.__dict__.keys()) | super_methods.keys()
        for method_name in method_names:
            method = cls.__dict__.get(method_name, None)
            super_method_conditions = super_methods.get(method_name)
            if super_method_conditions is not None:
                revised_sig = set_first_arg_type(super_method_conditions.sig, cls)
                super_method_conditions = replace(
                    super_method_conditions, sig=revised_sig
                )
            if method is None:
                if super_method_conditions is None:
                    continue
                else:
                    conditions: Conditions = super_method_conditions
            else:
                parsed_conditions = toplevel_parser.get_fn_conditions(
                    FunctionInfo.from_class(cls, method_name)
                )
                if parsed_conditions is None:
                    # debug(f'Skipping "{method_name}": Unable to determine the function signature.')
                    continue
                if super_method_conditions is None:
                    conditions = parsed_conditions
                else:
                    conditions = merge_fn_conditions(
                        parsed_conditions, super_method_conditions
                    )
            if method_name in ("__new__", "__repr__"):
                # __new__ isn't passed a concrete instance.
                # __repr__ is itself required for reporting problems with invariants.
                use_pre, use_post = False, False
            elif method_name == "__del__":
                use_pre, use_post = True, False
            elif method_name == "__init__":
                use_pre, use_post = False, True
            elif method_name.startswith("__") and method_name.endswith("__"):
                use_pre, use_post = True, True
            elif method_name.startswith("_"):
                use_pre, use_post = False, False
            else:
                use_pre, use_post = True, True
            if use_pre:
                conditions.pre.extend(inv)
            if use_post:
                conditions.post.extend(inv)
            if conditions.has_any():
                methods[method_name] = conditions

        return ClassConditions(inv, methods)


class CompositeConditionParser(ConditionParser):
    def __init__(self):
        self.parsers = []
        self.class_cache: Dict[type, ClassConditions] = {}

    def get_toplevel_parser(self) -> ConditionParser:
        return self

    def get_fn_conditions(self, fn: FunctionInfo) -> Optional[Conditions]:
        # TODO: clarify ths distinction between None and empty Conditions.
        ret = None
        for parser in self.parsers:
            conditions = parser.get_fn_conditions(fn)
            if conditions is not None:
                ret = conditions
                if conditions.has_any():
                    break
        if ret:
            add_completion_conditions(ret)
        return ret

    def get_class_conditions(self, cls: type) -> ClassConditions:
        cached_ret = self.class_cache.get(cls)
        if cached_ret is not None:
            return cached_ret
        ret = ClassConditions([], {})
        if cls.__module__ == "typing":
            # Partly for performance, but also class condition computation fails for
            # some typing classes:
            return ret
        for parser in self.parsers:
            conditions = parser.get_class_conditions(cls)
            if conditions.has_any():
                ret = conditions
                break
        add_completion_conditions_to_class(ret)
        self.class_cache[cls] = ret
        return ret


def condition_from_source_text(
    condition_type: ConditionExprType,
    filename: str,
    line: int,
    expr_source: str,
    namespace: Dict[str, object],
) -> ConditionExpr:
    evaluate, compile_err = None, None
    try:
        compiled = compile_expr(expr_source)

        def evaluatefn(bindings: Mapping[str, object]) -> bool:
            return eval(compiled, {**namespace, **bindings})

        evaluate = evaluatefn
    except:
        e = sys.exc_info()[1]
        compile_err = ConditionSyntaxMessage(filename, line, str(e))
    return ConditionExpr(
        condition_type=condition_type,
        filename=filename,
        line=line,
        expr_source=expr_source,
        evaluate=evaluate,
        compile_err=compile_err,
    )


_RAISE_SPHINX_RE = re.compile(r"\:raises\s+(\w+)\:", re.MULTILINE)


def parse_sphinx_raises(fn: Callable) -> Set[Type[BaseException]]:
    raises: Set[Type[BaseException]] = set()
    doc = getattr(fn, "__doc__", None)
    if doc is None:
        return raises
    for excname in _RAISE_SPHINX_RE.findall(doc):
        try:
            exc_type = eval(excname, fn_globals(fn))
        except:
            continue
        if not isinstance(exc_type, type):
            continue
        if not issubclass(exc_type, BaseException):
            continue
        raises.add(exc_type)
    return raises


class Pep316Parser(ConcreteConditionParser):
    def get_fn_conditions(self, ctxfn: FunctionInfo) -> Optional[Conditions]:
        fn_and_sig = ctxfn.get_callable()
        if fn_and_sig is None:
            return None
        (fn, sig) = fn_and_sig
        filename, first_line, _lines = sourcelines(fn)
        if isinstance(fn, types.BuiltinFunctionType):
            return Conditions(fn, fn, [], [], frozenset(), sig, frozenset(), [])
        lines = list(get_doc_lines(fn))
        parse = parse_sections(lines, ("pre", "post", "raises"), filename)
        pre: List[ConditionExpr] = []
        raises: Set[Type[BaseException]] = set()
        post_conditions: List[ConditionExpr] = []
        mutable_args: Optional[FrozenSet[str]] = None
        if parse.mutable_expr is not None:
            mutable_args = frozenset(
                expr.strip().split(".")[0]
                for expr in parse.mutable_expr.split(",")
                if expr != ""
            )
        for line_num, expr in parse.sections["pre"]:
            pre.append(
                condition_from_source_text(
                    PRECONDIITON,
                    filename,
                    line_num,
                    expr,
                    fn_globals(fn),
                )
            )
        for line_num, expr in parse.sections["raises"]:
            if "#" in expr:
                expr = expr.split("#")[0]
            for exc_source in expr.split(","):
                try:
                    exc_type = eval(exc_source, fn_globals(fn))
                except:
                    e = sys.exc_info()[1]
                    parse.syntax_messages.append(
                        ConditionSyntaxMessage(filename, line_num, str(e))
                    )
                    continue
                if not issubclass(exc_type, BaseException):
                    parse.syntax_messages.append(
                        ConditionSyntaxMessage(
                            filename,
                            line_num,
                            f'"{exc_type}" is not an exception class',
                        )
                    )
                    continue
                raises.add(exc_type)
        for line_num, expr in parse.sections["post"]:
            post_conditions.append(
                condition_from_source_text(
                    POSTCONDIITON,
                    filename,
                    line_num,
                    expr,
                    fn_globals(fn),
                )
            )
        return Conditions(
            fn,
            fn,
            pre,
            post_conditions,
            frozenset(raises),
            sig,
            mutable_args,
            parse.syntax_messages,
        )

    def get_class_invariants(self, cls: type) -> List[ConditionExpr]:
        try:
            filename = inspect.getsourcefile(cls)
        except TypeError:  # raises TypeError for builtins
            filename = None
        if filename is None:
            return []
        namespace = sys.modules[cls.__module__].__dict__

        parse = parse_sections(list(get_doc_lines(cls)), ("inv",), filename)
        inv = []
        for line_num, line in parse.sections["inv"]:
            inv.append(
                condition_from_source_text(
                    INVARIANT, filename, line_num, line, namespace
                )
            )
        return inv


class IcontractParser(ConcreteConditionParser):
    def __init__(self, toplevel_parser: ConditionParser = None):
        super().__init__(toplevel_parser)

    def contract_text(self, contract) -> str:
        l = icontract._represent.inspect_lambda_condition(condition=contract.condition)
        return l.text if l else ""

    def get_fn_conditions(self, ctxfn: FunctionInfo) -> Optional[Conditions]:
        if icontract is None:
            return None
        fn_and_sig = ctxfn.get_callable()
        if fn_and_sig is None:
            return None
        (fn, sig) = fn_and_sig

        checker = icontract._checkers.find_checker(func=fn)  # type: ignore
        contractless_fn = fn  # type: ignore
        while (
            hasattr(contractless_fn, "__is_invariant_check__")
            or hasattr(contractless_fn, "__preconditions__")
            or hasattr(contractless_fn, "__postconditions__")
        ):
            contractless_fn = contractless_fn.__wrapped__  # type: ignore
        if checker is None:
            return Conditions(
                contractless_fn, contractless_fn, [], [], frozenset(), sig, None, []
            )

        pre: List[ConditionExpr] = []
        post: List[ConditionExpr] = []

        def eval_contract(contract, kwargs: Mapping) -> bool:
            condition_kwargs = icontract._checkers.select_condition_kwargs(
                contract=contract, resolved_kwargs=kwargs
            )
            return contract.condition(**condition_kwargs)

        disjunction = checker.__preconditions__  # type: ignore
        if len(disjunction) == 0:
            pass
        elif len(disjunction) == 1:
            for contract in disjunction[0]:
                evalfn = partial(eval_contract, contract)
                filename, line_num, _lines = sourcelines(contract.condition)
                pre.append(
                    ConditionExpr(
                        PRECONDIITON,
                        evalfn,
                        filename,
                        line_num,
                        self.contract_text(contract),
                    )
                )
        else:

            def eval_disjunction(disjunction, kwargs: Mapping) -> bool:
                for conjunction in disjunction:
                    ok = True
                    for contract in conjunction:
                        if not eval_contract(contract, kwargs):
                            ok = False
                            break
                    if ok:
                        return True
                return False

            evalfn = partial(eval_disjunction, disjunction)
            filename, line_num, _lines = sourcelines(contractless_fn)
            source = (
                "("
                + ") or (".join(
                    [
                        " and ".join([self.contract_text(c) for c in conj])
                        for conj in disjunction
                    ]
                )
                + ")"
            )
            pre.append(ConditionExpr(PRECONDIITON, evalfn, filename, line_num, source))

        snapshots = checker.__postcondition_snapshots__  # type: ignore

        def take_snapshots(**kwargs):
            old_as_mapping: MutableMapping[str, Any] = {}
            for snap in snapshots:
                snap_kwargs = icontract._checkers.select_capture_kwargs(
                    a_snapshot=snap, resolved_kwargs=kwargs
                )
                old_as_mapping[snap.name] = snap.capture(**snap_kwargs)
            return icontract._checkers.Old(mapping=old_as_mapping)

        def post_eval(contract, orig_kwargs: Mapping) -> bool:
            kwargs = dict(orig_kwargs)
            _old = kwargs.pop("__old__")
            kwargs["OLD"] = take_snapshots(**_old.__dict__)
            kwargs["result"] = kwargs.pop("__return__")
            del kwargs["_"]
            condition_kwargs = icontract._checkers.select_condition_kwargs(
                contract=contract, resolved_kwargs=kwargs
            )
            return contract.condition(**condition_kwargs)

        for postcondition in checker.__postconditions__:  # type: ignore
            evalfn = partial(post_eval, postcondition)
            filename, line_num, _lines = sourcelines(postcondition.condition)
            post.append(
                ConditionExpr(
                    POSTCONDIITON,
                    evalfn,
                    filename,
                    line_num,
                    self.contract_text(postcondition),
                )
            )
        return Conditions(
            contractless_fn,
            contractless_fn,
            pre,
            post,
            raises=frozenset(parse_sphinx_raises(fn)),
            sig=sig,
            mutable_args=None,
            fn_syntax_messages=[],
        )

    def get_class_invariants(self, cls: type) -> List[ConditionExpr]:
        invariants = getattr(cls, "__invariants__", ())  # type: ignore
        ret = []

        def inv_eval(contract, kwargs):
            return contract.condition(self=kwargs["self"])

        for contract in invariants:
            filename, line_num, _lines = sourcelines(contract.condition)
            ret.append(
                ConditionExpr(
                    INVARIANT,
                    partial(inv_eval, contract),
                    filename,
                    line_num,
                    self.contract_text(contract),
                )
            )
        return ret


class AssertsParser(ConcreteConditionParser):
    def __init__(self, toplevel_parser: ConditionParser = None):
        super().__init__(toplevel_parser)

    @staticmethod
    def is_string_literal(node: ast.AST) -> bool:
        if sys.version_info >= (3, 8):
            return (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        else:
            return isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)

    @staticmethod
    def get_first_body_line(fn: Callable) -> Optional[int]:
        """
        Retrieve the first line of the body of the function ``fn``.

        :return:
            the line number of the first non-assert statement in the given function.

        :return:
            None if the function does not start with at least one assert statement.
        """
        _filename, first_fn_lineno, lines = sourcelines(fn)
        if not lines:
            return None
        ast_module = ast.parse(textwrap.dedent("".join(lines)))
        ast_fn = ast_module.body[0]
        if not isinstance(ast_fn, ast.FunctionDef):
            return None
        found_any_preconditions = False
        for statement in ast_fn.body:
            if isinstance(statement, ast.Assert):
                found_any_preconditions = True
                continue
            elif AssertsParser.is_string_literal(statement):
                # A docstring, keep looking:
                continue
            break
        if found_any_preconditions:
            return first_fn_lineno + (statement.lineno - 1)
        else:
            return None

    def get_fn_conditions(self, ctxfn: FunctionInfo) -> Optional[Conditions]:
        fn_and_sig = ctxfn.get_callable()
        if fn_and_sig is None:
            return None
        (fn, sig) = fn_and_sig
        # TODO replace this guard with package-level configuration?
        if (
            getattr(fn, "__module__", False)
            and fn.__module__.startswith("crosshair.")
            and not fn.__module__.endswith("_test")
        ):
            return None
        try:
            first_body_line = AssertsParser.get_first_body_line(fn)
        except OSError:
            return None
        if first_body_line is None:
            return None

        filename, first_line, _lines = sourcelines(fn)

        @wraps(fn)
        def wrappedfn(*a, **kw):
            try:
                return NoEnforce(fn)(*a, **kw)
            except AssertionError as e:
                # TODO: check that this isn't failing at an early line in a different
                # file?
                _, lineno = frame_summary_for_fn(
                    fn, traceback.extract_tb(e.__traceback__)
                )
                if lineno >= first_body_line:
                    raise

        post = [
            ConditionExpr(
                POSTCONDIITON,
                lambda _: True,
                filename,
                first_line,
                "",
            )
        ]
        return Conditions(
            wrappedfn,
            fn,
            [],  # (pre)
            post,
            raises=frozenset(parse_sphinx_raises(fn)),
            sig=sig,
            mutable_args=None,
            fn_syntax_messages=[],
        )

    def get_class_invariants(self, cls: type) -> List[ConditionExpr]:
        return []


class HypothesisParser(ConcreteConditionParser):
    def __init__(self, toplevel_parser: ConditionParser = None):
        super().__init__(toplevel_parser)

    def _generate_args(self, payload: bytes, decorated_fn: Callable):
        given_kwargs = decorated_fn.hypothesis._given_kwargs  # type: ignore
        strategy = st.fixed_dictionaries(given_kwargs)
        return ConjectureData.for_buffer(payload).draw(strategy)

    def get_fn_conditions(self, ctxfn: FunctionInfo) -> Optional[Conditions]:
        fn_and_sig = ctxfn.get_callable()
        if fn_and_sig is None:
            return None
        (fn, sig) = fn_and_sig
        if not getattr(fn, "is_hypothesis_test", False):
            return None
        fuzz_one = getattr(getattr(fn, "hypothesis", None), "fuzz_one_input", None)
        if fuzz_one is None:
            return None

        filename, first_line, _lines = sourcelines(fn)
        post = [
            ConditionExpr(
                POSTCONDIITON,
                lambda _: True,
                filename,
                first_line,
                "",
            )
        ]
        sig = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "seed", inspect.Parameter.POSITIONAL_ONLY, annotation=bytes
                )
            ]
        )

        return Conditions(
            fuzz_one,
            fn,
            [],  # (pre)
            post,
            raises=frozenset(),
            sig=sig,
            mutable_args=None,
            fn_syntax_messages=[],
        )

    def get_class_invariants(self, cls: type) -> List[ConditionExpr]:
        return []


_PARSER_MAP = {
    AnalysisKind.PEP316: Pep316Parser,
    AnalysisKind.icontract: IcontractParser,
    AnalysisKind.asserts: AssertsParser,
    AnalysisKind.hypothesis: HypothesisParser,
}


# Condition parsers may be needed at various places in the stack.
# We configure them through the use of a magic threadlocal value:
_CALLTREE_PARSER = DynamicScopeVar(ConditionParser, "calltree parser")


def condition_parser(
    analysis_kinds: Sequence[AnalysisKind],
) -> ContextManager[ConditionParser]:
    current = _CALLTREE_PARSER.get_if_in_scope()
    if current is not None:
        return contextlib.nullcontext(current)
    debug("Using parsers: ", analysis_kinds)
    condition_parser = CompositeConditionParser()
    condition_parser.parsers.extend(
        _PARSER_MAP[k](condition_parser) for k in analysis_kinds
    )
    return _CALLTREE_PARSER.open(condition_parser)


def get_current_parser() -> ConditionParser:
    return _CALLTREE_PARSER.get()
