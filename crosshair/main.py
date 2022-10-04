import argparse
import collections
import enum
import linecache
import os.path
import random
import shutil
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import (
    Any,
    Callable,
    Counter,
    Dict,
    Iterable,
    List,
    MutableMapping,
    NoReturn,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
    cast,
)

from crosshair import env_info
from crosshair.auditwall import disable_auditwall, engage_auditwall
from crosshair.core_and_libs import (
    AnalysisMessage,
    MessageType,
    analyze_any,
    installed_plugins,
    run_checkables,
)
from crosshair.diff_behavior import diff_behavior
from crosshair.fnutil import FunctionInfo, load_files_or_qualnames
from crosshair.options import (
    DEFAULT_OPTIONS,
    AnalysisKind,
    AnalysisOptions,
    AnalysisOptionSet,
    option_set_from_dict,
)
from crosshair.path_cover import (
    CoverageType,
    output_argument_dictionary_paths,
    output_eval_exression_paths,
    output_pytest_paths,
    path_cover,
)
from crosshair.pure_importer import prefer_pure_python_imports
from crosshair.register_contract import REGISTERED_CONTRACTS
from crosshair.statespace import NotDeterministic
from crosshair.util import ErrorDuringImport, add_to_pypath, debug, in_debug, set_debug
from crosshair.watcher import Watcher

create_lsp_server: Any = None
try:
    from crosshair.lsp_server import create_lsp_server
except ImportError:
    pass


class ExampleOutputFormat(enum.Enum):
    ARGUMENT_DICTIONARY = "ARGUMENT_DICTIONARY"
    EVAL_EXPRESSION = "EVAL_EXPRESSION"
    PYTEST = "PYTEST"


def analysis_kind(argstr: str) -> Sequence[AnalysisKind]:
    try:
        ret = [AnalysisKind[part.strip()] for part in argstr.split(",")]
    except KeyError:
        raise ValueError
    if AnalysisKind.hypothesis in ret:
        try:
            import hypothesis

            if hypothesis.__version_info__ < (6, 0, 0):
                raise Exception("CrossHair requires hypothesis version >= 6.0.0")
        except ImportError as e:
            raise Exception("Unable to import the hypothesis library") from e
    return ret


def command_line_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.RawTextHelpFormatter
    )
    common.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Output additional debugging information on stderr",
    )
    common.add_argument(
        "--extra_plugin",
        type=str,
        nargs="+",
        help="Plugin file(s) you wish to use during the current execution",
    )
    parser = argparse.ArgumentParser(
        prog="crosshair", description="CrossHair Analysis Tool"
    )
    subparsers = parser.add_subparsers(help="sub-command help", dest="action")
    check_parser = subparsers.add_parser(
        "check",
        help="Analyze a file or function",
        parents=[common],
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(
            """\
        The check command looks for counterexamples that break contracts.

        It outputs machine-readable messages in this format on stdout:
            <filename>:<line number>: error: <error message>

        It exits with one of the following codes:
            0 : No counterexamples are found
            1 : Counterexample(s) have been found
            2 : Other error
        """
        ),
    )
    check_parser.add_argument(
        "--report_all",
        action="store_true",
        help="Output analysis results for all postconditions (not just failing ones)",
    )
    check_parser.add_argument(
        "--report_verbose",
        dest="report_verbose",
        action="store_true",
        help="Output context and stack traces for counterexamples",
    )
    check_parser.add_argument(
        "target",
        metavar="TARGET",
        type=str,
        nargs="+",
        help=textwrap.dedent(
            """\
        A fully qualified module, class, or function, or
        a directory (which will be recursively analyzed), or
        a file path with an optional ":<line-number>" suffix.
        See https://crosshair.readthedocs.io/en/latest/contracts.html#targeting
        """
        ),
    )
    watch_parser = subparsers.add_parser(
        "watch",
        help="Continuously watch and analyze a directory",
        parents=[common],
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(
            """\
        The watch command continuously looks for contract counterexamples.
        Type Ctrl-C to stop this command.
        """
        ),
    )
    watch_parser.add_argument(
        "directory",
        metavar="TARGET",
        type=str,
        nargs="+",
        help=textwrap.dedent(
            """\
        File or directory to watch. Directories will be recursively analyzed.
        See https://crosshair.readthedocs.io/en/latest/contracts.html#targeting
        """
        ),
    )
    diffbehavior_parser = subparsers.add_parser(
        "diffbehavior",
        formatter_class=argparse.RawTextHelpFormatter,
        help="Find differences in the behavior of two functions",
        description=textwrap.dedent(
            """\
        Find differences in the behavior of two functions.
        See https://crosshair.readthedocs.io/en/latest/diff_behavior.html
            """
        ),
        parents=[common],
    )
    diffbehavior_parser.add_argument(
        "fn1",
        metavar="FUNCTION1",
        type=str,
        help='first fully-qualified function to compare (e.g. "mymodule.myfunc")',
    )
    diffbehavior_parser.add_argument(
        "fn2",
        metavar="FUNCTION2",
        type=str,
        help="second fully-qualified function to compare",
    )
    cover_parser = subparsers.add_parser(
        "cover",
        formatter_class=argparse.RawTextHelpFormatter,
        help="Generate inputs for a function, attempting to exercise different code paths",
        description=textwrap.dedent(
            """\
        Generates inputs to a function, hopefully getting good line, branch, and path
        coverage.
        See https://crosshair.readthedocs.io/en/latest/cover.html
            """
        ),
        parents=[common],
    )
    cover_parser.add_argument(
        "fn",
        metavar="FUNCTION",
        type=str,
        help='A fully-qualified function to explore (e.g. "mymodule.myfunc")',
    )
    cover_parser.add_argument(
        "--example_output_format",
        type=lambda e: ExampleOutputFormat[e.upper()],  # type: ignore
        choices=ExampleOutputFormat.__members__.values(),
        metavar="FORMAT",
        default=ExampleOutputFormat.EVAL_EXPRESSION,
        help=textwrap.dedent(
            """\
        Determines how to output examples.
            argument_dictionary : Output arguments as repr'd, ordered dictionaries
            eval_expression     : Output examples as expressions, suitable for eval()
            pytest              : Output examples as stub pytest tests
        """
        ),
    )
    cover_parser.add_argument(
        "--coverage_type",
        type=lambda e: CoverageType[e.upper()],  # type: ignore
        choices=CoverageType.__members__.values(),
        metavar="TYPE",
        default=CoverageType.OPCODE,
        help=textwrap.dedent(
            """\
        Determines what kind of coverage to achieve.
            opcode : Cover as many opcodes of the function as possible.
                     This is similar to "branch" coverage.
            path   : Cover any possible execution path.
                     There will usually be an infinite number of paths (e.g. loops are
                     effectively unrolled). Use max_iterations and/or
                     per_condition_timeout to bound results.
                     Many path decisions are internal to CrossHair, so you may see
                     more duplicative-ness in the output than you'd expect.
        """
        ),
    )
    for subparser in (check_parser, diffbehavior_parser, cover_parser):
        subparser.add_argument(
            "--per_path_timeout",
            type=float,
            metavar="FLOAT",
            help=textwrap.dedent(
                """\
            Maximum seconds to spend checking one execution path.
            If unspecified, CrossHair will timeout each path at the square root of the
            `per_condition_timeout`.
            """
            ),
        )
        subparser.add_argument(
            "--per_condition_timeout",
            type=float,
            metavar="FLOAT",
            help="Maximum seconds to spend checking execution paths for one condition",
        )
    lsp_server_parser = subparsers.add_parser(
        "server",
        help="Start a server, speaking the Language Server Protocol",
        parents=[common],
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(
            f"""\
            Many IDEs support the Language Server Protocol (LSP).
            CrossHair can produce various results and analysis through LSP.
            {'IMPORTANT: `pip install pygls` to enable this feature'
             if create_lsp_server is None else ''}
            """
        ),
    )

    for subparser in (check_parser, watch_parser, lsp_server_parser):
        subparser.add_argument(
            "--analysis_kind",
            type=analysis_kind,
            metavar="KIND",
            help=textwrap.dedent(
                """\
            Kind of contract to check.
            By default, the PEP316, deal, and icontract kinds are all checked.
            Multiple kinds (comma-separated) may be given.
            See https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html
                asserts    : check assert statements
                PEP316     : check PEP316 contracts (docstring-based)
                icontract  : check icontract contracts (decorator-based)
                deal       : check deal contracts (decorator-based)
                hypothesis : check hypothesis tests
            """
            ),
        )
    return parser


def run_watch_loop(
    watcher: Watcher,
    max_watch_iterations: int = sys.maxsize,
    term_lines_rewritable: bool = True,
) -> None:
    restart = True
    stats: Counter[str] = Counter()
    active_messages: Dict[Tuple[str, int], AnalysisMessage]
    for _ in range(max_watch_iterations):
        if restart:
            clear_screen()
            print_divider("-")
            line = f"  Analyzing {len(watcher._modtimes)} files."
            print(color(line, AnsiColor.OKBLUE), end="")
            max_condition_timeout = 0.5
            restart = False
            stats = Counter()
            active_messages = {}
        else:
            time.sleep(0.1)
            max_condition_timeout *= 2
        for curstats, messages in watcher.run_iteration(max_condition_timeout):
            messages = [m for m in messages if m.state > MessageType.PRE_UNSAT]
            stats.update(curstats)
            if messages_merged(active_messages, messages):
                linecache.checkcache()
                clear_screen()
                options = DEFAULT_OPTIONS.overlay(watcher._options)
                for message in active_messages.values():
                    lines = long_describe_message(message, options)
                    if lines is None:
                        continue
                    print_divider("-")
                    print(lines, end="")
                print_divider("-")
            else:
                if term_lines_rewritable:
                    print("\r", end="")
                else:
                    print(".", end="")
                    continue
            num_files = len(watcher._modtimes)
            if len(watcher._paths) > 1:
                loc_desc = f"{num_files} files"
            else:
                path_parts = Path(next(iter(watcher._paths))).parts
                path_desc = path_parts[-1] if path_parts else "."
                if num_files > 1:
                    loc_desc = f'"{path_desc}" ({num_files} files)'
                else:
                    loc_desc = f'"{path_desc}"'
            if term_lines_rewritable:
                line = f'  Analyzed {stats["num_paths"]} paths in {loc_desc}.       '
            else:
                line = f"  Analyzing paths in {loc_desc}: "
            print(color(line, AnsiColor.OKBLUE), end="")
            if watcher._change_flag:
                watcher._change_flag = False
                restart = True
                line = f"  Restarting analysis over {len(watcher._modtimes)} files."
                print(color(line, AnsiColor.OKBLUE), end="")


def clear_screen():
    # Print enough newlines to fill the screen:
    print("\n" * shutil.get_terminal_size().lines, end="")


def print_divider(ch=" "):
    try:
        cols = os.get_terminal_size().columns - 1
    except OSError:
        cols = 5
    print(ch * cols)


class AnsiColor(enum.Enum):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def color(text: str, *effects: AnsiColor) -> str:
    return "".join(e.value for e in effects) + text + AnsiColor.ENDC.value


def messages_merged(
    messages: MutableMapping[Tuple[str, int], AnalysisMessage],
    new_messages: Iterable[AnalysisMessage],
) -> bool:
    any_change = False
    for message in new_messages:
        key = (message.filename, message.line)
        if key not in messages:
            messages[key] = message
            any_change = True
    return any_change


_MOTD = [
    "Did I miss a counterexample? Let me know: https://github.com/pschanely/CrossHair/issues/new",
    "Help me be faster! Add to my benchmark suite: https://github.com/pschanely/crosshair-benchmark",
    "Share your CrossHair experience on YouTube, Twitter, your blog, ... even TikTok.",
    "Questions? Ask at https://github.com/pschanely/CrossHair/discussions/new?category=q-a",
    "Sign up for CrossHair updates at https://pschanely.github.io",
    # Use CrossHair? We'd like to reference your work here: ...
]


def watch(
    args: argparse.Namespace,
    options: AnalysisOptionSet,
    max_watch_iterations=sys.maxsize,
) -> int:
    if not args.directory:
        print("No files or directories given to watch", file=sys.stderr)
        return 2
    try:
        paths = [Path(d) for d in args.directory]
        watcher = Watcher(paths, options)
        watcher.check_changed()

        # Some terminals don't interpret \r correctly; we detect them here:
        term_lines_rewritable = "THONNY_USER_DIR" not in os.environ

        run_watch_loop(
            watcher, max_watch_iterations, term_lines_rewritable=term_lines_rewritable
        )
    except KeyboardInterrupt:
        pass
    watcher._pool.terminate()
    print()
    if random.uniform(0.0, 1.0) > 0.4:
        motd = "I enjoyed working with you today!"
    else:
        motd = random.choice(_MOTD)
    print(motd)
    return 0


def format_src_context(filename: str, lineno: int) -> str:
    amount = 3
    line_numbers = range(max(1, lineno - amount), lineno + amount + 1)
    output = [f"{filename}:{lineno}:\n"]
    for curline in line_numbers:
        text = linecache.getline(filename, curline)
        if text == "":  # (actual empty lines have a newline)
            continue
        output.append(
            ">" + color(text, AnsiColor.WARNING) if lineno == curline else "|" + text
        )
    return "".join(output)


def describe_message(
    message: AnalysisMessage, options: AnalysisOptions
) -> Optional[str]:
    if options.report_verbose:
        return long_describe_message(message, options)
    else:
        return short_describe_message(message, options)


def long_describe_message(
    message: AnalysisMessage, options: AnalysisOptions
) -> Optional[str]:
    tb, desc, state = message.traceback, message.message, message.state
    desc = desc.replace(" when ", "\nwhen ")
    context = format_src_context(message.filename, message.line)
    intro = ""
    if not options.report_all:
        if message.state <= MessageType.PRE_UNSAT:  # type: ignore
            return None
    if state == MessageType.CONFIRMED:
        intro = "I was able to confirm your postcondition over all paths."
    elif state == MessageType.CANNOT_CONFIRM:
        intro = "I wasn't able to find a counterexample."
    elif message.state == MessageType.PRE_UNSAT:
        intro = "I am having trouble finding any inputs that meet your preconditions."
    elif message.state == MessageType.POST_ERR:
        intro = "I got an error while checking your postcondition."
    elif message.state == MessageType.EXEC_ERR:
        intro = "I found an exception while running your function."
    elif message.state == MessageType.POST_FAIL:
        intro = "I was able to make your postcondition return False."
    elif message.state == MessageType.SYNTAX_ERR:
        intro = "One of your conditions isn't a valid python expression."
    elif message.state == MessageType.IMPORT_ERR:
        intro = "I couldn't import a file."
    if message.state <= MessageType.CANNOT_CONFIRM:  # type: ignore
        intro = color(intro, AnsiColor.OKGREEN)
    else:
        intro = color(intro, AnsiColor.FAIL)
    return f"{tb}\n{intro}\n{context}\n{desc}\n"


def short_describe_message(
    message: AnalysisMessage, options: AnalysisOptions
) -> Optional[str]:
    desc = message.message
    if message.state <= MessageType.PRE_UNSAT:  # type: ignore
        if options.report_all:
            return "{}:{}: {}: {}".format(message.filename, message.line, "info", desc)
        return None
    if message.state == MessageType.POST_ERR:
        desc = "Error while evaluating post condition: " + desc
    return "{}:{}: {}: {}".format(message.filename, message.line, "error", desc)


def checked_load(qualname: str, stderr: TextIO) -> Optional[FunctionInfo]:
    try:
        objs = list(load_files_or_qualnames([qualname]))
    except ErrorDuringImport as exc:
        cause = exc.__cause__ if exc.__cause__ is not None else exc
        print(
            f'Unable to load "{qualname}": {type(cause).__name__}: {cause}',
            file=stderr,
        )
        return None
    obj = objs[0]
    if not isinstance(obj, FunctionInfo):
        print(f'"{qualname}" does not target a function.', file=stderr)
        return None
    if obj.get_callable() is None:
        print(f'Cannot determine signature of "{qualname}"', file=stderr)
        return None
    return obj


def diffbehavior(
    args: argparse.Namespace, options: AnalysisOptions, stdout: TextIO, stderr: TextIO
) -> int:
    (fn_name1, fn_name2) = (args.fn1, args.fn2)
    fn1 = checked_load(fn_name1, stderr)
    fn2 = checked_load(fn_name2, stderr)
    if fn1 is None or fn2 is None:
        return 2
    options.stats = collections.Counter()
    diffs = diff_behavior(fn1, fn2, options)
    debug("stats", options.stats)
    if isinstance(diffs, str):
        print(diffs, file=stderr)
        return 2
    elif len(diffs) == 0:
        num_paths = options.stats["num_paths"]
        exhausted = options.stats["exhaustion"] > 0
        stdout.write(f"No differences found. (attempted {num_paths} iterations)\n")
        if exhausted:
            stdout.write("All paths exhausted, functions are likely the same!\n")
        else:
            stdout.write(
                "Consider trying longer with: --per_condition_timeout=<seconds>\n"
            )
        return 0
    else:
        width = max(len(fn_name1), len(fn_name2)) + 2
        for diff in diffs:
            inputs = ", ".join(f"{k}={v}" for k, v in diff.args.items())
            stdout.write(f"Given: ({inputs}),\n")
            result1, result2 = diff.result1, diff.result2
            differing_args = result1.get_differing_arg_mutations(result2)
            stdout.write(
                f"{fn_name1.rjust(width)} : {result1.describe(differing_args)}\n"
            )
            stdout.write(
                f"{fn_name2.rjust(width)} : {result2.describe(differing_args)}\n"
            )
        return 1


def cover(
    args: argparse.Namespace, options: AnalysisOptions, stdout: TextIO, stderr: TextIO
) -> int:
    ctxfn = checked_load(args.fn, stderr)
    if ctxfn is None:
        return 2
    options.stats = collections.Counter()
    try:
        paths = path_cover(ctxfn, options, args.coverage_type)
    except NotDeterministic:
        print("Repeated executions are not behaving deterministically.", file=stderr)
        if not in_debug():
            print("Re-run in verbose mode for debugging information.", file=stderr)
        return 2

    fn, _ = ctxfn.callable()
    example_output_format = args.example_output_format
    if example_output_format == ExampleOutputFormat.ARGUMENT_DICTIONARY:
        return output_argument_dictionary_paths(fn, paths, stdout, stderr)
    elif example_output_format == ExampleOutputFormat.EVAL_EXPRESSION:
        return output_eval_exression_paths(fn, paths, stdout, stderr)
    if example_output_format == ExampleOutputFormat.PYTEST:
        return output_pytest_paths(fn, paths, stdout, stderr)
    assert False


def server(
    args: argparse.Namespace, options: AnalysisOptionSet, stdout: TextIO, stderr: TextIO
) -> NoReturn:
    if create_lsp_server is None:
        print(
            "Install the `pygls` module from PyPI to enable the LSP server; exiting.",
            file=sys.stderr,
        )
        sys.exit(2)
    cast(Callable[[], NoReturn], create_lsp_server(options).start_io)()


def check(
    args: argparse.Namespace, options: AnalysisOptionSet, stdout: TextIO, stderr: TextIO
) -> int:
    any_problems = False
    try:
        entities = list(load_files_or_qualnames(args.target))
    except FileNotFoundError as exc:
        print(f'File not found: "{exc.args[0]}"', file=stderr)
        return 2
    except ErrorDuringImport as exc:
        cause = exc.__cause__ if exc.__cause__ is not None else exc
        print(f"Could not import your code:\n", file=stderr)
        traceback.print_exception(type(cause), cause, cause.__traceback__, file=stderr)
        return 2
    full_options = DEFAULT_OPTIONS.overlay(report_verbose=False).overlay(options)
    for entity in entities:
        debug("Check ", getattr(entity, "__name__", str(entity)))
        for message in run_checkables(analyze_any(entity, options)):
            line = describe_message(message, full_options)
            if line is None:
                continue
            stdout.write(line + "\n")
            debug("Traceback for output message:\n", message.traceback)
            if message.state > MessageType.PRE_UNSAT:
                any_problems = True
    return 1 if any_problems else 0


def unwalled_main(cmd_args: Union[List[str], argparse.Namespace]) -> int:
    parser = command_line_parser()
    if isinstance(cmd_args, argparse.Namespace):
        args = cmd_args
    else:
        args = parser.parse_args(cmd_args)
    if not args.action:
        parser.print_help(sys.stderr)
        return 2
    set_debug(args.verbose)
    if in_debug():
        debug(env_info())
        debug("Installed plugins:", installed_plugins)
    options = option_set_from_dict(args.__dict__)
    # fall back to current directory to look up modules
    path_additions = [""] if sys.path and sys.path[0] != "" else []
    with add_to_pypath(*path_additions), prefer_pure_python_imports():
        if args.extra_plugin:
            for plugin in args.extra_plugin:
                exec(Path(plugin).read_text())
            if len(REGISTERED_CONTRACTS):
                debug(
                    f"Registered {len(REGISTERED_CONTRACTS)} contract(s) "
                    f"from: {args.extra_plugin}"
                )
        if args.action == "check":
            return check(args, options, sys.stdout, sys.stderr)
        elif args.action == "diffbehavior":
            defaults = DEFAULT_OPTIONS.overlay(
                AnalysisOptionSet(
                    per_condition_timeout=3.0,
                    per_path_timeout=30.0,  # mostly, we don't want to time out paths
                )
            )
            return diffbehavior(args, defaults.overlay(options), sys.stdout, sys.stderr)
        elif args.action == "cover":
            defaults = DEFAULT_OPTIONS.overlay(
                AnalysisOptionSet(
                    per_condition_timeout=3.0,
                    per_path_timeout=30.0,  # mostly, we don't want to time out paths
                )
            )
            return cover(args, defaults.overlay(options), sys.stdout, sys.stderr)
        elif args.action == "watch":
            disable_auditwall()  # (we'll engage auditwall in the workers)
            return watch(args, options)
        elif args.action == "server":
            disable_auditwall()  # (we'll engage auditwall in the workers)
            server(args, options, sys.stdout, sys.stderr)
        else:
            print(f'Unknown action: "{args.action}"', file=sys.stderr)
            return 2


def mypy_and_check(cmd_args: Optional[List[str]] = None) -> None:
    if cmd_args is None:
        cmd_args = sys.argv[1:]
    cmd_args = ["check"] + cmd_args
    check_args, mypy_args = command_line_parser().parse_known_args(cmd_args)
    set_debug(check_args.verbose)
    mypy_cmd_args = mypy_args + check_args.target
    debug("Running mypy with the following arguments:", " ".join(mypy_cmd_args))
    try:
        from mypy import api
    except ModuleNotFoundError:
        print("Unable to find mypy; skipping", file=sys.stderr)
    else:
        _mypy_out, mypy_err, mypy_ret = api.run(mypy_cmd_args)
        print(mypy_err, file=sys.stderr)
        if mypy_ret != 0:
            sys.exit(mypy_ret)
    engage_auditwall()
    debug("Running crosshair with these args:", check_args)
    sys.exit(unwalled_main(check_args))


def main(cmd_args: Optional[List[str]] = None) -> None:
    if cmd_args is None:
        cmd_args = sys.argv[1:]
    engage_auditwall()
    sys.exit(unwalled_main(cmd_args))


if __name__ == "__main__":
    main()
