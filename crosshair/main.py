import argparse
import collections
import dataclasses
import enum
import heapq
import importlib
import importlib.util
import inspect
import linecache
import multiprocessing
import os
import os.path
from pathlib import Path
import re
import shutil
import sys
import textwrap
import time
import traceback
import types
from typing import *
from typing import TextIO

from crosshair.auditwall import engage_auditwall
from crosshair.auditwall import opened_auditwall
from crosshair.diff_behavior import diff_behavior
from crosshair.core_and_libs import analyze_any
from crosshair.core_and_libs import analyze_module
from crosshair.core_and_libs import run_checkables
from crosshair.core_and_libs import AnalysisMessage
from crosshair.core_and_libs import MessageType
from crosshair.fnutil import load_by_qualname
from crosshair.fnutil import load_files_or_qualnames
from crosshair.fnutil import FunctionInfo
from crosshair.fnutil import NotFound
from crosshair.options import option_set_from_dict
from crosshair.options import AnalysisKind
from crosshair.options import AnalysisOptionSet
from crosshair.options import AnalysisOptions
from crosshair.options import DEFAULT_OPTIONS
from crosshair.util import debug
from crosshair.util import extract_module_from_file
from crosshair.util import load_file
from crosshair.util import set_debug
from crosshair.util import CrosshairInternal
from crosshair.util import ErrorDuringImport
from crosshair.watcher import Watcher
import crosshair.core_and_libs


def analysis_kind(argstr: str) -> Sequence[AnalysisKind]:
    try:
        return [AnalysisKind[part.strip()] for part in argstr.split(",")]
    except KeyError:
        raise ValueError


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
        "--per_path_timeout",
        type=float,
        metavar="FLOAT",
        help="Maximum seconds to spend checking one execution path",
    )
    common.add_argument(
        "--per_condition_timeout",
        type=float,
        metavar="FLOAT",
        help="Maximum seconds to spend checking execution paths for one condition",
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
        See https://crosshair.readthedocs.io/en/latest/what_code_is_analyzed.html
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
        See https://crosshair.readthedocs.io/en/latest/what_code_is_analyzed.html
        """
        ),
    )
    for subparser in (check_parser, watch_parser):
        subparser.add_argument(
            "--analysis_kind",
            type=analysis_kind,
            metavar="KIND",
            default=(AnalysisKind.PEP316, AnalysisKind.icontract, AnalysisKind.asserts),
            help=textwrap.dedent(
                """\
            Kind of contract to check. By default, all kinds are checked.
            See https://crosshair.readthedocs.io/en/latest/kinds_of_contracts.html
                PEP316    : docstring-based contracts
                icontract : decorator-based contracts
                asserts   : interpret asserts as contracts
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
    return parser


def run_watch_loop(watcher, max_watch_iterations=sys.maxsize) -> None:
    restart = True
    stats: Counter[str] = Counter()
    active_messages: Dict[Tuple[str, int], AnalysisMessage]
    for itr_num in range(max_watch_iterations):
        if restart:
            clear_screen()
            clear_line("-")
            line = f"  Analyzing {len(watcher._modtimes)} files."
            sys.stdout.write(color(line, AnsiColor.OKBLUE))
            sys.stdout.flush()
            max_condition_timeout = 0.5
            restart = False
            stats = Counter()
            active_messages = {}
        else:
            time.sleep(0.5)
            max_condition_timeout *= 2
        for curstats, messages in watcher.run_iteration(max_condition_timeout):
            debug("stats", curstats, messages)
            stats.update(curstats)
            clear_screen()
            if messages_merged(active_messages, messages):
                linecache.checkcache()
            options = DEFAULT_OPTIONS.overlay(watcher._options)
            for message in active_messages.values():
                lines = long_describe_message(message, options)
                if lines is None:
                    continue
                clear_line("-")
                print(lines, end="")
            clear_line("-")
            num_files = len(watcher._modtimes)
            if len(watcher._paths) > 1:
                loc_desc = f"{num_files} files"
            else:
                path_desc = Path(next(iter(watcher._paths))).parts[-1]
                if num_files > 1:
                    loc_desc = f'"{path_desc}" ({num_files} files)'
                else:
                    loc_desc = f'"{path_desc}"'
            line = f'  Analyzed {stats["num_paths"]} paths in {loc_desc}.'
            sys.stdout.write(color(line, AnsiColor.OKBLUE))
            sys.stdout.flush()
        if watcher._change_flag:
            watcher._change_flag = False
            restart = True
            line = f"  Restarting analysis over {len(watcher._modtimes)} files."
            sys.stdout.write(color(line, AnsiColor.OKBLUE))
            sys.stdout.flush()


def clear_screen():
    # Print enough newlines to fill the screen:
    print("\n" * shutil.get_terminal_size().lines, end="")


def clear_line(ch=" "):
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        sys.stdout.write(ch * 5 + "\n")
        return
    sys.stdout.write(ch * cols)


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
        if key not in messages or messages[key] != message:
            messages[key] = message
            any_change = True
    return any_change


def watch(
    args: argparse.Namespace,
    options: AnalysisOptionSet,
    max_watch_iterations=sys.maxsize,
) -> int:
    if not args.directory:
        print("No files or directories given to watch", file=sys.stderr)
        return 2
    try:
        watcher = Watcher(options, args.directory)
        watcher.check_changed()
        run_watch_loop(watcher, max_watch_iterations)
    except KeyboardInterrupt:
        pass
    watcher._pool.terminate()
    print()
    print("I enjoyed working with you today!")
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


def diffbehavior(
    args: argparse.Namespace, options: AnalysisOptions, stdout: TextIO, stderr: TextIO
) -> int:
    def checked_load(qualname: str) -> Optional[FunctionInfo]:
        try:
            objs = list(load_files_or_qualnames([qualname]))
        except Exception as exc:
            print(f'Unable to load "{qualname}": {exc}', file=stderr)
            return None
        obj = objs[0]
        if not isinstance(obj, FunctionInfo):
            print(f'"{qualname}" does not target a function.', file=stderr)
            return None
        return obj

    (fn_name1, fn_name2) = (args.fn1, args.fn2)
    fn1 = checked_load(fn_name1)
    fn2 = checked_load(fn_name2)
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
        print(exc.args[0], file=stderr)
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


def unwalled_main(cmd_args: Union[List[str], argparse.Namespace]) -> None:
    if isinstance(cmd_args, argparse.Namespace):
        args = cmd_args
    else:
        args = command_line_parser().parse_args(cmd_args)
    set_debug(args.verbose)
    options = option_set_from_dict(args.__dict__)
    if sys.path and sys.path[0] != "":
        # fall back to current directory to look up modules
        sys.path.append("")
    if args.action == "check":
        exitcode = check(args, options, sys.stdout, sys.stderr)
    elif args.action == "diffbehavior":
        defaults = DEFAULT_OPTIONS.overlay(
            AnalysisOptionSet(
                per_condition_timeout=2.5,
                per_path_timeout=30.0,  # mostly, we don't want to time out paths
            )
        )
        exitcode = diffbehavior(args, defaults.overlay(options), sys.stdout, sys.stderr)
    elif args.action == "watch":
        exitcode = watch(args, options)
    else:
        print(f'Unknown action: "{args.action}"', file=sys.stderr)
        exitcode = 2
    sys.exit(exitcode)


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
    unwalled_main(check_args)


def main(cmd_args: Optional[List[str]] = None) -> None:
    if cmd_args is None:
        cmd_args = sys.argv[1:]
    engage_auditwall()
    unwalled_main(cmd_args)


if __name__ == "__main__":
    main()
