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
import multiprocessing.queues
import os
import os.path
import queue
import re
import shutil
import signal
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
from crosshair.fnutil import load_function_at_line
from crosshair.fnutil import walk_paths
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
        "file",
        metavar="FILE",
        type=str,
        nargs="+",
        help="file/directory or fully qualified module, class, or function",
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
        metavar="FILE",
        type=str,
        nargs="+",
        help=textwrap.dedent(
            """\
        File or directory to watch. Directories will be recursively analyzed.
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


def mtime(path: str) -> Optional[float]:
    try:
        return os.stat(path).st_mtime
    except FileNotFoundError:
        return None


@dataclasses.dataclass(init=False)
class WatchedMember:
    qual_name: str  # (just for debugging)
    content_hash: int
    last_modified: float

    def __init__(self, qual_name: str, body: str) -> None:
        self.qual_name = qual_name
        self.content_hash = hash(body)
        self.last_modified = time.time()

    def consider_new(self, new_version: "WatchedMember") -> bool:
        if self.content_hash != new_version.content_hash:
            self.content_hash = new_version.content_hash
            self.last_modified = time.time()
            return True
        return False


WorkItemInput = Tuple[
    str, AnalysisOptionSet, float  # (filename)
]  # (float is a deadline)
WorkItemOutput = Tuple[WatchedMember, Counter[str], List[AnalysisMessage]]


def import_error_msg(err: ErrorDuringImport) -> AnalysisMessage:
    orig, frame = err.args
    return AnalysisMessage(
        MessageType.IMPORT_ERR, str(orig), frame.filename, frame.lineno, 0, ""
    )


def pool_worker_process_item(
    item: WorkItemInput,
) -> Tuple[Counter[str], List[AnalysisMessage]]:
    filename, options, deadline = item
    stats: Counter[str] = Counter()
    options.stats = stats
    try:
        module = load_file(filename)
    except NotFound as e:
        debug(f'Not analyzing "{filename}" because sub-module import failed: {e}')
        return (stats, [])
    except ErrorDuringImport as e:
        debug(f'Not analyzing "{filename}" because import failed: {e}')
        return (stats, [import_error_msg(e)])
    messages = run_checkables(analyze_module(module, options))
    return (stats, messages)


def pool_worker_main(item: WorkItemInput, output: multiprocessing.queues.Queue) -> None:
    try:
        # TODO figure out a more reliable way to suppress this. Redirect output?
        # Ignore ctrl-c in workers to reduce noisy tracebacks (the parent will kill us):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if hasattr(os, "nice"):  # analysis should run at a low priority
            os.nice(10)
        set_debug(False)
        engage_auditwall()
        (stats, messages) = pool_worker_process_item(item)
        filename = item[0]
        output.put((filename, stats, messages))
    except BaseException as e:
        raise CrosshairInternal("Worker failed while analyzing " + filename) from e


class Pool:
    _workers: List[Tuple[multiprocessing.Process, WorkItemInput]]
    _work: List[WorkItemInput]
    _results: multiprocessing.queues.Queue
    _max_processes: int

    def __init__(self, max_processes: int) -> None:
        self._workers = []
        self._work = []
        self._results = multiprocessing.Queue()
        self._max_processes = max_processes

    def _spawn_workers(self):
        work_list = self._work
        workers = self._workers
        while work_list and len(self._workers) < self._max_processes:
            work_item = work_list.pop()
            with opened_auditwall():
                process = multiprocessing.Process(
                    target=pool_worker_main, args=(work_item, self._results)
                )
                workers.append((process, work_item))
                process.start()

    def _prune_workers(self, curtime):
        for worker, item in self._workers:
            (_, _, deadline) = item
            if worker.is_alive() and curtime > deadline:
                debug("Killing worker over deadline", worker)
                with opened_auditwall():
                    worker.terminate()
                    time.sleep(0.5)
                    if worker.is_alive():
                        worker.kill()
                        worker.join()
        self._workers = [(w, i) for w, i in self._workers if w.is_alive()]

    def terminate(self):
        self._prune_workers(float("+inf"))
        self._work = []
        self._results.close()

    def garden_workers(self):
        self._prune_workers(time.time())
        self._spawn_workers()

    def is_working(self):
        return self._workers or self._work

    def submit(self, item: WorkItemInput) -> None:
        self._work.append(item)

    def has_result(self):
        return not self._results.empty()

    def get_result(self, timeout: float) -> Optional[WorkItemOutput]:
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None


def worker_initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class Watcher:
    _paths: Set[str]
    _pool: Pool
    _modtimes: Dict[str, float]
    _options: AnalysisOptionSet
    _next_file_check: float = 0.0
    _change_flag: bool = False

    def __init__(self, options: AnalysisOptionSet, files: Iterable[str]):
        self._paths = set(files)
        self._pool = self.startpool()
        self._modtimes = {}
        self._options = options
        try:
            # just to force an exit if we can't find a path:
            list(walk_paths(self._paths))
        except FileNotFoundError as exc:
            print(f'Watch path "{exc.args[0]}" does not exist.', file=sys.stderr)
            sys.exit(2)

    def startpool(self) -> Pool:
        return Pool(multiprocessing.cpu_count() - 1)

    def run_iteration(
        self, max_condition_timeout=0.5
    ) -> Iterator[Tuple[Counter[str], List[AnalysisMessage]]]:
        debug(f"starting pass " f"with a condition timeout of {max_condition_timeout}")
        debug("Files:", self._modtimes.keys())
        pool = self._pool
        for filename in self._modtimes.keys():
            worker_timeout = max(10.0, max_condition_timeout * 20.0)
            iter_options = AnalysisOptionSet(
                per_condition_timeout=max_condition_timeout,
                per_path_timeout=max_condition_timeout / 4,
            )
            options = self._options.overlay(iter_options)
            pool.submit((filename, options, time.time() + worker_timeout))

        pool.garden_workers()
        while pool.is_working():
            result = pool.get_result(timeout=1.0)
            if result is not None:
                (_, counters, messages) = result
                yield (counters, messages)
                if pool.has_result():
                    continue
            change_detected = self.check_changed()
            if change_detected:
                self._change_flag = True
                debug("Aborting iteration on change detection")
                pool.terminate()
                self._pool = self.startpool()
                return
            pool.garden_workers()
        debug("Worker pool tasks complete")
        yield (Counter(), [])

    def run_watch_loop(self, max_watch_iterations=sys.maxsize) -> None:
        restart = True
        stats: Counter[str] = Counter()
        active_messages: Dict[Tuple[str, int], AnalysisMessage]
        for itr_num in range(max_watch_iterations):
            if restart:
                clear_screen()
                clear_line("-")
                line = f"  Analyzing {len(self._modtimes)} files.          \r"
                sys.stdout.write(color(line, AnsiColor.OKBLUE))
                max_condition_timeout = 0.5
                restart = False
                stats = Counter()
                active_messages = {}
            else:
                time.sleep(0.5)
                max_condition_timeout *= 2
            for curstats, messages in self.run_iteration(max_condition_timeout):
                debug("stats", curstats, messages)
                stats.update(curstats)
                if messages_merged(active_messages, messages):
                    linecache.checkcache()
                    clear_screen()
                    options = DEFAULT_OPTIONS.overlay(self._options)
                    for message in active_messages.values():
                        lines = long_describe_message(message, options)
                        if lines is None:
                            continue
                        clear_line("-")
                        print(lines, end="")
                    clear_line("-")
                line = f'  Analyzed {stats["num_paths"]} paths in {len(self._modtimes)} files.          \r'
                sys.stdout.write(color(line, AnsiColor.OKBLUE))
            if self._change_flag:
                self._change_flag = False
                restart = True
                line = f"  Restarting analysis over {len(self._modtimes)} files.          \r"
                sys.stdout.write(color(line, AnsiColor.OKBLUE))

    def check_changed(self) -> bool:
        if time.time() < self._next_file_check:
            return False
        modtimes = self._modtimes
        changed = False
        for curfile in walk_paths(self._paths):
            cur_mtime = mtime(curfile)
            if cur_mtime == modtimes.get(curfile):
                continue
            changed = True
            if cur_mtime is None:
                del modtimes[curfile]
            else:
                modtimes[curfile] = cur_mtime
        self._next_file_check = time.time() + 1.0
        if not changed:
            return False
        return True


def clear_screen():
    print("\n" * shutil.get_terminal_size().lines, end="")


def clear_line(ch=" "):
    sys.stdout.write(ch * shutil.get_terminal_size().columns)


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
    # Avoid fork() because we've already imported the code we're watching:
    multiprocessing.set_start_method("spawn")
    if not args.directory:
        print("No files or directories given to watch", file=sys.stderr)
        return 2
    try:
        watcher = Watcher(options, args.directory)
        watcher.check_changed()
        watcher.run_watch_loop(max_watch_iterations)
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
            # TODO detect not a function?
            return load_by_qualname(qualname)
        except Exception as exc:
            print(f'Unable to load "{qualname}": {exc}', file=stderr)
            return None

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
        entities = list(load_files_or_qualnames(args.file))
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
    mypy_cmd_args = mypy_args + check_args.file
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
