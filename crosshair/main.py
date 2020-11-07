import argparse
import collections
import dataclasses
import enum
import heapq
import importlib
import importlib.util
import inspect
import json
import linecache
import multiprocessing
import multiprocessing.queues
import os
import os.path
import queue
import shutil
import signal
import sys
import time
import traceback
from typing import *
from typing import TextIO

from crosshair.localhost_comms import StateUpdater, read_states
from crosshair.core_and_libs import AnalysisMessage, AnalysisOptions, MessageType, analyzable_members, analyze_module, analyze_any
from crosshair.util import debug, extract_module_from_file, set_debug, CrosshairInternal, load_file, load_by_qualname, NotFound, ErrorDuringImport
import crosshair.core_and_libs

def command_line_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--verbose', '-v', action='store_true')
    common.add_argument('--per_path_timeout', type=float,
                        help='Maximum seconds to spend checking one execution path')
    common.add_argument('--per_condition_timeout', type=float,
                        help='Maximum seconds to spend checking the execution paths of one postcondition')
    parser = argparse.ArgumentParser(description='CrossHair Analysis Tool')
    subparsers = parser.add_subparsers(help='sub-command help', dest='action')
    check_parser = subparsers.add_parser(
        'check', help='Analyze a file', parents=[common])
    check_parser.add_argument('--report_all', action='store_true',
                              help='Output analysis results for all postconditions (not just failing ones)')
    check_parser.add_argument('file', metavar='F', type=str, nargs='+',
                              help='file or fully qualified module, class, or function')
    watch_parser = subparsers.add_parser(
        'watch', help='Continuously watch and analyze a directory', parents=[common])
    watch_parser.add_argument('directory', metavar='F', type=str, nargs='+',
                              help='file or directory to analyze')
    showresults_parser = subparsers.add_parser(
        'showresults', help='Display results from a currently running `watch` command', parents=[common])
    showresults_parser.add_argument('file', metavar='F', type=str, nargs='+',
                                    help='file or directory to analyze')
    return parser

def mtime(path: str) -> Optional[float]:
    try:
        return os.stat(path).st_mtime
    except FileNotFoundError:
        return None

def process_level_options(command_line_args: argparse.Namespace) -> AnalysisOptions:
    options = AnalysisOptions()
    for optname in ('per_path_timeout', 'per_condition_timeout', 'report_all'):
        arg_val = getattr(command_line_args, optname, False)
        if arg_val is not None:
            setattr(options, optname, arg_val)
    return options


@dataclasses.dataclass(init=False)
class WatchedMember:
    qual_name: str
    content_hash: int
    last_modified: float

    def get_member(self):
        return load_by_qualname(self.qual_name)

    def __init__(self, qual_name: str, body: str) -> None:
        self.qual_name = qual_name
        self.content_hash = hash(body)
        self.last_modified = time.time()

    def consider_new(self, new_version: 'WatchedMember') -> bool:
        if self.content_hash != new_version.content_hash:
            self.content_hash = new_version.content_hash
            self.last_modified = time.time()
            return True
        return False


WorkItemInput = Tuple[str, # (filename)
                      AnalysisOptions, float]  # (float is a deadline)
WorkItemOutput = Tuple[WatchedMember, Counter[str], List[AnalysisMessage]]

def import_error_msg(err: ErrorDuringImport) -> AnalysisMessage:
    orig, frame = err.args
    return AnalysisMessage(MessageType.IMPORT_ERR, str(orig),
                           frame.filename, frame.lineno, 0, '')

def pool_worker_main(item: WorkItemInput, output: multiprocessing.queues.Queue) -> None:
    try:
        # TODO figure out a more reliable way to suppress this. Redirect output?
        # Ignore ctrl-c in workers to reduce noisy tracebacks (the parent will kill us):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if hasattr(os, 'nice'): # analysis should run at a low priority
            os.nice(10)
        set_debug(False)
        filename, options, deadline = item
        stats: Counter[str] = Counter()
        options.stats = stats
        _, module_name = extract_module_from_file(filename)
        try:
            module = load_by_qualname(module_name)
        except NotFound:
            return
        except ErrorDuringImport as e:
            output.put((filename, stats, [import_error_msg(e)]))
            debug(f'Not analyzing "{filename}" because import failed: {e}')
            return
        messages = analyze_any(module, options)
        output.put((filename, stats, messages))
    except BaseException as e:
        raise CrosshairInternal(
            'Worker failed while analyzing ' + filename) from e


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
            process = multiprocessing.Process(
                target=pool_worker_main, args=(work_item, self._results))
            workers.append((process, work_item))
            process.start()

    def _prune_workers(self, curtime):
        for worker, item in self._workers:
            (_, _, deadline) = item
            if worker.is_alive() and curtime > deadline:
                debug('Killing worker over deadline', worker)
                worker.terminate()
                time.sleep(0.5)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
        self._workers = [(w, i) for w, i in self._workers if w.is_alive()]

    def terminate(self):
        self._prune_workers(float('+inf'))
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

def analyzable_filename(filename: str) -> bool:
    if not filename.endswith('.py'):
        return False
    lead_char = filename[0]
    if (not lead_char.isalpha()) and (not lead_char.isidentifier()):
        # (skip temporary editor files, backups, etc)
        debug(
            f'Skipping {filename} because it begins with a special character.')
        return False
    if filename in ('setup.py',):
        debug(
            f'Skipping {filename} because files with this name are not usually import-able.')
        return False
    return True

def walk_paths(paths: Iterable[str]) -> Iterable[str]:
    for name in paths:
        if not os.path.exists(name):
            print(f'Watch path "{name}" does not exist.', file=sys.stderr)
            sys.exit(1)
        if os.path.isdir(name):
            for (dirpath, dirs, files) in os.walk(name):
                for curfile in files:
                    if analyzable_filename(curfile):
                        yield os.path.join(dirpath, curfile)
        else:
            yield name

class Watcher:
    _paths: Set[str]
    _pool: Pool
    _modtimes: Dict[str, float]
    _options: AnalysisOptions
    _next_file_check: float = 0.0
    _change_flag: bool = False

    def __init__(self, options: AnalysisOptions, files: Iterable[str], state_updater: StateUpdater):
        self._paths = set(files)
        self._state_updater = state_updater
        self._pool = self.startpool()
        self._modtimes = {}
        self._options = options
        _ = list(walk_paths(self._paths)) # just to force an exit if we can't find a path

    def startpool(self) -> Pool:
        return Pool(multiprocessing.cpu_count() - 1)

    def run_iteration(self,
                      max_condition_timeout=0.5) -> Iterator[
                          Tuple[Counter[str], List[AnalysisMessage]]]:
        debug(f'starting pass '
              f'with a condition timeout of {max_condition_timeout}')
        debug('Files:', self._modtimes.keys())
        pool = self._pool
        for filename in self._modtimes.keys():
            worker_timeout = max(10.0, max_condition_timeout * 20.0)
            options = dataclasses.replace(
                self._options, per_condition_timeout=max_condition_timeout)
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
                debug('Aborting iteration on change detection')
                pool.terminate()
                self._pool = self.startpool()
                return
            pool.garden_workers()
        debug('Worker pool tasks complete')
        yield (Counter(), [])

    def run_watch_loop(self) -> NoReturn:
        restart = True
        stats: Counter[str] = Counter()
        active_messages: Dict[Tuple[str, int], AnalysisMessage]
        while True:
            if restart:
                clear_screen()
                clear_line('-')
                line = f'  Analyzing {len(self._modtimes)} files.          \r'
                sys.stdout.write(color(line, AnsiColor.OKBLUE))
                max_condition_timeout = 0.5
                restart = False
                stats = Counter()
                active_messages = {}
            else:
                time.sleep(0.5)
                max_condition_timeout *= 2
            for curstats, messages in self.run_iteration(max_condition_timeout):
                debug('stats', curstats, messages)
                stats.update(curstats)
                if messages_merged(active_messages, messages):
                    self._state_updater.update(json.dumps({
                        'version': 1,
                        'time': time.time(),
                        'messages': [m.toJSON() for m in active_messages.values()]}))
                    linecache.checkcache()
                    clear_screen()
                    for message in active_messages.values():
                        lines = long_describe_message(message)
                        if lines is None:
                            continue
                        clear_line('-')
                        print(lines, end='')
                    clear_line('-')
                line = f'  Analyzed {stats["num_paths"]} paths in {len(self._modtimes)} files.          \r'
                sys.stdout.write(color(line, AnsiColor.OKBLUE))
            if self._change_flag:
                self._change_flag = False
                restart = True
                line = f'  Restarting analysis over {len(self._modtimes)} files.          \r'
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
    print("\n" * shutil.get_terminal_size().lines, end='')


def clear_line(ch=' '):
    sys.stdout.write(ch * shutil.get_terminal_size().columns)


class AnsiColor(enum.Enum):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def color(text: str, *effects: AnsiColor) -> str:
    return ''.join(e.value for e in effects) + text + AnsiColor.ENDC.value


def messages_merged(messages: MutableMapping[Tuple[str, int], AnalysisMessage],
                    new_messages: Iterable[AnalysisMessage]) -> bool:
    any_change = False
    for message in new_messages:
        key = (message.filename, message.line)
        if key not in messages or messages[key] != message:
            messages[key] = message
            any_change = True
    return any_change


def watch(args: argparse.Namespace, options: AnalysisOptions) -> int:
    # Avoid fork() because we've already imported the code we're watching:
    multiprocessing.set_start_method('spawn')
    if not args.directory:
        print('No files or directories given to watch', file=sys.stderr)
        return 1
    try:
        with StateUpdater() as state_updater:
            watcher = Watcher(options, args.directory, state_updater)
            watcher.check_changed()
            watcher.run_watch_loop()
    except KeyboardInterrupt:
        watcher._pool.terminate()
        print()
        print('I enjoyed working with you today!')
        return 0


def format_src_context(filename: str, lineno: int) -> str:
    amount = 3
    line_numbers = range(max(1, lineno - amount), lineno + amount + 1)
    output = [f'{filename}:{lineno}:\n']
    for curline in line_numbers:
        text = linecache.getline(filename, curline)
        if text == '': # (actual empty lines have a newline)
            continue
        output.append('>' + color(text, AnsiColor.WARNING)
                      if lineno == curline else '|' + text)
    return ''.join(output)


def long_describe_message(message: AnalysisMessage) -> Optional[str]:
    tb, desc, state = message.traceback, message.message, message.state
    desc = desc.replace(' when ', '\nwhen ')
    context = format_src_context(message.filename, message.line)
    intro = ''
    if state <= MessageType.CANNOT_CONFIRM:  # type: ignore
        return None
    elif message.state == MessageType.PRE_UNSAT:
        # TODO: This is disabled as unsat reasons are too common
        # intro = "I am having trouble finding any inputs that meet this precondition."
        return None
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
    intro = color(intro, AnsiColor.FAIL)
    return f'{tb}\n{intro}\n{context}\n{desc}\n'


def short_describe_message(message: AnalysisMessage, options: AnalysisOptions) -> Optional[str]:
    desc = message.message
    if message.state <= MessageType.PRE_UNSAT:  # type: ignore
        if options.report_all:
            return '{}:{}:{}:{}'.format(message.filename, message.line, 'info', desc)
        return None
    if message.state == MessageType.POST_ERR:
        desc = 'Error while evaluating post condition: ' + desc
    return '{}:{}:{}:{}'.format(message.filename, message.line, 'error', desc)


def showresults(args: argparse.Namespace, options: AnalysisOptions) -> int:
    messages_by_file: Dict[str, List[AnalysisMessage]] = collections.defaultdict(list)
    states = list(read_states())
    for fname, content in states:
        debug('Found watch state file at ', fname)
        state = json.loads(content)
        for message in state['messages']:
            messages_by_file[message['filename']].append(AnalysisMessage.fromJSON(message))
    debug('Found results for these files: [', ', '.join(messages_by_file.keys()), ']')
    for name in walk_paths(args.file):
        name = os.path.abspath(name)
        debug('Checking file ', name)
        for message in messages_by_file[name]:
            desc = short_describe_message(message, options)
            debug('Describing ', message)
            if desc is not None:
                print(desc)
    return 0

def check(args: argparse.Namespace, options: AnalysisOptions, stdout: TextIO) -> int:
    any_problems = False
    for name in args.file:
        entity: object
        try:
            entity = load_file(name) if name.endswith('.py') else load_by_qualname(name)
        except ErrorDuringImport as e:
            stdout.write(str(short_describe_message(import_error_msg(e), options)) + '\n')
            any_problems = True
            continue
        debug('Check ', getattr(entity, '__name__', str(entity)))
        for message in analyze_any(entity, options):
            line = short_describe_message(message, options)
            if line is None:
                continue
            stdout.write(line + '\n')
            debug('Traceback for output message:\n', message.traceback)
            if message.state > MessageType.PRE_UNSAT:
                any_problems = True
    return 2 if any_problems else 0


def main(cmd_args: Optional[List[str]] = None) -> None:
    if cmd_args is None:
        cmd_args = sys.argv[1:]
    args = command_line_parser().parse_args(cmd_args)
    set_debug(args.verbose)
    options = process_level_options(args)
    if sys.path and sys.path[0] != '':
        # fall back to current directory to look up modules
        sys.path.append('')
    if args.action == 'check':
        exitcode = check(args, options, sys.stdout)
    elif args.action == 'showresults':
        exitcode = showresults(args, options)
    elif args.action == 'watch':
        exitcode = watch(args, options)
    else:
        print(f'Unknown action: "{args.action}"', file=sys.stderr)
        exitcode = 1
    sys.exit(exitcode)


if __name__ == '__main__':
    main()
