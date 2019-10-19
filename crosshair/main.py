import argparse
import dataclasses
import enum
import heapq
import importlib
import importlib.util
import inspect
import linecache
import multiprocessing
import multiprocessing.queues
import os.path
import queue
import shutil
import signal
import sys
import time
import traceback
import types
from typing import *


from crosshair.core import AnalysisMessage, AnalysisOptions, MessageType, analyzable_members, analyze_module, analyze_any, exception_line_in_file
from crosshair.util import debug, extract_module_from_file, set_debug, CrosshairInternal, load_by_qualname, NotFound

def command_line_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CrossHair Analysis Tool')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--per_path_timeout', type=float)
    parser.add_argument('--per_condition_timeout', type=float)
    subparsers = parser.add_subparsers(help='sub-command help', dest='action')
    check_parser = subparsers.add_parser('check', help='Analyze one or more files')
    check_parser.add_argument('files', metavar='F', type=str, nargs='+',
                              help='files or fully qualified modules, classes, or functions')
    watch_parser = subparsers.add_parser('watch', help='Continuously watch and analyze files')
    watch_parser.add_argument('files', metavar='F', type=str, nargs='+',
                              help='files or directories to analyze')
    return parser
    
def process_level_options(command_line_args: argparse.Namespace) -> AnalysisOptions:
    options = AnalysisOptions()
    for optname in ('per_path_timeout', 'per_condition_timeout'):
        arg_val = getattr(command_line_args, optname)
        if arg_val is not None:
            setattr(options, optname, arg_val)
    return options

@dataclasses.dataclass(init=False)
class WatchedMember:
    #member: object
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


WorkItemInput = Tuple[WatchedMember, AnalysisOptions, float] # (float is a deadline)
WorkItemOutput = Tuple[WatchedMember, Counter[str], List[AnalysisMessage]]

def pool_worker_main(item: WorkItemInput, output: multiprocessing.queues.Queue) -> None:
    try:
        # TODO figure out a more reliable way to suppress this. Redirect output?
        # Ignore ctrl-c in workers to reduce noisy tracebacks (the parent will kill us):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(os, 'nice'): # <- is this the right way to detect availability?
            os.nice(10) # analysis should run at a low priority
        set_debug(False)
        member, options, deadline = item
        stats: Counter[str] = Counter()
        options.stats = stats
        try:
            fn = member.get_member()
        except NotFound:
            return
        messages = analyze_any(fn, options)
        output.put((member, stats, messages))
    except BaseException as e:
        raise CrosshairInternal('Worker failed while analyzing ' + member.qual_name) from e

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
            process = multiprocessing.Process(target=pool_worker_main, args=(work_item, self._results))
            workers.append((process, work_item))
            process.start()
    def _prune_workers(self, curtime):
        for worker, item in self._workers:
            (_, _, deadline) = item
            if worker.is_alive() and curtime > deadline:
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


class Watcher:
    _dirs: Set[str]
    _files: Set[str]
    _pool: Pool
    _members: Dict[str, WatchedMember]
    _options: AnalysisOptions
    _next_file_check: float = 0.0
    _change_flag: bool = False
    
    def __init__(self, options: AnalysisOptions, files: Iterable[str]):
        self._dirs = set()
        self._files = set()
        self._pool = self.startpool()
        self._members = {}
        self._options = options
        for name in files:
            if not os.path.exists(name):
                print(f'Watch path "{name}" does not exist.', file=sys.stderr)
                sys.exit(1)
            if os.path.isdir(name):
                self._dirs.add(name)
            else:
                self._files.add(name)

    def startpool(self) -> Pool:
        return Pool(multiprocessing.cpu_count() - 1)

    def run_iteration(self,
                      max_analyze_count=sys.maxsize,
                      max_condition_timeout=0.5) -> Iterator[
                          Tuple[Counter[str], List[AnalysisMessage]]]:
        members = heapq.nlargest(max_analyze_count, self._members.values(), key=lambda m: m.last_modified)
        debug(f'starting pass on {len(members)}/{len(self._members)} members, '
              f'with a condition timeout of {max_condition_timeout}')
        def timeout_for_position(pos: int) -> float:
            return max(0.3, (1.0 - pos / max_analyze_count) * max_condition_timeout)
        pool = self._pool
        for (index, member) in enumerate(members):
            condition_timeout = timeout_for_position(index)
            worker_timeout = max(1.0, condition_timeout * 3.0)
            options = dataclasses.replace(self._options, per_condition_timeout=condition_timeout)
            pool.submit((member, options, time.time() + worker_timeout))
                               
        pool.garden_workers()
        while pool.is_working():
            result = pool.get_result(timeout=1.0)
            if result is not None:
                (_, counters, messages) = result
                yield (counters, messages)
                if pool.has_result():
                    continue
            messages = self.check_all_files()
            if messages:
                yield (Counter(), messages)
            if self._change_flag:
                debug('Aborting iteration on change detection')
                pool.terminate()
                self._pool = self.startpool()
                return
            pool.garden_workers()
        debug('Worker pool tasks complete')
        yield (Counter(), self.check_all_files())

    def check_all_files(self) -> List[AnalysisMessage]:
        if time.time() < self._next_file_check:
            return []
        messages = []
        members_found: Set[str] = set()
        for curfile in self._files:
            messages.extend(self.check_file(curfile, members_found))
        for curdir in self._dirs:
            for (dirpath, dirs, files) in os.walk(curdir):
                for curfile in files:
                    if curfile.endswith('.py') and curfile[:-3].isidentifier():
                        messages.extend(self.check_file(os.path.join(dirpath, curfile), members_found))
        # prune missing members:
        for k in list(self._members.keys()):
            if k not in members_found:
                del self._members[k]
                self._change_flag = True
        self._next_file_check = time.time() + 1.0
        return messages

    def check_file(self, curfile: str, found: Set[str]) -> List[AnalysisMessage]:
        debug('check_file', curfile)
        members = self._members
        _, name = extract_module_from_file(curfile)
        try:
            module = importlib.import_module(name)
            importlib.reload(module)
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            assert exc_traceback is not None
            lineno = exception_line_in_file(traceback.extract_tb(exc_traceback), curfile)
            if lineno is None:
                lineno = 1
            debug(f'Unable to reload the module in {curfile}: {exc_value}')
            return [AnalysisMessage(MessageType.IMPORT_ERR, str(exc_value), curfile, lineno, 0, '')]

        for (name, member) in analyzable_members(module):
            qualname = module.__name__ + '.' + member.__name__
            found.add(qualname)
            src = inspect.getsource(member)
            wm = WatchedMember(qualname, src)
            if qualname in members:
                changed = members[qualname].consider_new(wm)
                if changed:
                    debug('Updated', qualname)
                    self._change_flag = True
            else:
                members[qualname] = wm
                debug('Found', qualname)
                self._change_flag = True
        return []

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

def color(text: str, *effects:AnsiColor) -> str:
    return ''.join(e.value for e in effects) + text + AnsiColor.ENDC.value

def messages_merged(messages: MutableMapping[Tuple[str, int], AnalysisMessage],
                    new_messages: Iterable[AnalysisMessage]):
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

    if not args.files:
        print('No files or directories given to watch', file=sys.stderr)
        return 1
    watcher = Watcher(options, args.files)
    watcher.check_all_files()
    restart = True
    stats: Counter[str] = Counter()
    active_messages: Dict[Tuple[str, int], AnalysisMessage]
    try:
        while True:
            if restart:
                clear_screen()
                clear_line('-')
                line = f'  Analyzing {len(watcher._members)} classes/functions.          \r'
                sys.stdout.write(color(line, AnsiColor.OKBLUE))
                max_analyze_count = 5
                max_condition_timeout = 0.5
                restart = False
                stats = Counter()
                active_messages = {}
            else:
                time.sleep(0.5)
                max_analyze_count *= 2
                max_condition_timeout *= 2
            for curstats, messages in watcher.run_iteration(max_analyze_count, max_condition_timeout):
                debug('stats', curstats, messages)
                stats.update(curstats)
                if messages_merged(active_messages, messages):
                    clear_screen()
                    for message in active_messages.values():
                        lines = long_describe_message(message, max_condition_timeout)
                        if lines is None:
                            continue
                        clear_line('-')
                        print(lines, end='')
                    clear_line('-')
                line = f'  Analyzed {stats["num_paths"]} paths in {len(watcher._members)} classes/functions.          \r'
                sys.stdout.write(color(line, AnsiColor.OKBLUE))
            if watcher._change_flag:
                watcher._change_flag = False
                restart = True
                line = f'  Restarting analysis over {len(watcher._members)} classes/functions.          \r'
                sys.stdout.write(color(line, AnsiColor.OKBLUE))
    except KeyboardInterrupt:
        watcher._pool.terminate()
        print()
        print('I enjoyed working with you today!')
        sys.exit(0)

def format_src_context(filename: str, lineno: int) -> str:
    amount = 2
    line_numbers = range(max(1, lineno - amount), lineno + amount + 1)
    output = [f'{filename}:{lineno}:\n']
    for curline in line_numbers:
        text = linecache.getline(filename, curline)
        output.append('>' + color(text, AnsiColor.WARNING) if lineno == curline else '|'+text)
    return ''.join(output)

def long_describe_message(message: AnalysisMessage, max_condition_timeout: float) -> Optional[str]:
    tb, desc, state = message.traceback, message.message, message.state
    desc = desc.replace(' when ', '\nwhen ')
    context = format_src_context(message.filename, message.line)
    intro = ''
    if state == MessageType.CANNOT_CONFIRM:
        return None
    elif message.state == MessageType.PRE_UNSAT:
        if max_condition_timeout < 10.0:
            return None
        intro = "I am having trouble finding any inputs that meet this precondition."
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

def short_describe_message(message: AnalysisMessage) -> Optional[str]:
    if message.state == MessageType.CANNOT_CONFIRM:
        return None
    desc = message.message
    if message.state == MessageType.POST_ERR:
        desc = 'Error while evaluating post condition: ' + desc
    return '{}:{}:{}:{}'.format(message.filename, message.line, 'error', desc)

def check(args: argparse.Namespace, options: AnalysisOptions) -> int:
    any_errors = False
    for name in args.files:
        entity: object
        if name.endswith('.py'):
            _, name = extract_module_from_file(name)
            try:
                entity = importlib.import_module(name)
            except Exception as e:
                debug(f'Not analyzing "{name}" because import failed: {e}')
                continue
        else:
            entity = load_by_qualname(name)
        debug('Check ', getattr(entity, '__name__', str(entity)))
        for message in analyze_any(entity, options):
            line = short_describe_message(message)
            if line is not None:
                debug(message.traceback)
                print(line)
                any_errors = True
    return 1 if any_errors else 0

def main() -> None:
    args = command_line_parser().parse_args()
    set_debug(args.verbose)
    options = process_level_options(args)
    if args.action == 'check':
        exitcode = check(args, options)
    elif args.action == 'watch':
        exitcode = watch(args, options)
    else:
        print(f'Unknown action: "{args.action}"', file=sys.stderr)
        exitcode = 1
    sys.exit(exitcode)

if __name__ == '__main__':
    main()
