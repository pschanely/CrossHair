import argparse
import dataclasses
import enum
import heapq
import importlib
import importlib.util
import inspect
import linecache
import multiprocessing
import multiprocessing.pool
import os.path
import shutil
import signal
import sys
import time
import traceback
import types
from typing import *


from crosshair.core import AnalysisMessage, AnalysisOptions, MessageType, analyzable_members, analyze_module, analyze_any
from crosshair.util import debug, extract_module_from_file, set_debug, CrosshairInternal, load_by_qualname

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
    member: object
    qual_name: str
    content_hash: int
    last_modified: float

    def __init__(self, member: object, qual_name: str, body: str) -> None:
        self.member = member
        self.qual_name = qual_name
        self.content_hash = hash(body)
        self.last_modified = time.time()
    
    def consider_new(self, new_version: 'WatchedMember') -> bool:
        if self.content_hash != new_version.content_hash:
            self.content_hash = new_version.content_hash
            self.last_modified = time.time()
            return True
        return False

def worker_initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def worker_main(args: Tuple[WatchedMember, AnalysisOptions]) -> Tuple[WatchedMember, Counter[str], List[AnalysisMessage]]:
    try:
        set_debug(False)
        member, options = args
        stats: Counter[str] = Counter()
        options.stats = stats
        ret = analyze_any(member.member, options)
        return (member, stats, ret)
    except BaseException as e:
        raise CrosshairInternal('Worker failed while analyzing ' + member.qual_name) from e

class Watcher:
    _dirs: Set[str]
    _files: Set[str]
    _pool: multiprocessing.pool.Pool
    _members: Dict[str, WatchedMember]
    _options: AnalysisOptions
    _next_file_check: float = 0.0
    
    def __init__(self, options: AnalysisOptions, files: Iterable[str]):
        self._dirs = set()
        self._files = set()
        self._pool = multiprocessing.Pool(
            processes = max(1, multiprocessing.cpu_count() - 1),
            initializer = worker_initializer,
        )
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

    def run_iteration(self, max_analyze_count=sys.maxsize, max_condition_timeout=0.5) -> Iterator[Tuple[Counter[str], List[AnalysisMessage]]]:
        members = heapq.nlargest(max_analyze_count, self._members.values(), key=lambda m: m.last_modified)
        debug('starting pass on', len(members), 'members, with a condition timeout of', max_condition_timeout)
        def timeout_for_position(pos: int) -> float:
            return max(0.3, (1.0 - pos / max_analyze_count) * max_condition_timeout)
        members_and_options = [(m, dataclasses.replace(self._options, per_condition_timeout=timeout_for_position(i)))
                               for (i, m) in enumerate(members)]
        for member, stats, messages in self._pool.imap_unordered(worker_main, members_and_options):
            if messages:
                yield (stats, messages)
        debug('Worker pool tasks complete')

    def check_all_files(self) -> bool:
        if time.time() < self._next_file_check:
            return False
        any_changed = False
        for curfile in self._files:
            any_changed |= self.check_file(curfile)
        for curdir in self._dirs:
            for (dirpath, dirs, files) in os.walk(curdir):
                for curfile in files:
                    if not curfile.endswith('.py'):
                        continue
                    any_changed |= self.check_file(os.path.join(dirpath, curfile))
        self._next_file_check = time.time() + 1.0
        if any_changed:
            debug('Noticed change(s) among', len(self._members), 'files')
        return any_changed

    def check_file(self, curfile: str) -> bool:
        any_changed = False
        members = self._members

        _, name = extract_module_from_file(curfile)
        module = importlib.import_module(name)
        
        for (name, member) in analyzable_members(module):
            qualname = module.__name__ + '.' + member.__name__ # type: ignore
            src = inspect.getsource(member)
            wm = WatchedMember(member, qualname, src)
            if qualname in members:
                changed = members[qualname].consider_new(wm)
                any_changed |= changed
                if changed:
                    debug('Updated', qualname)
            else:
                members[qualname] = wm
                debug('Found', qualname)
        return any_changed

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

def watch(args: argparse.Namespace, options: AnalysisOptions) -> int:
    if not args.files:
        print('No files or directories given to watch', file=sys.stderr)
        return 1
    watcher = Watcher(options, args.files)
    watcher.check_all_files()
    restart = True
    stats: Counter[str] = Counter()
    clear_screen()
    sys.stdout.write(color(f'  Analyzing {len(watcher._members)} classes/functions.          \r', AnsiColor.OKBLUE))
    try:
        while True:
            if restart:
                max_analyze_count = 5
                max_condition_timeout = 0.5
                restart = False
            else:
                max_analyze_count *= 2
                max_condition_timeout *= 2
            for curstats, messages in watcher.run_iteration(max_analyze_count, max_condition_timeout):
                stats.update(curstats)
                sys.stdout.write(color(f'  Analyzed {stats["num_paths"]} paths in {len(watcher._members)} classes/functions.          \r', AnsiColor.OKBLUE))
                if watcher.check_all_files():
                    restart = True
                    break
                for message in messages:
                    lines = long_describe_message(message, max_condition_timeout)
                    if lines is None:
                        continue
                    clear_line('-')
                    print(lines, end='')
    except KeyboardInterrupt:
        watcher._pool.close()
        time.sleep(0.2)
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
            entity = importlib.import_module(name)
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
