import argparse
import dataclasses
import heapq
import importlib
import importlib.util
import inspect
import multiprocessing
import multiprocessing.pool
import os.path
import sys
import time
import traceback
import types
from typing import *

from crosshair import core
from crosshair.util import debug, extract_module_from_file, set_debug, CrosshairInternal

def command_line_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='CrossHair Analysis Tool')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--per_path_timeout', type=float)
    parser.add_argument('--per_condition_timeout', type=float)
    subparsers = parser.add_subparsers(help='sub-command help', dest='action')
    check_parser = subparsers.add_parser('check', help='Analyze one or more files')
    check_parser.add_argument('files', metavar='F', type=str, nargs='+',
                              help='files or directories to analyze')
    watch_parser = subparsers.add_parser('watch', help='Continuously watch and analyze files')
    watch_parser.add_argument('files', metavar='F', type=str, nargs='+',
                              help='files or directories to analyze')
    return parser
    
def process_level_options(command_line_args: argparse.Namespace) -> core.AnalysisOptions:
    options = core.AnalysisOptions()
    for optname in ('per_path_timeout', 'per_condition_timeout'):
        arg_val = getattr(command_line_args, optname)
        if arg_val is not None:
            setattr(options, optname, arg_val)
    return options

def module_for_file(filepath:str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location('crosshair.examples.tic_tac_toe', filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod) # type:ignore
    return mod

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

def worker_main(args: Tuple[WatchedMember, core.AnalysisOptions]) -> Tuple[WatchedMember, List[core.AnalysisMessage]]:
    try:
        set_debug(False)
        member, options = args
        ret = core.analyze_any(member.member, options)
        return (member, ret)
    except BaseException as e:
        raise CrosshairInternal('Worker failed while analyzing ' + member.qual_name) from e

class Watcher:
    _dirs: Set[str]
    _files: Set[str]
    _pool: multiprocessing.pool.Pool
    _members: Dict[str, WatchedMember]
    _options: core.AnalysisOptions
    _next_file_check: float = 0.0
    
    def __init__(self, options: core.AnalysisOptions, files: Iterable[str]):
        self._dirs = set()
        self._files = set()
        self._pool = multiprocessing.Pool(processes = max(1, multiprocessing.cpu_count() - 1))
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

    def run_iteration(self, max_analyze_count=sys.maxsize, max_condition_timeout=0.5) -> Iterator[List[core.AnalysisMessage]]:
        members = heapq.nlargest(max_analyze_count, self._members.values(), key=lambda m: m.last_modified)
        debug('starting pass on', len(members), 'members, with a condition timeout of', max_condition_timeout)
        def timeout_for_position(pos: int) -> float:
            return max(0.3, (1.0 - pos / max_analyze_count) * max_condition_timeout)
        members_and_options = [(m, dataclasses.replace(self._options, per_condition_timeout=timeout_for_position(i)))
                               for (i, m) in enumerate(members)]
        for member, messages in self._pool.imap_unordered(worker_main, members_and_options):
            if messages:
                yield messages
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
        
        #module = module_for_file(curfile)
        for (name, member) in core.analyzable_members(module):
            qualname = module.__name__ + '.' + member.__name__ # type: ignore
            src = inspect.getsource(member)
            wm = WatchedMember(member, qualname, src)
            if qualname in members:
                any_changed |= members[qualname].consider_new(wm)
            else:
                members[qualname] = wm
        return any_changed

def watch(args: argparse.Namespace) -> int:
    options = core.AnalysisOptions()
    if not args.files:
        print('No files or directories given to watch', file=sys.stderr)
        return 1
    watcher = Watcher(options, args.files)
    watcher.check_all_files()
    restart = True
    while True:
        if restart:
            max_analyze_count = 5
            max_condition_timeout = 0.5
            restart = False
        else:
            max_analyze_count *= 2
            max_condition_timeout *= 2
        for messages in watcher.run_iteration(max_analyze_count, max_condition_timeout):
            if watcher.check_all_files():
                restart = True
                break
            for message in messages:
                line = short_describe_message(message)
                if line is not None:
                    print(line)

def short_describe_message(message: core.AnalysisMessage) -> Optional[str]:
    if message.state == core.MessageType.CANNOT_CONFIRM:
        return None
    desc = message.message
    if message.state == core.MessageType.POST_ERR:
        desc = 'Error while evaluating post condition: ' + desc
    return '{}:{}:{}:{}'.format(message.filename, message.line, 'error', desc)

def check(args: argparse.Namespace) -> int:
    any_errors = False
    for name in args.files:
        if name.endswith('.py'):
            _, name = extract_module_from_file(name)
        module = importlib.import_module(name)
        debug('Analyzing module ', module.__name__)
        for message in core.analyze_module(module, options):
            line = short_describe_message(message)
            if line is not None:
                debug(message.traceback)
                print(line)
                any_errors = True
    return 1 if any_errors else 0


if __name__ == '__main__':
    args = command_line_parser().parse_args()
    set_debug(args.verbose)
    options = process_level_options(args)
    if args.action == 'check':
        exitcode = check(args)
    elif args.action == 'watch':
        exitcode = watch(args)
    else:
        print(f'Unknown action: "{args.action}"', file=sys.stderr)
        exitcode = 1
    sys.exit(exitcode)
