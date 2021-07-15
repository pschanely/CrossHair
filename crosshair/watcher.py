import dataclasses
import multiprocessing
import multiprocessing.queues
import os
from pathlib import Path
import queue
import sys
import time
import signal
from typing import *

from crosshair.auditwall import engage_auditwall
from crosshair.auditwall import opened_auditwall
from crosshair.core_and_libs import analyze_module
from crosshair.core_and_libs import run_checkables
from crosshair.core_and_libs import AnalysisMessage
from crosshair.core_and_libs import MessageType
from crosshair.fnutil import walk_paths
from crosshair.options import AnalysisOptionSet, AnalysisOptions
from crosshair.options import DEFAULT_OPTIONS
from crosshair.fnutil import NotFound
from crosshair.util import debug
from crosshair.util import load_file
from crosshair.util import set_debug
from crosshair.util import CrosshairInternal
from crosshair.util import ErrorDuringImport


# Use "spawn" in stead of fork() because we've already imported the code we're watching:
multiproc_spawn = multiprocessing.get_context("spawn")


def mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
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


WorkItemInput = Tuple[Path, AnalysisOptionSet, float]  # (file, opts, deadline)
WorkItemOutput = Tuple[WatchedMember, Counter[str], List[AnalysisMessage]]


def import_error_msg(err: ErrorDuringImport) -> AnalysisMessage:
    cause = err.__cause__ if err.__cause__ else err
    tb = cause.__traceback__
    if tb:
        filename, line = tb.tb_frame.f_code.co_filename, tb.tb_lineno
    else:
        filename, line = "<unknown>", 0
    return AnalysisMessage(MessageType.IMPORT_ERR, str(cause), filename, line, 0, "")


def pool_worker_process_item(
    item: WorkItemInput,
) -> Tuple[Counter[str], List[AnalysisMessage]]:
    filename, options, deadline = item
    stats: Counter[str] = Counter()
    options.stats = stats
    try:
        module = load_file(str(filename))
    except NotFound as e:
        debug(f'Not analyzing "{filename}" because sub-module import failed: {e}')
        return (stats, [])
    except ErrorDuringImport as e:
        debug(f'Not analyzing "{filename}" because import failed: {e}')
        return (stats, [import_error_msg(e)])
    messages = run_checkables(analyze_module(module, options))
    return (stats, messages)


def pool_worker_main(item: WorkItemInput, output: multiprocessing.queues.Queue) -> None:
    filename = item[0]
    try:
        # TODO figure out a more reliable way to suppress this. Redirect output?
        # Ignore ctrl-c in workers to reduce noisy tracebacks (the parent will kill us):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if hasattr(os, "nice"):  # analysis should run at a low priority
            os.nice(10)
        set_debug(False)
        engage_auditwall()
        (stats, messages) = pool_worker_process_item(item)
        output.put((filename, stats, messages))
    except BaseException as e:
        raise CrosshairInternal("Worker failed while analyzing " + str(filename)) from e


class Pool:
    _workers: List[Tuple[multiprocessing.Process, WorkItemInput]]
    _work: List[WorkItemInput]
    _results: multiprocessing.queues.Queue
    _max_processes: int

    def __init__(self, max_processes: int) -> None:
        self._workers = []
        self._work = []
        self._results = multiproc_spawn.Queue()
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
    _paths: Set[Path]
    _pool: Pool
    _modtimes: Dict[Path, float]
    _options: AnalysisOptionSet
    _next_file_check: float = 0.0
    _change_flag: bool = False

    def __init__(
        self, files: Iterable[Path], options: AnalysisOptionSet = AnalysisOptionSet()
    ):
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
            if time.time() >= self._next_file_check:
                self._next_file_check = time.time() + 1.0
                if self.check_changed():
                    self._change_flag = True
                    debug("Aborting iteration on change detection")
                    pool.terminate()
                    yield (Counter(), [])  # to break the parent from waiting
                    self._pool = self.startpool()
                    return
            pool.garden_workers()
        debug("Worker pool tasks complete")

    def check_changed(self) -> bool:
        unchecked_modtimes = self._modtimes.copy()
        changed = False
        for curfile in walk_paths(self._paths):
            cur_mtime = mtime(curfile)
            if cur_mtime is None:
                # Unlikely; race condition on an interleaved file delete
                continue
            if cur_mtime == unchecked_modtimes.pop(curfile, None):
                continue
            changed = True
            self._modtimes[curfile] = cur_mtime
        if unchecked_modtimes:
            # Files known but not found; something was deleted
            changed = True
            for delfile in unchecked_modtimes.keys():
                del self._modtimes[delfile]
        return changed
