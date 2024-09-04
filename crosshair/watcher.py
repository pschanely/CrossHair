import base64
import binascii
import multiprocessing
import os
import pickle
import queue
import subprocess
import sys
import threading
import time
import traceback
import zlib
from pathlib import Path
from queue import Queue
from typing import (
    Any,
    Counter,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from crosshair.auditwall import engage_auditwall, opened_auditwall
from crosshair.core_and_libs import (
    AnalysisMessage,
    MessageType,
    analyze_module,
    run_checkables,
)
from crosshair.fnutil import NotFound, walk_paths
from crosshair.options import AnalysisOptionSet
from crosshair.util import (
    CrossHairInternal,
    ErrorDuringImport,
    debug,
    load_file,
    set_debug,
)

# Use "spawn" in stead of fork() because we've already imported the code we're watching:
multiproc_spawn = multiprocessing.get_context("spawn")


def mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


WorkItemInput = Tuple[Path, AnalysisOptionSet, float]  # (file, opts, deadline)
WorkItemOutput = Tuple[Path, Counter[str], List[AnalysisMessage]]


def serialize(obj: object) -> str:
    return str(base64.b64encode(zlib.compress(pickle.dumps(obj))), "ascii")


def deserialize(data: Union[bytes, str]) -> Any:
    try:
        return pickle.loads(zlib.decompress(base64.b64decode(data)))
    except binascii.Error:
        debug(f"Unable to deserialize this data: {data!r}")
        raise


def import_error_msg(err: ErrorDuringImport) -> AnalysisMessage:
    cause = err.__cause__ if err.__cause__ else err
    tb = cause.__traceback__
    if tb:
        filename, line = tb.tb_frame.f_code.co_filename, tb.tb_lineno
        tbstr = "".join(traceback.format_tb(tb))
    else:
        filename, line = "<unknown>", 0
        tbstr = ""
    return AnalysisMessage(MessageType.IMPORT_ERR, str(cause), filename, line, 0, tbstr)


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


class PoolWorkerShell(threading.Thread):
    def __init__(
        self, input_item: WorkItemInput, results: "queue.Queue[WorkItemOutput]"
    ):
        self.input_item = input_item
        self.results = results
        self.proc: Optional[subprocess.Popen] = None
        super().__init__()

    def run(self) -> None:
        encoded_input = serialize(self.input_item)
        worker_args = [
            sys.executable,
            "-c",
            f"import crosshair.watcher; crosshair.watcher.pool_worker_main()",
            encoded_input,
        ]
        self.proc = subprocess.Popen(
            worker_args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        (stdout, _stderr) = self.proc.communicate(b"")
        if stdout:
            last_line = stdout.splitlines()[-1]  # (in case of spurious print()s)
            self.results.put(deserialize(last_line))


def pool_worker_main() -> None:
    item: WorkItemInput = deserialize(sys.argv[-1])
    filename = item[0]
    try:
        if hasattr(os, "nice"):  # analysis should run at a low priority
            # Note that the following "type: ignore" is ONLY required for mypy on
            # Windows, where the nice function does not exist:
            os.nice(10)  # type: ignore
        set_debug(False)
        engage_auditwall()
        (stats, messages) = pool_worker_process_item(item)
        output: WorkItemOutput = (filename, stats, messages)
        print(serialize(output))
    except BaseException as e:
        raise CrossHairInternal("Worker failed while analyzing " + str(filename)) from e


class Pool:
    _workers: List[Tuple[PoolWorkerShell, WorkItemInput]]
    _work: List[WorkItemInput]
    _results: "Queue[WorkItemOutput]"
    _max_processes: int

    def __init__(self, max_processes: int) -> None:
        self._workers = []
        self._work = []
        self._results = Queue()
        self._max_processes = max_processes

    def _spawn_workers(self):
        work_list = self._work
        workers = self._workers
        while work_list and len(self._workers) < self._max_processes:
            work_item = work_list.pop()
            # NOTE: We are martialling data manually.
            # Earlier versions used multiprocessing and Queues, but
            # multiprocessing.Process is incompatible with pygls on windows
            # (something with the async blocking on stdin, which must remain open
            # in the child).

            thread = PoolWorkerShell(work_item, self._results)
            workers.append((thread, work_item))
            thread.start()

    def _prune_workers(self, curtime: float) -> None:
        for worker, item in self._workers:
            (_, _, deadline) = item
            if worker.is_alive() and curtime > deadline and worker.proc is not None:
                debug("Killing worker over deadline", worker)
                worker.proc.terminate()
                time.sleep(0.5)
                if worker.is_alive():
                    worker.proc.kill()
                    worker.join()
        self._workers = [(w, i) for w, i in self._workers if w.is_alive()]

    def terminate(self) -> None:
        self._prune_workers(float("+inf"))
        self._work = []

    def garden_workers(self) -> None:
        self._prune_workers(time.time())
        self._spawn_workers()

    def is_working(self) -> bool:
        return bool(self._workers or self._work)

    def submit(self, item: WorkItemInput) -> None:
        self._work.append(item)

    def has_result(self) -> bool:
        return not self._results.empty()

    def get_result(self, timeout: float) -> Optional[WorkItemOutput]:
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None


class Watcher:
    _paths: Set[Path]
    _pool: Pool
    _modtimes: Dict[Path, float]
    _options: AnalysisOptionSet
    _next_file_check: float = 0.0
    _change_flag: bool = False
    _stop_flag: bool = False

    def __init__(
        self, files: Iterable[Path], options: AnalysisOptionSet = AnalysisOptionSet()
    ):
        self._paths = set(files)
        self._pool = self.startpool()
        self._modtimes = {}
        self._options = options

    def shutdown(self):
        self._stop_flag = True

    def update_paths(self, paths: Iterable[Path]):
        self._paths = set(paths)

    def startpool(self) -> Pool:
        return Pool(multiprocessing.cpu_count() - 1)

    def run_iteration(
        self, max_uninteresting_iterations=5
    ) -> Iterator[Tuple[Counter[str], List[AnalysisMessage]]]:
        debug(
            f"starting pass with max_uninteresting_iterations={max_uninteresting_iterations}"
        )
        debug("Files:", self._modtimes.keys())
        pool = self._pool
        for filename, _ in sorted(self._modtimes.items(), key=lambda pair: -pair[1]):
            worker_timeout = max(
                10.0, max_uninteresting_iterations * 100.0
            )  # TODO: times 100? is that right?
            iter_options = AnalysisOptionSet(
                max_uninteresting_iterations=max_uninteresting_iterations,
            )
            options = self._options.overlay(iter_options)
            pool.submit((filename, options, time.time() + worker_timeout))

        pool.garden_workers()
        if not pool.is_working():
            # Unusual case where there is nothing to do:
            time.sleep(1.5)
            self.handle_periodic()  # (keep checking for changes!)
            yield (Counter(), [])
            return
        while pool.is_working():
            result = pool.get_result(timeout=1.0)
            if result is not None:
                (_, counters, messages) = result
                yield (counters, messages)
                if pool.has_result():
                    continue
            if self.handle_periodic():
                yield (Counter(), [])  # to break the parent from waiting
                return
            pool.garden_workers()
        debug("Worker pool tasks complete")

    def handle_periodic(self) -> bool:
        if self._stop_flag:
            debug("Aborting iteration on shutdown request")
            self._pool.terminate()
            return True
        if time.time() >= self._next_file_check:
            self._next_file_check = time.time() + 1.0
            if self.check_changed():
                self._change_flag = True
                debug("Aborting iteration on change detection")
                self._pool.terminate()
                self._pool = self.startpool()
                return True
        return False

    def check_changed(self) -> bool:
        unchecked_modtimes = self._modtimes.copy()
        changed = False
        for curfile in walk_paths(self._paths, ignore_missing=True):
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
