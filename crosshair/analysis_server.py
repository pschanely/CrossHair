import os
from os.path import join
import json
import hashlib
import sys
import tempfile
import time
import shutil
import signal
import subprocess
from typing import *

import psutil  # type: ignore


def atomic_write(path: str, contents: str) -> None:
    containing, filename = os.path.split(path)
    tmpfile = tempfile.NamedTemporaryFile(dir=containing, prefix=filename,
                                          delete=False)
    with tmpfile:
        tmpfile.write(contents.encode('utf-8'))
    os.replace(tmpfile.name, path)


def read_input_files(path: str) -> Iterator[bytes]:
    if not os.path.exists(path):
        return
    for exfile in os.listdir(path):
        if exfile.startswith('.'):
            continue
        with open(join(path, exfile), 'rb') as fh:
            yield fh.read()


class Fuzzer:
    def __init__(self, homedir: str, args: List[str]) -> None:
        self.homedir = homedir
        self.args = args
        self.examplesfile = join(homedir, 'examples.json')
        self.inputdir = join(homedir, 'input')
        self.outputdir = join(homedir, 'output')
        self.crashdir = join(self.outputdir, 'crashes')
        self.coveragedir = join(self.outputdir, 'queue')
        self.statsfile = join(self.outputdir, 'fuzzer_stats')
        self.plotfile = join(self.outputdir, 'plot_data')
        self.proc: subprocess.Popen = None
        self.time_started: float = None

    def get_stats(self) -> Mapping[str, str]:
        if not os.path.exists(self.statsfile):
            return {}
        stats = {}
        with open(self.statsfile) as fh:
            for line in fh.readlines():
                if not line.strip():
                    continue
                key, val = line.split(':', 2)
                stats[key.strip()] = val.strip()
        return stats

    def get_plotdata(self) -> List[Dict[str, str]]:
        if not os.path.exists(self.plotfile):
            return []
        plot = []
        with open(self.plotfile) as fh:
            lines = list(fh.readlines())
            if lines:
                headings = lines[0][2:].split(', ')
                for line in lines[1:]:
                    plot.append(dict(zip(headings, line.split(', '))))
        return plot

    def start(self) -> None:
        if not os.path.exists(self.inputdir):
            os.mkdir(self.inputdir)
            with open(join(self.inputdir, 'seed.input'), 'wb') as fh:
                fh.write(b'\0' * 8)
        if os.path.exists(self.outputdir):
            shutil.rmtree(self.outputdir)
        os.mkdir(self.outputdir)
        cmd = ['py-afl-fuzz',
               '-m', '2000',
               '-o', self.outputdir,
               '-i', self.inputdir,
               '--',
               sys.executable, '-m', 'crosshair.fuzz_worker'] + self.args
        self.time_started = time.time()
        print(' '.join(c if c else "''" for c in cmd))
        sys.stdout.flush()

        # Detministic hash order makes fuzz iterations
        # behave more deterministically, which helps the fuzzer's stability(?):
        env = {**os.environ, 'PYTHONHASHSEED': '0'}

        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, env=env)

    def get_cur_examples(self) -> Set[bytes]:
        inputs = set(read_input_files(self.crashdir))
        inputs.update(set(read_input_files(self.inputdir)))
        inputs.update(set(read_input_files(self.coveragedir)))
        return inputs

    def is_running(self) -> bool:
        return self.proc.poll() is None

    def get_examples(self) -> Set[bytes]:
        examples = self.get_cur_examples()
        if os.path.exists(self.examplesfile):
            with open(self.examplesfile, 'r') as fh:
                examples.update(set(bytes.fromhex(ex) for ex in json.load(fh)))
        return examples

    def stop(self) -> None:
        cur_examples = [ex.hex() for ex in self.get_cur_examples()]
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None
        with open(self.examplesfile, 'w') as fh:
            json.dump(cur_examples, fh)


class Server:
    def __init__(self, homedir: str) -> None:
        self.homedir = homedir
        self.pidfile = join(homedir, 'pid')
        self.cmdfile = join(homedir, 'commands')
        self.fuzzdir = join(homedir, 'fuzz')
        self.fuzzer: Fuzzer = None
        self.cur_content_hash = ''
        self.quiet_until = 0.0

    def start(self) -> None:
        if not os.path.exists(self.fuzzdir):
            os.mkdir(self.fuzzdir)
        with open(self.pidfile, 'w') as fh:
            fh.write(str(os.getpid()))
        self.quiet_until = time.time() + 10

    def stop(self) -> None:
        if self.fuzzer is not None:
            self.fuzzer.stop()
            self.fuzzer = None

    def get_fuzzer(self, target: dict) -> Fuzzer:
        package, module, fns = (
            target['package'], target['module'], target['fns'])
        target_dir = os.path.join(self.fuzzdir, self.target_hash(target))
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        return Fuzzer(target_dir, [package, module] + fns)

    def running_stale(self) -> bool:
        if not self.fuzzer:
            return False
        stats = self.fuzzer.get_stats()
        if not stats:
            return False
        now, last_path = int(stats['last_update']), int(stats['last_path'])
        if last_path == 0:
            return False
        stale_secs = now - last_path
        # print('stale check', now, last_path, stale_secs)
        sys.stdout.flush()
        return stale_secs > 600

    def target_hash(self, target: dict) -> str:
        return target['module']

    def run_cycle(self) -> bool:
        cmds = self.get_commands()
        if self.running_stale():
            self.stop()
            return False
        target = cmds['target']
        if target is None:
            self.stop()
            return True
        hash_text = cmds['target_content_hash'].encode('utf-8')
        content_hash = hashlib.sha224(hash_text).hexdigest()[:32]
        expected_fuzzer = self.get_fuzzer(target)
        if self.fuzzer is not None:
            if self.fuzzer.is_running():
                if content_hash == self.cur_content_hash:
                    return True
                self.fuzzer.stop()
            print('afl subprocess failed; exit code ',
                  getattr(self.fuzzer.proc, 'returncode', None))
            self.fuzzer = None
        # do not restart in a tight loop unnecessarily:
        if (time.time() < self.quiet_until and
            content_hash == self.cur_content_hash):
            return True
        self.cur_content_hash = content_hash
        expected_fuzzer.start()
        self.fuzzer = expected_fuzzer
        return True

    def get_commands(self) -> dict:
        with open(self.cmdfile) as fh:
            return json.load(fh)

    def cleanup(self) -> None:
        shutil.rmtree(self.homedir)

    # Following members are intended to be used by external (client) processes
    def write_commands(self, cmds: dict) -> None:
        atomic_write(self.cmdfile, json.dumps(cmds))
        self.fuzzer = (self.get_fuzzer(cmds['target']) if cmds['target']
                       else None)

    def is_running(self) -> bool:
        if not os.path.exists(self.pidfile):
            return False
        with open(self.pidfile) as fh:
            pid = int(fh.read())
        if not psutil.pid_exists(pid):
            return False
        return 'crosshair' in ' '.join(psutil.Process(pid).cmdline())

    def get_status(self) -> dict:
        if not self.fuzzer:
            return {}
        return {
            'examples': self.fuzzer.get_examples(),
            'plotdata': self.fuzzer.get_plotdata(),
        }


def find_or_spawn_server() -> Server:
    _SERVER_HOME_PREFIX = 'crosshair_analysis_'
    tmp = tempfile.gettempdir()
    # print('scanning for temp files in ', tmp)
    server = None
    for tmpdir in os.listdir(tmp):
        if not tmpdir.startswith(_SERVER_HOME_PREFIX):
            continue
        curserver = Server(join(tmp, tmpdir))
        if server is None:
            server = curserver
        else:
            if curserver.is_running():
                curserver.stop()
            curserver.cleanup()
    if server is None:
        server = Server(tempfile.mkdtemp(prefix=_SERVER_HOME_PREFIX))
    if not server.is_running():
        homedir = server.homedir
        server.write_commands({'target': None})
        out = open(join(homedir, 'log.txt'), 'w')
        subprocess.Popen(
            [sys.executable, '-m', 'crosshair.analysis_server', homedir],
            stdout=out, stderr=out, stdin=subprocess.DEVNULL)
    return server


def main() -> None:
    try:
        os.setpgrp()
    except Exception:
        pass

    server = Server(sys.argv[1])
    server.start()
    try:
        keep_running = True
        while keep_running:
            keep_running = server.run_cycle()
            time.sleep(0.5)
    finally:
        server.stop()
        try:
            os.killpg(0, signal.SIGINT)
        except BaseException:
            pass


if __name__ == '__main__':
    main()
