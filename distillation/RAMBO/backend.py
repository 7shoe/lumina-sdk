"""A persistent, isolated Julia subprocess; hard wall deadlines per request."""
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading

DEFAULT_ENV = Path('/lus/flare/projects/FRAME-IDP/siebenschuh/envs/rambo_julia')
PROJECT = Path(__file__).with_name('julia')


class JuliaBackend:
    def __init__(self, log, startup_seconds=600):
        # prefer an explicit executable, then PATH, then the managed installation
        executable = os.environ.get('RAMBO_JULIA') or shutil.which('julia') or str(DEFAULT_ENV / 'julia-1.10.10/bin/julia')
        if not Path(executable).is_file():
            raise FileNotFoundError('Run distillation/RAMBO/aux/setup_julia.sh or set RAMBO_JULIA')
        
        # env hygene: limit CPU threads and clear inherited library paths
        env = dict(os.environ, JULIA_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   LD_LIBRARY_PATH='', JULIA_DEPOT_PATH=os.environ.get('RAMBO_JULIA_DEPOT', str(DEFAULT_ENV / 'depot')))
        # keep one worker alive 
        self.process = subprocess.Popen([executable, '--startup-file=no', '--project=' + str(PROJECT),
                                         str(PROJECT / 'worker.jl')], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=env, start_new_session=True)
        self.lines = queue.Queue()
        def reader():
            # route protocol responses to the caller
            with Path(log).open('a') as stream:
                for line in self.process.stdout:
                    if line.startswith('RAMBO_RESULT '):
                        self.lines.put(line[len('RAMBO_RESULT '):])
                    else:
                        stream.write(line)
                        stream.flush()
            self.lines.put(None)
        self.thread = threading.Thread(target=reader, daemon=True)
        self.thread.start()
        try:
            # allow initial Julia startup/compilation and verify the worker responds
            self.versions = self.request(dict(stage='versions'), timeout=startup_seconds)
            if 'julia' not in self.versions:
                raise RuntimeError(self.versions)
        except BaseException:
            self.close()
            raise

    def request(self, request, timeout):
        # Exchange newline-delimited JSON
        self.process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
        self.process.stdin.flush()
        try:
            # response deadline starts after the request has been written
            response = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError('Julia request exceeded its wall-clock deadline; inspect backend.log')
        if response is None:
            raise RuntimeError('Julia exited unexpectedly; inspect backend.log')
        return json.loads(response)

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Lustre I/O can delay even a killed worker's exit. Reap it
                    # later without turning a saved pause into a solver failure.
                    threading.Thread(target=self.process.wait, daemon=True).start()
        self.thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
