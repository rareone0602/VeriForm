import os
import re
import time
import json
import ctypes
import signal
import resource
import tempfile
import traceback
import threading
import subprocess
import multiprocessing as mp
from pprint import pprint

import numpy as np

from .ast_parser import lean4_parser
from ..workers import ProcessScheduler
from ..utils import AttrDict


HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = os.path.join(os.path.dirname(__file__), '..', '..', 'lean')
DEFAULT_REPL_BINARY = './.lake/packages/REPL/.lake/build/bin/repl'

# Persistent REPL prelude: Mathlib is imported once at worker startup and
# every subsequent TC reuses that environment via `"env": <id>`.
_PRELUDE = "import Mathlib"
_IMPORT_LINE_RE = re.compile(r'^\s*import\s+\S')


def _strip_leading_imports(code: str) -> str:
    """Drop the file-header `import ...` lines so the body can be submitted
    against a pre-imported env. The REPL rejects `import` outside of file
    headers, so we must not forward them."""
    lines = code.split('\n')
    i = 0
    n = len(lines)
    while i < n:
        s = lines[i].strip()
        if not s or s.startswith('--'):
            i += 1
            continue
        if _IMPORT_LINE_RE.match(lines[i]):
            i += 1
            continue
        break
    return '\n'.join(lines[i:])


class PersistentRepl:
    """Long-lived `lake env <repl>` subprocess with `import Mathlib` already
    in env 0. One REPL per worker for the worker's lifetime; Mathlib is paid
    once at startup (~30-60 s) rather than per TC call."""

    def __init__(self, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE,
                 repl_binary=DEFAULT_REPL_BINARY, warmup_timeout=600):
        self.lake_path = lake_path
        self.lean_workspace = lean_workspace
        self.repl_binary = repl_binary
        self.warmup_timeout = warmup_timeout
        self.proc = None
        self.prelude_env = None
        self._spawn()

    def _spawn(self):
        # `stdbuf -oL` is mandatory: Lean's `IO.println` uses glibc stdio which
        # block-buffers when stdout is a pipe. Without this, the REPL's
        # response JSON sits in the buffer until the buffer fills or the REPL
        # exits, deadlocking our `readline()` loop. With `-oL` glibc flushes
        # on every newline, which matches the REPL's two-newlines-after-each-
        # response protocol observed in tests.
        # `start_new_session=True` puts the wrapper and its grandchild Lean
        # `repl` in their own process group, so `_kill_proc` can SIGKILL the
        # whole group — see `_kill_proc` for why a plain `self.proc.kill()`
        # leaves the repl orphaned and deadlocks `readline()`.
        self.proc = subprocess.Popen(
            ["stdbuf", "-oL", self.lake_path, "env", self.repl_binary],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.lean_workspace, text=True, bufsize=1,
            start_new_session=True,
        )
        # Warm Mathlib once. Response is JSON terminated by a blank line.
        env, _ = self._send_raw(_PRELUDE, env=None, timeout=self.warmup_timeout)
        if env is None:
            self.kill()
            raise RuntimeError("REPL Mathlib prelude failed to return an env id")
        self.prelude_env = env

    def _kill_proc(self):
        """SIGKILL the whole process group. `self.proc` is the `stdbuf/lake
        env` wrapper; the actual Lean `repl` runs as a grandchild. A plain
        `self.proc.kill()` reaps only the wrapper and leaves the `repl`
        orphaned — it keeps the stdout pipe open, so our `readline()` loop
        never sees EOF and the worker deadlocks. Killing the process group
        (created via `start_new_session=True`) takes both down together."""
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # group already gone, or pgid unavailable — fall back to a
            # direct kill so we at least reap the wrapper.
            try:
                self.proc.kill()
            except Exception:
                pass
        # Reap the wrapper so it doesn't linger as a zombie. `wait()` is
        # safe to call here (Timer thread) alongside `poll()` elsewhere —
        # Popen serialises waitpid internally. The grandchild `repl` is
        # killed by the killpg above; it gets reparented to init and
        # reaped there.
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def _send_raw(self, code, env, timeout):
        cmd = {"cmd": code}
        if env is not None:
            cmd["env"] = env
        msg = json.dumps(cmd, ensure_ascii=False) + "\n\n"
        self.proc.stdin.write(msg)
        self.proc.stdin.flush()
        # Read until blank-line terminator, enforcing a wall-clock deadline.
        lines = []
        deadline = time.time() + timeout
        killer = threading.Timer(timeout, self._kill_proc)
        killer.start()
        try:
            while True:
                line = self.proc.stdout.readline()
                if line == '':  # EOF — REPL died
                    raise EOFError("REPL closed stdout")
                if line.strip() == '':
                    if lines:
                        break
                    else:
                        # leading blank from previous response; skip
                        continue
                lines.append(line)
                if time.time() > deadline:
                    raise TimeoutError(f"REPL response timeout after {timeout}s")
        finally:
            killer.cancel()
        raw = ''.join(lines)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"REPL produced non-JSON response: {raw[:500]!r}") from e
        return result.get('env'), result

    def query(self, code, timeout, allTactics=False, ast=False, premises=False, tactics=False):
        """Submit `code` reusing the pre-imported Mathlib env. Returns the
        dict in the shape that verify_lean4_file used to return."""
        start_time = time.time()
        cmd = {"cmd": _strip_leading_imports(code), "env": self.prelude_env}
        if allTactics: cmd["allTactics"] = True
        if ast:        cmd["ast"] = True
        if premises:   cmd["premises"] = True
        if tactics:    cmd["tactics"] = True
        msg = json.dumps(cmd, ensure_ascii=False) + "\n\n"
        try:
            self.proc.stdin.write(msg)
            self.proc.stdin.flush()
            lines = []
            killer = threading.Timer(timeout, self._kill_proc)
            killer.start()
            try:
                while True:
                    line = self.proc.stdout.readline()
                    if line == '':
                        raise EOFError("REPL closed stdout")
                    if line.strip() == '':
                        if lines:
                            break
                        else:
                            continue
                    lines.append(line)
            finally:
                killer.cancel()
            raw_response = json.loads(''.join(lines))
            ast_results = lean4_parser(code, raw_response['ast']) if raw_response.get('ast') else {}
            result = {
                "sorries": raw_response.get('sorries', []),
                "tactics": raw_response.get('tactics', []),
                "errors": [m for m in raw_response.get('messages', []) if m['severity'] == 'error'],
                "warnings": [m for m in raw_response.get('messages', []) if m['severity'] == 'warning'],
                "infos": [m for m in raw_response.get('messages', []) if m['severity'] == 'info'],
                "system_messages": '',
                "system_errors": None,
                "ast": ast_results,
                "verified_code": code,
            }
            result['pass'] = not result['errors']
            result['complete'] = result['pass'] and not result['sorries'] and not any(
                "declaration uses 'sorry'" in w['data'] or 'failed' in w['data']
                for w in result['warnings']
            )
        except Exception:
            result = {
                "pass": False,
                "complete": False,
                "system_errors": traceback.format_exc(),
                "system_messages": '',
            }
        result['verify_time'] = time.time() - start_time
        return result

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def kill(self):
        if self.proc is not None:
            self._kill_proc()
            self.proc = None


def verify_lean4_file(code, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, last_env=None, verbose=False, timeout=300, allTactics=False, ast=False, premises=False, tactics=False):
    command = dict(cmd=code, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises)
    if last_env is not None:
        command.update(env=last_env)
    message_str = json.dumps(command, ensure_ascii=False)
    if verbose:
        print(message_str)
    start_time = time.time()
    system_messages = ''
    try:
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            # `lake exe repl` triggers an incremental build-replay step that
            # is broken on this cluster (Lean writes fail with ENOENT and
            # clang crashes compiling REPL/Lean/InfoTree.c). Invoke the
            # already-built repl binary directly via `lake env`, which only
            # sets up LEAN_PATH/LD_LIBRARY_PATH and skips the rebuild.
            outputs = subprocess.run(
                [lake_path, "env", "./.lake/packages/REPL/.lake/build/bin/repl"],
                stdin=temp_file, capture_output=True, text=True,
                cwd=lean_workspace, timeout=timeout,
            )
        result = json.loads(outputs.stdout)
        ast_results = lean4_parser(code, result['ast']) if 'ast' in result and result['ast'] else {}
        result = {
            "sorries" : result.get('sorries', []), 
            "tactics" : result.get('tactics', []),
            "errors" : [m for m in result.get('messages', []) if m['severity'] == 'error'],
            "warnings" : [m for m in result.get('messages', []) if m['severity'] == 'warning'],
            "infos" : [m for m in result.get('messages', []) if m['severity'] == 'info'],
            "system_messages" : system_messages,
            "system_errors" : None,
            "ast" : ast_results,
            "verified_code" : code,
        }
        result['pass'] = not result['errors']
        result['complete'] = result['pass'] and not result['sorries'] and not any("declaration uses 'sorry'" in warning['data'] or 'failed' in warning['data'] for warning in result['warnings'])
    except:
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages
        }
    result['verify_time'] = time.time() - start_time
    return result


class Lean4ServerProcess(mp.Process):
    def __init__(self, idx, task_queue, request_statuses, lock, extra_args=AttrDict()):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.extra_args = extra_args

        self.timeout = extra_args.get('timeout', 300)
        self.memory_limit = extra_args.get('memory_limit', -1)
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)
    
    def run(self):
        if self.memory_limit > 0:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit * (1000 ** 3), self.memory_limit * (1000 ** 3))
            )
        # One persistent REPL per worker: Mathlib import is paid once here,
        # then every task reuses env_id. Cold-import cost goes from ~30-60 s
        # per TC to ~0 (only the warmup pays it).
        repl = PersistentRepl(warmup_timeout=max(self.timeout, 600))
        try:
            while True:
                inputs = self.task_queue.get()
                if inputs is None:
                    break
                for _, request_id, task in inputs:
                    if isinstance(task, str):
                        task = dict(code=task)
                    task_timeout = task.get('timeout', self.timeout)
                    code = task['code']
                    kwargs = {k: task.get(k, False)
                              for k in ('allTactics', 'ast', 'premises', 'tactics')}
                    if not repl.alive():
                        # Mathlib re-import is expensive but rare. Only
                        # happens after a crash on a previous task.
                        try:
                            repl = PersistentRepl(warmup_timeout=max(self.timeout, 600))
                        except Exception:
                            result = {
                                "pass": False, "complete": False,
                                "system_errors": traceback.format_exc(),
                                "system_messages": '',
                                "verify_time": 0.0,
                            }
                            with self.lock:
                                self.request_statuses[request_id] = result
                                self.last_output_time.value = time.time()
                                self.complete_count.value += 1
                            continue
                    result = repl.query(code, timeout=task_timeout, **kwargs)
                    # If REPL died mid-task, restart it before serving the
                    # next request — the current task already returned an
                    # error result via the query() exception handler.
                    if not repl.alive():
                        try:
                            repl = PersistentRepl(warmup_timeout=max(self.timeout, 600))
                        except Exception:
                            pass
                    with self.lock:
                        self.request_statuses[request_id] = result
                        self.last_output_time.value = time.time()
                        self.complete_count.value += 1
        finally:
            repl.kill()


class Lean4ServerScheduler(ProcessScheduler):
    def __init__(self, max_concurrent_requests=64, timeout=300, memory_limit=-1, name='verifier'):
        super().__init__(batch_size=1, name=name)
        
        self.processes = [
            Lean4ServerProcess(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                extra_args=AttrDict(
                    timeout=timeout,
                    memory_limit=memory_limit,
                )
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        print(f'Complete launching {len(self.processes)} LeanServerProcesses')

        self.timeout = timeout
        # NOTE: the old `killall repl --older-than=Xs` monitor is incompatible
        # with the persistent-REPL model — those long-lived REPLs are exactly
        # what we *want* to keep alive. Per-task timeouts are now enforced
        # inside PersistentRepl.query() via a threading.Timer on the proc.

    def close(self):
        super().close()
        for p in self.processes:
            p.join()
        print(f'All {len(self.processes)} LeanServerProcesses stopped')


if __name__ == '__main__':
    code = open('mathlib4/.lake/packages/REPL/test/aime_1983_p9.code.in').read()
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=300, memory_limit=10, name='verifier')
    request_id_list = lean4_scheduler.submit_all_request([dict(code=code, ast=True, tactics=True)])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()
    pprint(outputs_list)