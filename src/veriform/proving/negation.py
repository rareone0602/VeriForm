"""Lean-side theorem negation via the bundled `lake exe negate` subprocess.

Replaces the regex-based ``DeepSeekProver.get_refute_theorem`` (CLAUDE.md
improvement #2). The Lean implementation lives at
``src/veriform/proving/lean_server/lean/Negate.lean``.

Wire protocol (see Negate.lean::main):
  - Server prints ``READY\\n`` once Mathlib is loaded.
  - Each request: ``<mode>\\t<base64-src>\\n`` (mode ∈ {``strong``, ``full``}).
  - Each reply: ``OK\\t<base64-result>\\n`` or ``ERR\\t<base64-msg>\\n``.
  - Closing stdin terminates the server cleanly.

Usage::

    with LeanNegator() as neg:
        out = neg.negate("theorem foo : 2 + 2 = 4 := by sorry", mode="full")
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

NegationMode = Literal["strong", "full"]

DEFAULT_LAKE_PATH = Path(os.path.expanduser("~/.elan/bin/lake"))
DEFAULT_LEAN_WORKSPACE = (
    Path(__file__).resolve().parent / "lean_server" / "lean"
)

_READY_TOKEN = "READY"


class LeanNegatorError(RuntimeError):
    """Raised when the Lean negator subprocess fails or returns ERR."""


class LeanNegator(AbstractContextManager):
    """Persistent ``lake exe negate`` subprocess.

    Spawn once per benchmark run (Mathlib startup ~6s). Thread-safe — guarded
    by a single lock; calls block other callers.
    """

    def __init__(
        self,
        lake_path: Path | str = DEFAULT_LAKE_PATH,
        lean_workspace: Path | str = DEFAULT_LEAN_WORKSPACE,
        startup_timeout: float = 120.0,
        request_timeout: float = 30.0,
    ) -> None:
        self.lake_path = Path(lake_path)
        self.lean_workspace = Path(lean_workspace)
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.lake_path.exists():
            raise LeanNegatorError(
                f"lake not found at {self.lake_path}. Set lake_path or install elan."
            )
        if not (self.lean_workspace / "lakefile.toml").exists():
            raise LeanNegatorError(
                f"no lakefile.toml at {self.lean_workspace}"
            )
        # Use `lake env` so LEAN_PATH/LD_LIBRARY_PATH point at the bundled
        # Mathlib build inside the project's .lake/packages/.
        self._proc = subprocess.Popen(
            [str(self.lake_path), "env", "./.lake/build/bin/negate"],
            cwd=str(self.lean_workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Wait for READY
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc.stdout is None:
                raise LeanNegatorError("subprocess has no stdout")
            line = self._proc.stdout.readline()
            if not line:
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise LeanNegatorError(
                    f"negate exited before READY; stderr=\n{err}"
                )
            if line.strip() == _READY_TOKEN:
                return
        raise LeanNegatorError(
            f"negate did not emit READY within {self.startup_timeout}s"
        )

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        finally:
            self._proc = None

    def __enter__(self) -> "LeanNegator":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- request ---------------------------------------------------------

    def negate(self, theorem_src: str, mode: NegationMode = "full") -> str:
        if mode not in ("strong", "full"):
            raise ValueError(f"mode must be 'strong' or 'full', got {mode!r}")
        if self._proc is None or self._proc.poll() is not None:
            self.start()
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        b64 = base64.b64encode(theorem_src.encode("utf-8")).decode("ascii")
        request = f"{mode}\t{b64}\n"
        with self._lock:
            try:
                self._proc.stdin.write(request)
                self._proc.stdin.flush()
            except BrokenPipeError as e:
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise LeanNegatorError(
                    f"broken pipe writing request; stderr=\n{err}"
                ) from e
            line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise LeanNegatorError(
                f"negate closed before reply; stderr=\n{err}"
            )
        line = line.rstrip("\n")
        try:
            tag, payload = line.split("\t", 1)
        except ValueError:
            raise LeanNegatorError(f"malformed reply line: {line!r}")
        decoded = base64.b64decode(payload).decode("utf-8")
        if tag == "OK":
            return decoded
        if tag == "ERR":
            raise LeanNegatorError(decoded)
        raise LeanNegatorError(f"unknown reply tag {tag!r}: {decoded}")
