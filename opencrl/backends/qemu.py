"""QEMU microVM backend for kernel-target tasks.

Boots a task-supplied kernel + initramfs and drives an unprivileged serial
shell with a marker protocol. No guest networking, so egress is impossible
by construction.
"""
from __future__ import annotations

import os
import re
import shlex
import socket
import subprocess
import tempfile
import time
import uuid

from opencrl.backend import register_backend
from opencrl.task import Caps

# Cap on any single exec/read_file's returned output (mirrors the docker backend).
_MAX_OUTPUT = 100_000


class QemuWorld:
    agent = "guest"                 # single VM; the `host` arg is ignored

    def __init__(self, proc, chan, sock_dir: str = "", exec_timeout: float = 120.0):
        self.proc = proc            # the qemu subprocess.Popen (None in unit tests)
        self._chan = chan           # duck-typed: sendall / recv / settimeout / close
        self._sock_dir = sock_dir   # temp dir holding the serial socket (removed in down())
        self.exec_timeout = exec_timeout

    # --- serial plumbing -------------------------------------------------
    def _send(self, text: str) -> None:
        self._chan.sendall(text.encode())

    def _read_until(self, pattern: "re.Pattern[bytes]") -> "re.Match[bytes]":
        """Read until `pattern` matches; raise TimeoutError at exec_timeout."""
        buf = bytearray()
        deadline = time.monotonic() + self.exec_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            self._chan.settimeout(remaining)
            try:
                chunk = self._chan.recv(4096)
            except (socket.timeout, TimeoutError):
                raise TimeoutError
            if not chunk:                       # EOF: serial closed (qemu exited/panicked)
                raise TimeoutError              # fail fast, don't busy-spin to the deadline
            buf += chunk
            m = pattern.search(buf)
            if m:
                return m

    def _drain(self, quiet: float = 0.3) -> None:
        """Discard buffered bytes (boot noise, echoed setup) until it goes quiet."""
        self._chan.settimeout(quiet)
        while True:
            try:
                if not self._chan.recv(4096):
                    return
            except (socket.timeout, TimeoutError):
                return

    def _handshake(self) -> None:
        # Turn off terminal echo + prompt so only real command output returns,
        # drain the boot noise, then confirm the shell is live via the exec path.
        self._send("\n")
        self._send("stty -echo 2>/dev/null; PS1=''\n")
        self._drain()
        self._run_marked("true")    # raises TimeoutError if the shell never answers

    def _run_marked(self, command: str) -> tuple[str, int]:
        mark = f"--opencrl-{uuid.uuid4().hex}--"
        self._send(f"{command}; echo {mark}$?{mark}\n")
        mb = mark.encode()
        pattern = re.compile(re.escape(mb) + rb"(-?\d+)" + re.escape(mb))
        m = self._read_until(pattern)
        out = bytes(m.string[: m.start()]).decode(errors="replace")
        if out.endswith("\n"):
            out = out[:-1]
        return out, int(m.group(1))

    # --- World protocol --------------------------------------------------
    def exec(self, command: str, host: str | None = None) -> str:
        try:
            out, _rc = self._run_marked(command)
        except TimeoutError:
            return f"[opencrl: command timed out after {self.exec_timeout}s]"
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + "\n[opencrl: output truncated]"
        return out

    def read_file(self, path: str, host: str | None = None) -> str | None:
        try:
            out, rc = self._run_marked(f"cat -- {shlex.quote(path)}")
        except TimeoutError:
            return None
        if rc != 0:
            return None
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + "\n[opencrl: output truncated]"
        return out
