"""QEMU microVM backend for kernel-target tasks.

Boots a task-supplied kernel + initramfs and drives an unprivileged serial
shell with a marker protocol. No guest networking, so egress is impossible
by construction.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
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
        """Discard buffered bytes (boot noise, echoed setup) until it goes quiet.

        Bounded by exec_timeout so a kernel that never stops printing to the
        console can't hang up().
        """
        self._chan.settimeout(quiet)
        deadline = time.monotonic() + self.exec_timeout
        while time.monotonic() < deadline:
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


class Qemu:
    def __init__(self, cpus: int = 1, memory: str = "256m", exec_timeout: float = 120.0,
                 accel: str = "hvf:kvm:tcg", qemu_bin: str = "qemu-system-x86_64",
                 boot_timeout: float = 60.0):
        self.cpus = cpus
        self.memory = memory
        self.exec_timeout = exec_timeout
        self.accel = accel
        self.qemu_bin = qemu_bin
        self.boot_timeout = boot_timeout

    def _resolve(self, path, basedir: str, what: str) -> str:
        if not path:
            raise ValueError(f"opencrl: qemu world requires '{what}'")
        if os.path.isabs(path):
            return path
        if not basedir:
            raise ValueError(
                f"opencrl: relative '{what}' path {path!r} requires a world.yml "
                f"file (no basedir); use an absolute path in an inline world")
        return os.path.join(basedir, path)

    def _argv(self, spec: dict, caps: Caps, serial_sock: str) -> list[str]:
        if caps.needs_internet:
            raise ValueError(
                "opencrl: the qemu backend has no networking; "
                "needs_internet=True is not supported")
        basedir = (spec.get("x-opencrl") or {}).get("basedir", "")
        kernel = self._resolve(spec.get("kernel"), basedir, "kernel")
        initrd = self._resolve(spec.get("initrd"), basedir, "initrd")
        append = "console=ttyS0"
        if spec.get("append"):
            append += " " + str(spec["append"])
        return [
            self.qemu_bin,
            "-kernel", kernel,
            "-initrd", initrd,
            "-append", append,
            "-m", str(spec.get("memory", self.memory)),
            "-smp", str(spec.get("cpus", self.cpus)),
            "-nographic", "-monitor", "none", "-no-reboot",
            "-serial", f"unix:{serial_sock},server,nowait",
            "-machine", f"accel={self.accel}",
        ]

    def up(self, spec: dict, caps: Caps) -> QemuWorld:
        spec = spec or {}
        sock_dir = tempfile.mkdtemp(prefix="opencrl-qemu-")
        serial_sock = os.path.join(sock_dir, "serial.sock")
        try:
            argv = self._argv(spec, caps, serial_sock)   # raises on bad spec first
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, text=True)
        except Exception:
            _teardown(None, None, sock_dir)
            raise
        chan = None
        try:
            chan = _connect(serial_sock, proc, time.monotonic() + self.boot_timeout)
            world = QemuWorld(proc, chan, sock_dir, exec_timeout=self.exec_timeout)
            world._handshake()
            return world
        except Exception:
            _teardown(proc, chan, sock_dir)
            raise

    def down(self, world: QemuWorld) -> None:
        _teardown(world.proc, world._chan, world._sock_dir)


def _connect(sock_path: str, proc, deadline: float):
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr is not None else ""
            raise RuntimeError(
                f"qemu exited before serial was ready (code {proc.returncode}):\n{err}")
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(sock_path)
            return s
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("qemu serial socket did not become ready")


def _teardown(proc, chan, sock_dir: str) -> None:
    if chan is not None:
        try:
            chan.close()
        except OSError:
            pass
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if sock_dir:
        shutil.rmtree(sock_dir, ignore_errors=True)


register_backend("qemu", lambda: Qemu())
