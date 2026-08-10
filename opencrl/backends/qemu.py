"""QEMU microVM backend for kernel-target tasks.

Boots a task-supplied kernel + initramfs and drives an unprivileged serial
shell with a marker protocol. The default NIC is disabled (`-nic none`), so
the guest has no network and egress is impossible by construction.
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
# Extra trailing bytes kept while scanning so the end marker is never split when
# _read_until trims an over-long buffer (a continuous producer like `yes`).
_MARKER_TAIL = 256
# Brief bound for resyncing the shell after a timeout (not a 2nd exec_timeout).
_RESYNC_TIMEOUT = 2.0


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

    def _read_until(self, pattern: "re.Pattern[bytes]") -> tuple["re.Match[bytes]", bool]:
        """Read until `pattern` matches; raise TimeoutError at exec_timeout.

        Returns the match and whether the buffer was trimmed — trimmed output is
        incomplete, so the caller must report truncation regardless of length.
        """
        buf = bytearray()
        trimmed = False
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
                return m, trimmed
            # Bound memory against an endless producer (e.g. `yes`): keep the
            # first _MAX_OUTPUT bytes (all we'd ever return) plus a rolling tail
            # big enough that the end marker is never split by the trim.
            if len(buf) > _MAX_OUTPUT + _MARKER_TAIL:
                del buf[_MAX_OUTPUT:-_MARKER_TAIL]
                trimmed = True

    def _drain(self, quiet: float = 0.3, bound: float | None = None) -> None:
        """Discard buffered bytes (boot noise, echoed setup) until it goes quiet.

        Bounded (default: exec_timeout) so a kernel that never stops printing to
        the console can't hang up(); callers already past a deadline pass a
        shorter bound so draining can't run a second full timeout.
        """
        deadline = time.monotonic() + (self.exec_timeout if bound is None else bound)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            # Cap each read by whichever is smaller — the idle window or the
            # time left — so a short bound stays a real wall-clock limit even
            # when it is under `quiet`.
            self._chan.settimeout(min(quiet, remaining))
            try:
                if not self._chan.recv(4096):
                    return
            except (socket.timeout, TimeoutError):
                return

    def _handshake(self) -> None:
        # Turn off terminal echo + prompt so only real command output returns,
        # drain the boot noise, then confirm the shell is live via the exec path.
        self._send("\n")
        self._send("stty -echo -onlcr 2>/dev/null; PS1=''\n")
        self._drain()
        self._run_marked("true")    # raises TimeoutError if the shell never answers

    def _run_marked(self, command: str) -> tuple[str, int]:
        mark = f"--opencrl-{uuid.uuid4().hex}--"
        # Run via `sh -c` so any command frames cleanly: an empty command or one
        # already ending in `;` would otherwise make `<cmd>; echo ...` a syntax
        # error and the marker would never print. Each exec is its own child
        # shell, matching the docker backend's per-exec statelessness.
        self._send(f"sh -c {shlex.quote(command)}; echo {mark}$?{mark}\n")
        mb = mark.encode()
        pattern = re.compile(re.escape(mb) + rb"(-?\d+)" + re.escape(mb))
        m, trimmed = self._read_until(pattern)
        out = bytes(m.string[: m.start()]).decode(errors="replace")
        if out.endswith("\r\n"):        # a real serial tty terminates lines with CRLF
            out = out[:-2]
        elif out.endswith("\n"):
            out = out[:-1]
        # Truncate in one place so a trimmed (over-long) buffer is always
        # reported, even when multibyte output has fewer chars than bytes.
        if trimmed or len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + "\n[opencrl: output truncated]"
        return out, int(m.group(1))

    def _interrupt(self) -> None:
        """SIGINT the stuck foreground command so the one serial shell resyncs.

        A timed-out command keeps running in the sole persistent shell; without
        this, every later command would queue behind it and also time out. The
        drain uses a short bound, not another full exec_timeout, since we are
        already past the deadline.
        """
        try:
            self._send("\x03")          # Ctrl-C
            self._drain(bound=min(_RESYNC_TIMEOUT, self.exec_timeout))
        except OSError:
            pass

    # --- World protocol --------------------------------------------------
    def exec(self, command: str, host: str | None = None) -> str:
        try:
            out, _rc = self._run_marked(command)
        except TimeoutError:
            self._interrupt()
            return f"[opencrl: command timed out after {self.exec_timeout}s]"
        return out

    def read_file(self, path: str, host: str | None = None) -> str | None:
        try:
            out, rc = self._run_marked(f"cat -- {shlex.quote(path)}")
        except TimeoutError:
            self._interrupt()
            return None
        if rc != 0:
            return None
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
            "-nic", "none",         # disable QEMU's default user-mode NIC (no egress)
            "-serial", f"unix:{serial_sock},server,nowait",
            "-machine", f"accel={self.accel}",
        ]

    def up(self, spec: dict, caps: Caps) -> QemuWorld:
        spec = spec or {}
        sock_dir = tempfile.mkdtemp(prefix="opencrl-qemu-")
        serial_sock = os.path.join(sock_dir, "serial.sock")
        stderr_path = os.path.join(sock_dir, "qemu.stderr")
        try:
            argv = self._argv(spec, caps, serial_sock)   # raises on bad spec first
            # stderr to a file, not a PIPE: an unread PIPE can fill and block qemu.
            with open(stderr_path, "wb") as errf:
                proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=errf)
        except Exception:
            _teardown(None, None, sock_dir)
            raise
        chan = None
        try:
            chan = _connect(serial_sock, proc,
                            time.monotonic() + self.boot_timeout, stderr_path)
            world = QemuWorld(proc, chan, sock_dir, exec_timeout=self.exec_timeout)
            world._handshake()
            return world
        except Exception:
            _teardown(proc, chan, sock_dir)
            raise

    def down(self, world: QemuWorld) -> None:
        _teardown(world.proc, world._chan, world._sock_dir)


def _connect(sock_path: str, proc, deadline: float, stderr_path: str = ""):
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"qemu exited before serial was ready (code {proc.returncode}):\n"
                f"{_read_stderr(stderr_path)}")
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(sock_path)
            return s
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("qemu serial socket did not become ready")


def _read_stderr(path: str) -> str:
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


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
            proc.wait()             # reap the killed child so it isn't left a zombie
    if sock_dir:
        shutil.rmtree(sock_dir, ignore_errors=True)


register_backend("qemu", lambda: Qemu())
