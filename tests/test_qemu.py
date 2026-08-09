import re
import shlex
import socket
import time

from opencrl.backends.qemu import QemuWorld


class FakeSerialShell:
    """A fake serial channel that answers QemuWorld's marker protocol.

    exec_map: command -> output str, or command -> (output, exit_code).
    files:    path -> content (answers `cat -- <path>`).
    hang:     commands that never respond (to exercise the timeout path).
    Setup/drain lines (no marker) produce no output.
    """
    _MARK = re.compile(rb"; echo (--opencrl-[0-9a-f]+--)\$\?\1\n")

    def __init__(self, exec_map=None, files=None, hang=()):
        self._exec = exec_map or {}
        self._files = files or {}
        self._hang = set(hang)
        self._out = bytearray()     # queued bytes to hand back on recv
        self._line = bytearray()    # accumulates sent bytes until newline
        self._timeout = 0.0

    def sendall(self, data: bytes) -> None:
        self._line += data
        while b"\n" in self._line:
            i = self._line.index(b"\n")
            line = bytes(self._line[: i + 1])
            del self._line[: i + 1]
            self._handle(line)

    def _handle(self, line: bytes) -> None:
        m = self._MARK.search(line)
        if not m:
            return                              # setup/drain line: no response
        mark = m.group(1)
        command = line[: m.start()].decode()
        if command in self._hang:
            return                              # never answer -> timeout
        output, rc = self._respond(command)
        self._out += output.encode() + mark + str(rc).encode() + mark + b"\n"

    def _respond(self, command: str):
        if command.startswith("cat -- "):
            path = shlex.split(command)[2]
            if path in self._files:
                return self._files[path], 0
            return f"cat: {path}: No such file or directory", 1
        val = self._exec.get(command, ("", 0))
        return val if isinstance(val, tuple) else (val, 0)

    def recv(self, n: int) -> bytes:
        if self._out:
            chunk = bytes(self._out[:n])
            del self._out[:n]
            return chunk
        if self._timeout:
            time.sleep(self._timeout)           # honor settimeout like a real socket
        raise socket.timeout

    def settimeout(self, t) -> None:
        self._timeout = t or 0.0

    def close(self) -> None:
        pass


def test_exec_returns_command_output():
    w = QemuWorld(None, FakeSerialShell(exec_map={"id": "uid=1000(agent)"}))
    w._handshake()
    assert w.exec("id") == "uid=1000(agent)"


def test_read_file_hit_and_miss():
    w = QemuWorld(None, FakeSerialShell(files={"/flag": "CTF{ok}"}))
    w._handshake()
    assert w.read_file("/flag") == "CTF{ok}"
    assert w.read_file("/nope") is None


def test_exec_times_out_to_sentinel():
    w = QemuWorld(None, FakeSerialShell(hang={"sleep 9999"}), exec_timeout=0.2)
    w._handshake()
    assert w.exec("sleep 9999") == "[opencrl: command timed out after 0.2s]"


def test_exec_truncates_huge_output():
    w = QemuWorld(None, FakeSerialShell(exec_map={"big": "x" * 200_000}))
    w._handshake()
    out = w.exec("big")
    assert out.endswith("[opencrl: output truncated]")
    assert len(out) <= 100_000 + len("\n[opencrl: output truncated]")


def test_handshake_raises_if_shell_never_answers():
    w = QemuWorld(None, FakeSerialShell(hang={"true"}), exec_timeout=0.2)
    import pytest
    with pytest.raises(TimeoutError):
        w._handshake()


def test_read_until_fails_fast_on_eof():
    # If the serial closes (qemu exits/panics), recv() returns b"" repeatedly;
    # QemuWorld must fail fast instead of busy-spinning to the deadline.
    class _EOFChan:
        def sendall(self, data): pass
        def recv(self, n): return b""
        def settimeout(self, t): pass
        def close(self): pass

    w = QemuWorld(None, _EOFChan(), exec_timeout=5.0)
    assert w.exec("id") == "[opencrl: command timed out after 5.0s]"


import pytest
from opencrl.task import Caps
from opencrl.backend import resolve_backend
from opencrl.backends.qemu import Qemu


def test_argv_resolves_relative_paths_against_basedir():
    spec = {"kernel": "bzImage", "initrd": "rootfs.cpio.gz",
            "x-opencrl": {"basedir": "/base"}}
    argv = Qemu()._argv(spec, Caps(), "/tmp/s.sock")
    assert "-kernel" in argv and argv[argv.index("-kernel") + 1] == "/base/bzImage"
    assert "-initrd" in argv and argv[argv.index("-initrd") + 1] == "/base/rootfs.cpio.gz"
    assert argv[argv.index("-serial") + 1] == "unix:/tmp/s.sock,server,nowait"
    assert argv[argv.index("-machine") + 1] == "accel=hvf:kvm:tcg"
    assert "console=ttyS0" in argv[argv.index("-append") + 1]
    # no networking at all
    assert not any(a in ("-netdev", "-net", "-nic") for a in argv)


def test_argv_keeps_absolute_paths():
    spec = {"kernel": "/abs/bzImage", "initrd": "/abs/rootfs.cpio.gz"}
    argv = Qemu()._argv(spec, Caps(), "/tmp/s.sock")
    assert argv[argv.index("-kernel") + 1] == "/abs/bzImage"


def test_argv_honors_spec_and_ctor_resources():
    spec = {"kernel": "/k", "initrd": "/i", "cpus": 2, "memory": "512m",
            "append": "nokaslr"}
    argv = Qemu(cpus=1, memory="256m")._argv(spec, Caps(), "/s")
    assert argv[argv.index("-smp") + 1] == "2"
    assert argv[argv.index("-m") + 1] == "512m"
    assert argv[argv.index("-append") + 1] == "console=ttyS0 nokaslr"


def test_argv_needs_internet_raises():
    with pytest.raises(ValueError, match="needs_internet"):
        Qemu()._argv({"kernel": "/k", "initrd": "/i"}, Caps(needs_internet=True), "/s")


def test_argv_missing_kernel_raises():
    with pytest.raises(ValueError, match="requires 'kernel'"):
        Qemu()._argv({"initrd": "/i"}, Caps(), "/s")


def test_argv_relative_without_basedir_raises():
    with pytest.raises(ValueError, match="requires a world.yml"):
        Qemu()._argv({"kernel": "bzImage", "initrd": "i"}, Caps(), "/s")


def test_up_rejects_needs_internet_without_spawning(tmp_path):
    # The guard raises in _argv before Popen, so this never launches qemu.
    with pytest.raises(ValueError, match="needs_internet"):
        Qemu().up({"kernel": "/k", "initrd": "/i"}, Caps(needs_internet=True))


def test_qemu_registered():
    assert isinstance(resolve_backend("qemu"), Qemu)


def test_drain_is_bounded_on_a_never_idle_channel():
    # A channel that never idles must not make _drain (and thus up()) hang.
    class _NoisyChan:
        def sendall(self, data): pass
        def recv(self, n): return b"boot-noise "
        def settimeout(self, t): pass
        def close(self): pass

    w = QemuWorld(None, _NoisyChan(), exec_timeout=0.3)
    start = time.monotonic()
    w._drain()                       # returns at the deadline; must not hang
    assert time.monotonic() - start < 3.0


def test_connect_surfaces_qemu_stderr_on_early_exit():
    import io
    from opencrl.backends.qemu import _connect

    class _DeadProc:
        returncode = 1
        stderr = io.StringIO("qemu: could not load kernel 'bad'")
        def poll(self):
            return 1

    with pytest.raises(RuntimeError, match="could not load kernel"):
        _connect("/nonexistent/opencrl.sock", _DeadProc(),
                 deadline=time.monotonic() + 1.0)
