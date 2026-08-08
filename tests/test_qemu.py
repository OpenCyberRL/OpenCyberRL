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
