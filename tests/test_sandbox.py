import subprocess
import pytest
from opencrl.backends import sandbox as sbx


def cp(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


class Recorder:
    """Fake _run: records every call's argv; routes a canned result by the
    sbx subcommand (args[1]). A routed Exception is raised."""
    def __init__(self, routes=None):
        self.calls = []
        self.routes = routes or {}

    def __call__(self, args, timeout=None):
        self.calls.append(list(args))
        r = self.routes.get(args[1] if len(args) > 1 else "", cp())
        if isinstance(r, Exception):
            raise r
        return r


def test_exec_runs_sbx_exec_and_returns_output(monkeypatch):
    rec = Recorder(routes={"exec": cp(stdout="uid=0(root)\n")})
    monkeypatch.setattr(sbx, "_run", rec)
    w = sbx.SandboxWorld("opencrl-abc", "/tmp/wd")
    assert w.exec("id") == "uid=0(root)\n"
    assert rec.calls[-1] == ["sbx", "exec", "opencrl-abc", "sh", "-c", "id"]


def test_exec_timeout_returns_sentinel(monkeypatch):
    def boom(args, timeout=None):
        raise subprocess.TimeoutExpired(args, timeout)
    monkeypatch.setattr(sbx, "_run", boom)
    w = sbx.SandboxWorld("n", "/tmp/wd", exec_timeout=1.0)
    assert w.exec("sleep 5") == "[opencrl: command timed out after 1.0s]"


def test_exec_caps_large_output(monkeypatch):
    monkeypatch.setattr(sbx, "_run", Recorder(routes={"exec": cp(stdout="x" * 200_000)}))
    out = sbx.SandboxWorld("n", "/tmp/wd").exec("yes")
    assert out.endswith("\n[opencrl: output truncated]")
    assert len(out) <= 100_000 + len("\n[opencrl: output truncated]")


def test_read_file_returns_contents(monkeypatch):
    rec = Recorder(routes={"cp": cp(stdout="CTF{x}")})
    monkeypatch.setattr(sbx, "_run", rec)
    w = sbx.SandboxWorld("n", "/tmp/wd")
    assert w.read_file("/root/flag") == "CTF{x}"
    assert rec.calls[-1] == ["sbx", "cp", "n:/root/flag", "-"]


def test_read_file_missing_returns_none(monkeypatch):
    monkeypatch.setattr(sbx, "_run", Recorder(routes={"cp": cp(returncode=1, stderr="no file")}))
    assert sbx.SandboxWorld("n", "/tmp/wd").read_file("/nope") is None
