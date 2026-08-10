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


import os
from opencrl.task import Caps


def test_kit_denies_egress_by_default():
    kit = sbx.Sandbox()._kit({"image": "ubuntu:24.04"}, Caps())
    assert kit["sandbox"]["image"] == "ubuntu:24.04"
    assert kit["sandbox"]["network"] == "deny"


def test_kit_allows_egress_with_needs_internet():
    kit = sbx.Sandbox()._kit({"image": "x"}, Caps(needs_internet=True))
    assert kit["sandbox"]["network"] == "allow"


def test_missing_image_raises():
    with pytest.raises(ValueError):
        sbx.Sandbox()._kit({}, Caps())


def test_up_creates_sandbox_and_runs_setup_in_order(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(sbx, "_run", rec)
    s = sbx.Sandbox()
    w = s.up({"image": "ubuntu:24.04", "setup": ["echo hi", "id"]}, Caps())
    try:
        assert isinstance(w, sbx.SandboxWorld)
        assert rec.calls[0][1] == "create"
        setup = [c[-1] for c in rec.calls if c[1] == "exec"]
        assert setup == ["echo hi", "id"]        # run once, in order
    finally:
        s.down(w)


def test_up_raises_and_cleans_up_on_create_failure(monkeypatch):
    rec = Recorder(routes={"create": cp(returncode=1, stderr="boom")})
    monkeypatch.setattr(sbx, "_run", rec)
    with pytest.raises(RuntimeError, match="boom"):
        sbx.Sandbox().up({"image": "x"}, Caps())
    assert ["sbx", "rm", "--force"] == [c[:3] for c in rec.calls if c[1] == "rm"][0][:3]


def test_up_raises_on_setup_failure(monkeypatch):
    def run(args, timeout=None):
        return cp(returncode=1, stderr="setup boom") if args[1] == "exec" else cp()
    monkeypatch.setattr(sbx, "_run", run)
    with pytest.raises(RuntimeError, match="setup boom"):
        sbx.Sandbox().up({"image": "x", "setup": ["bad"]}, Caps())


def test_relative_files_in_inline_world_raises(monkeypatch):
    monkeypatch.setattr(sbx, "_run", Recorder())
    with pytest.raises(ValueError):
        sbx.Sandbox().up({"image": "x", "files": "payload"}, Caps())   # no basedir


def test_down_removes_sandbox_and_workdir(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(sbx, "_run", rec)
    s = sbx.Sandbox()
    w = s.up({"image": "x"}, Caps())
    workdir = w._workdir
    assert os.path.isdir(workdir)
    s.down(w)
    assert not os.path.exists(workdir)
    assert ["sbx", "rm", "--force", w.name] in rec.calls
