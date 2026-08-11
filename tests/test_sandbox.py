import os
import subprocess

import pytest

from opencrl.backends import sandbox as sbx
from opencrl.task import Caps


def cp(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


class Recorder:
    """Fake _run: records every call's argv; routes a canned result by the
    `docker sandbox` subcommand (args[2]). A routed Exception is raised."""
    def __init__(self, routes=None):
        self.calls = []
        self.routes = routes or {}

    def __call__(self, args, timeout=None):
        self.calls.append(list(args))
        r = self.routes.get(args[2] if len(args) > 2 else "", cp())
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def _stub_allowlist(monkeypatch):
    # up() reads the daemon proxy allowlist to block it; stub it so unit tests
    # never touch ~/.sandboxd. A test can re-monkeypatch to None to exercise
    # the fail-closed path.
    monkeypatch.setattr(sbx, "_proxy_allowlist",
                        lambda: ["github.com:443", "pypi.org:443"])


# --- SandboxWorld -------------------------------------------------------

def test_exec_runs_docker_sandbox_exec_and_returns_output(monkeypatch):
    rec = Recorder(routes={"exec": cp(stdout="uid=0(root)\n")})
    monkeypatch.setattr(sbx, "_run", rec)
    w = sbx.SandboxWorld("opencrl-abc", "/tmp/ws")
    assert w.exec("id") == "uid=0(root)\n"
    assert rec.calls[-1] == ["docker", "sandbox", "exec", "opencrl-abc", "sh", "-c", "id"]


def test_exec_timeout_returns_sentinel(monkeypatch):
    def boom(args, timeout=None):
        raise subprocess.TimeoutExpired(args, timeout)
    monkeypatch.setattr(sbx, "_run", boom)
    w = sbx.SandboxWorld("n", "/tmp/ws", exec_timeout=1.0)
    assert w.exec("sleep 5") == "[opencrl: command timed out after 1.0s]"


def test_exec_caps_large_output(monkeypatch):
    monkeypatch.setattr(sbx, "_run", Recorder(routes={"exec": cp(stdout="x" * 200_000)}))
    out = sbx.SandboxWorld("n", "/tmp/ws").exec("yes")
    assert out.endswith("\n[opencrl: output truncated]")
    assert len(out) <= 100_000 + len("\n[opencrl: output truncated]")


def test_read_file_uses_exec_cat_and_returns_contents(monkeypatch):
    rec = Recorder(routes={"exec": cp(stdout="CTF{x}")})
    monkeypatch.setattr(sbx, "_run", rec)
    w = sbx.SandboxWorld("n", "/tmp/ws")
    assert w.read_file("/root/flag") == "CTF{x}"
    assert rec.calls[-1] == ["docker", "sandbox", "exec", "n", "cat", "--", "/root/flag"]


def test_read_file_missing_returns_none(monkeypatch):
    monkeypatch.setattr(sbx, "_run", Recorder(routes={"exec": cp(returncode=1, stderr="no file")}))
    assert sbx.SandboxWorld("n", "/tmp/ws").read_file("/nope") is None


def test_read_file_timeout_returns_none(monkeypatch):
    def boom(args, timeout=None):
        raise subprocess.TimeoutExpired(args, timeout)
    monkeypatch.setattr(sbx, "_run", boom)
    w = sbx.SandboxWorld("n", "/tmp/ws", exec_timeout=1.0)
    assert w.read_file("/root/flag") is None


def test_read_file_caps_large_output(monkeypatch):
    monkeypatch.setattr(sbx, "_run", Recorder(routes={"exec": cp(stdout="x" * 200_000)}))
    out = sbx.SandboxWorld("n", "/tmp/ws").read_file("/big")
    assert out.endswith("\n[opencrl: output truncated]")
    assert len(out) <= 100_000 + len("\n[opencrl: output truncated]")


# --- Sandbox.up / down --------------------------------------------------

def test_missing_image_raises(monkeypatch):
    monkeypatch.setattr(sbx, "_run", Recorder())
    with pytest.raises(ValueError, match="requires 'image'"):
        sbx.Sandbox().up({}, Caps())


def test_up_creates_with_template_and_agent_then_runs_setup_in_order(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(sbx, "_run", rec)
    s = sbx.Sandbox()
    w = s.up({"image": "ubuntu:24.04", "setup": ["echo hi", "id"]}, Caps())
    try:
        create = next(c for c in rec.calls if c[2] == "create")
        assert create[create.index("--template") + 1] == "ubuntu:24.04"
        assert "codex" in create                       # the default create-agent
        setup = [c[-1] for c in rec.calls if c[2] == "exec"]
        assert setup == ["echo hi", "id"]              # run once, in order
    finally:
        s.down(w)


def test_up_denies_egress_and_blocks_the_allowlist_by_default(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(sbx, "_run", rec)
    s = sbx.Sandbox()
    w = s.up({"image": "x"}, Caps())
    try:
        proxy = next(c for c in rec.calls if c[2] == "network")
        # deny the default policy AND block each built-in allowlist entry
        assert proxy == ["docker", "sandbox", "network", "proxy", w.name,
                         "--policy", "deny",
                         "--block-host", "github.com:443",
                         "--block-host", "pypi.org:443"]
    finally:
        s.down(w)


def test_up_fails_closed_when_allowlist_unreadable(monkeypatch):
    monkeypatch.setattr(sbx, "_run", Recorder())
    monkeypatch.setattr(sbx, "_proxy_allowlist", lambda: None)
    with pytest.raises(RuntimeError, match="cannot read"):
        sbx.Sandbox().up({"image": "x"}, Caps())


def test_up_allows_egress_with_needs_internet(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(sbx, "_run", rec)
    s = sbx.Sandbox()
    w = s.up({"image": "x"}, Caps(needs_internet=True))
    try:
        assert not any(c[2] == "network" for c in rec.calls)   # no deny applied
    finally:
        s.down(w)


def test_up_copies_files_into_workspace(monkeypatch, tmp_path):
    src = tmp_path / "payload"
    src.mkdir()
    (src / "vuln").write_text("payload")
    rec = Recorder()
    monkeypatch.setattr(sbx, "_run", rec)
    s = sbx.Sandbox()
    w = s.up({"image": "x", "files": str(src)}, Caps())
    try:
        create = next(c for c in rec.calls if c[2] == "create")
        workspace = create[-1]                          # the mounted workspace
        assert os.path.isfile(os.path.join(workspace, "vuln"))
    finally:
        s.down(w)


def test_up_preserves_symlinks_rather_than_inlining_host_files(monkeypatch, tmp_path):
    # A link in `files` escaping the tree must be copied as a link (dangling in
    # the guest), NOT dereferenced into the host file's contents.
    src = tmp_path / "payload"
    src.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("HOST-SECRET")
    os.symlink(secret, src / "link")
    rec = Recorder()
    monkeypatch.setattr(sbx, "_run", rec)
    s = sbx.Sandbox()
    w = s.up({"image": "x", "files": str(src)}, Caps())
    try:
        workspace = next(c for c in rec.calls if c[2] == "create")[-1]
        assert os.path.islink(os.path.join(workspace, "link"))   # link, not the secret
    finally:
        s.down(w)


def test_up_raises_and_cleans_up_on_create_failure(monkeypatch):
    rec = Recorder(routes={"create": cp(returncode=1, stderr="boom")})
    monkeypatch.setattr(sbx, "_run", rec)
    with pytest.raises(RuntimeError, match="boom"):
        sbx.Sandbox().up({"image": "x"}, Caps())
    rm = [c for c in rec.calls if c[2] == "rm"]
    assert rm and rm[0][:3] == ["docker", "sandbox", "rm"]


def test_up_raises_on_setup_failure(monkeypatch):
    def run(args, timeout=None):
        return cp(returncode=1, stderr="setup boom") if args[2] == "exec" else cp()
    monkeypatch.setattr(sbx, "_run", run)
    with pytest.raises(RuntimeError, match="setup boom"):
        sbx.Sandbox().up({"image": "x", "setup": ["bad"]}, Caps())


def test_relative_files_in_inline_world_raises(monkeypatch):
    monkeypatch.setattr(sbx, "_run", Recorder())
    with pytest.raises(ValueError):
        sbx.Sandbox().up({"image": "x", "files": "payload"}, Caps())   # no basedir


def test_down_removes_sandbox_and_workspace(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(sbx, "_run", rec)
    s = sbx.Sandbox()
    w = s.up({"image": "x"}, Caps())
    workspace = w._workspace
    assert os.path.isdir(workspace)
    s.down(w)
    assert not os.path.exists(workspace)
    assert ["docker", "sandbox", "rm", w.name] in rec.calls


def test_down_warns_when_rm_fails(monkeypatch, capsys):
    rec = Recorder(routes={"rm": cp(returncode=1, stderr="daemon down")})
    monkeypatch.setattr(sbx, "_run", rec)
    s = sbx.Sandbox()
    w = s.up({"image": "x"}, Caps())
    s.down(w)
    err = capsys.readouterr().err
    assert "warning" in err and "may still be running" in err
