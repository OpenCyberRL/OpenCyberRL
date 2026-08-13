"""Tests for opencrl.modules — module repo management helpers."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import opencrl.modules as m


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def test_modules_dir_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    assert m.modules_dir() == tmp_path / "modules" / "opencyberrl-modules"


def test_modules_dir_default(monkeypatch):
    monkeypatch.delenv("OPENCRL_HOME", raising=False)
    monkeypatch.setattr("os.path.expanduser", lambda p: p.replace("~", "/home/user"))
    assert m.modules_dir() == Path("/home/user/.opencrl/modules/opencyberrl-modules")


def test_active_file_path(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    assert m.active_file() == tmp_path / "active"


# ---------------------------------------------------------------------------
# active_modules
# ---------------------------------------------------------------------------

def test_active_modules_empty_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    assert m.active_modules() == []


def test_active_modules_reads_multiple_entries(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    m._write_active(["web_sqli", "privesc", "lateral"])
    assert m.active_modules() == ["web_sqli", "privesc", "lateral"]


def test_active_modules_skips_blank_lines(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    m.active_file().write_text("web_sqli\n\nprivesc\n\n")
    assert m.active_modules() == ["web_sqli", "privesc"]


# ---------------------------------------------------------------------------
# activate / deactivate
# ---------------------------------------------------------------------------

def test_activate_module_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    m.activate_module("web_sqli")
    assert m.active_modules() == ["web_sqli"]
    assert m.active_file().exists()


def test_activate_module_no_duplicate(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    m.activate_module("web_sqli")
    m.activate_module("web_sqli")
    assert m.active_modules() == ["web_sqli"]


def test_activate_module_preserves_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    m.activate_module("web_sqli")
    m.activate_module("privesc")
    assert m.active_modules() == ["web_sqli", "privesc"]


def test_deactivate_module_removes(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    m._write_active(["web_sqli", "privesc"])
    m.deactivate_module("web_sqli")
    assert m.active_modules() == ["privesc"]


def test_deactivate_module_noop_when_not_active(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    m._write_active(["web_sqli"])
    m.deactivate_module("privesc")
    assert m.active_modules() == ["web_sqli"]


def test_deactivate_module_empty_file(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    m.deactivate_module("web_sqli")
    assert m.active_modules() == []


# ---------------------------------------------------------------------------
# is_cloned
# ---------------------------------------------------------------------------

def test_is_cloned_false_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    assert m.is_cloned() is False


def test_is_cloned_true_when_git_dir_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    (m.modules_dir() / ".git").mkdir(parents=True)
    assert m.is_cloned() is True


# ---------------------------------------------------------------------------
# clone / update (subprocess mocked)
# ---------------------------------------------------------------------------

def test_clone_modules_repo_calls_git(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    m.clone_modules_repo()
    assert calls == [
        ["git", "clone", "--depth", "1",
         "https://github.com/OpenCyberRL/opencyberrl-modules.git",
         str(m.modules_dir())]
    ]
    # parent dirs created
    assert m.modules_dir().parent.exists()


def test_update_modules_repo_calls_git(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    m.update_modules_repo()
    assert calls == [
        ["git", "-C", str(m.modules_dir()), "pull", "--ff-only"]
    ]


# ---------------------------------------------------------------------------
# list_available_modules
# ---------------------------------------------------------------------------

def test_list_available_modules_empty_when_not_cloned(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    assert m.list_available_modules() == []


def test_list_available_modules_lists_subdirs_with_task_py(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    md = m.modules_dir()
    (md / "web_sqli" / "sub").mkdir(parents=True)
    (md / "web_sqli" / "sub" / "task.py").write_text("")
    (md / "privesc" / "sub").mkdir(parents=True)
    (md / "privesc" / "sub" / "task.py").write_text("")
    assert m.list_available_modules() == ["privesc", "web_sqli"]


def test_list_available_modules_ignores_dotdirs(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    md = m.modules_dir()
    (md / ".hidden" / "sub").mkdir(parents=True)
    (md / ".hidden" / "sub" / "task.py").write_text("")
    (md / "visible" / "sub").mkdir(parents=True)
    (md / "visible" / "sub" / "task.py").write_text("")
    assert m.list_available_modules() == ["visible"]


def test_list_available_modules_skips_dirs_without_task_py(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    md = m.modules_dir()
    (md / "empty").mkdir(parents=True)
    (md / "has_task" / "sub").mkdir(parents=True)
    (md / "has_task" / "sub" / "task.py").write_text("")
    assert m.list_available_modules() == ["has_task"]


def test_list_available_modules_sorted(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    md = m.modules_dir()
    for name in ["zeta", "alpha", "middle"]:
        (md / name / "sub").mkdir(parents=True)
        (md / name / "sub" / "task.py").write_text("")
    assert m.list_available_modules() == ["alpha", "middle", "zeta"]


# ---------------------------------------------------------------------------
# active_module_paths
# ---------------------------------------------------------------------------

def test_active_module_paths_filters_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    md = m.modules_dir()
    (md / "web_sqli" / "sub").mkdir(parents=True)
    (md / "web_sqli" / "sub" / "task.py").write_text("")
    m._write_active(["web_sqli", "nonexistent"])
    paths = m.active_module_paths()
    assert paths == [md / "web_sqli"]


def test_active_module_paths_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    assert m.active_module_paths() == []
