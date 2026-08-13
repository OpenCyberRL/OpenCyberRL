"""Tests for opencrl CLI — install/uninstall/update/list/new commands."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import opencrl.cli as cli
from opencrl.cli import main


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_fake_modules_repo(home: Path, module_name: str = "examples") -> Path:
    """Create a fake modules repo structure under OPENCRL_HOME.

    Returns the modules_dir path. Creates:
      home/modules/opencyberrl-modules/.git/
      home/modules/opencyberrl-modules/<module>/<sub>/task.py
    """
    md = home / "modules" / "opencyberrl-modules"
    (md / ".git").mkdir(parents=True, exist_ok=True)
    mod_dir = md / module_name
    sub = mod_dir / "demo" / "task.py"
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_text("# fake task\n")
    return md
def _clear_registry():
    """Clear the task registry so tests don't leak state."""
    sys.modules["opencrl.task"]._REGISTRY.clear()

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def test_cli_install_no_arg_lists_available(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    _make_fake_modules_repo(tmp_path, "examples")
    _clear_registry()
    rc = main(["install"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "examples" in out


def test_cli_install_module_activates(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    _make_fake_modules_repo(tmp_path, "examples")
    _clear_registry()
    rc = main(["install", "examples"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Activated" in out
    import opencrl.modules as m
    assert "examples" in m.active_modules()


def test_cli_install_nonexistent_module_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    _make_fake_modules_repo(tmp_path, "examples")
    _clear_registry()
    rc = main(["install", "nonexistent"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not found" in out


# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------

def test_cli_uninstall_deactivates(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    _make_fake_modules_repo(tmp_path, "examples")
    import opencrl.modules as m
    m.activate_module("examples")
    assert "examples" in m.active_modules()
    _clear_registry()
    rc = main(["uninstall", "examples"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Deactivated" in out
    assert "examples" not in m.active_modules()


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

def test_cli_update_pulls_repo(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    _make_fake_modules_repo(tmp_path, "examples")
    _clear_registry()
    with patch("opencrl.modules.update_modules_repo") as mock_update:
        rc = main(["update"])
        out = capsys.readouterr().out
    assert rc == 0
    assert "Updated" in out
    mock_update.assert_called_once()


def test_cli_update_not_cloned_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    # Don't create the .git dir — is_cloned() returns False
    _clear_registry()
    rc = main(["update"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "No modules repo" in out


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def test_cli_list_shows_tasks(monkeypatch, tmp_path, capsys):
    _clear_registry()
    # Create a fake tasks dir with a task.py that registers a task
    tasks_dir = tmp_path / "tasks" / "test_task"
    tasks_dir.mkdir(parents=True)
    tasks_dir.joinpath("task.py").write_text(
        "from opencrl import task, Task, shell, flag, Caps\n"
        "@task\n"
        "def test_task() -> Task:\n"
        "    return Task(goal='test', reward=flag('CTF{x}'), tools=(shell,), caps=Caps(offensive=True))\n"
    )
    rc = main(["--path", str(tmp_path / "tasks"), "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "test_task" in out
    _clear_registry()


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------

def test_cli_new_still_works(monkeypatch, tmp_path, capsys):
    _clear_registry()
    monkeypatch.chdir(tmp_path)
    rc = main(["new", "mytask"])
    out = capsys.readouterr().out
    assert rc == 0
    assert (tmp_path / "tasks" / "mytask" / "task.py").exists()
    assert (tmp_path / "tasks" / "mytask" / "world.yml").exists()
    assert "Created" in out


# ---------------------------------------------------------------------------
# run / eval removed
# ---------------------------------------------------------------------------

def test_cli_no_run_command(monkeypatch, tmp_path):
    _clear_registry()
    import pytest
    with pytest.raises(SystemExit):
        main(["run", "web_sqli"])


def test_cli_no_eval_command(monkeypatch, tmp_path):
    _clear_registry()
    import pytest
    with pytest.raises(SystemExit):
        main(["eval", "web_sqli"])


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def test_cli_info_shows_task_details(monkeypatch, tmp_path, capsys):
    _clear_registry()
    tasks_dir = tmp_path / "tasks" / "test_task"
    tasks_dir.mkdir(parents=True)
    tasks_dir.joinpath("task.py").write_text(
        "from opencrl import task, Task, shell, flag, Caps\n"
        "@task\n"
        "def test_task() -> Task:\n"
        "    return Task(goal='Read the flag and report it.', "
        "reward=flag('CTF{x}'), tools=(shell,), caps=Caps(offensive=True), "
        "max_steps=10)\n"
    )
    rc = main(["--path", str(tmp_path / "tasks"), "info", "test_task"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "test_task" in out
    assert "Read the flag" in out
    assert "offensive" in out.lower() or "Offensive" in out
    _clear_registry()


def test_cli_info_nonexistent_task_fails(monkeypatch, tmp_path, capsys):
    _clear_registry()
    rc = main(["info", "nonexistent"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not found" in out
    _clear_registry()


# ---------------------------------------------------------------------------
# uninstall validation
# ---------------------------------------------------------------------------

def test_cli_uninstall_non_active_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    _clear_registry()
    rc = main(["uninstall", "nonactive"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not active" in out


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

def test_cli_version_flag(monkeypatch, tmp_path, capsys):
    _clear_registry()
    rc = main(["--version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "opencrl" in out
    _clear_registry()
