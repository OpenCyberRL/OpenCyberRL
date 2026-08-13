"""Tests for opencrl.task.discover — local + active module auto-discovery."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from opencrl.task import _REGISTRY, discover, list_tasks


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_FAKE_TASK = (
    "from opencrl.task import Task, task\n"
    "\n"
    "@task\n"
    "def {name}() -> Task:\n"
    "    return Task(goal='g', reward=lambda s: 1.0)\n"
)


def _clear_registry() -> None:
    """Wipe the task registry and any dynamically imported task modules."""
    _REGISTRY.clear()
    for mod_name in list(sys.modules):
        if mod_name.startswith("opencrl_tasks."):
            del sys.modules[mod_name]


@pytest.fixture(autouse=True)
def clean_registry():
    _clear_registry()
    yield
    _clear_registry()


def _write_task(parent: Path, name: str) -> None:
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "task.py").write_text(_FAKE_TASK.format(name=name))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_discover_default_searches_local(tmp_path, monkeypatch):
    """discover(path=None) finds tasks in local ./tasks/."""
    monkeypatch.chdir(tmp_path)
    _write_task(tmp_path / "tasks", "local_demo")

    discover(path=None)

    assert "local_demo" in list_tasks()


def test_discover_default_searches_active_modules(tmp_path, monkeypatch):
    """discover(path=None) finds tasks inside activated community modules."""
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)  # ensure no local ./tasks interferes

    import opencrl.modules as m

    md = m.modules_dir()
    # module repo layout: modules_dir/<module>/<taskdir>/task.py
    _write_task(md / "web_sqli", "sqli_task")
    m.activate_module("web_sqli")

    discover(path=None)

    assert "sqli_task" in list_tasks()


def test_discover_explicit_path_works(tmp_path, monkeypatch):
    """discover(path=<explicit>) searches only that directory."""
    monkeypatch.chdir(tmp_path)
    _write_task(tmp_path / "custom", "explicit_demo")

    # Also create a local ./tasks task that should NOT be found
    _write_task(tmp_path / "tasks", "should_not_appear")

    discover(path="custom")

    assert "explicit_demo" in list_tasks()
    assert "should_not_appear" not in list_tasks()


def test_discover_none_does_not_fail_when_nothing_exists(tmp_path, monkeypatch):
    """discover(path=None) is a no-op when nothing exists — no error."""
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    # No ./tasks, no cloned modules, no active modules
    discover(path=None)

    assert list_tasks() == []
