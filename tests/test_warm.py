"""Tests for `opencrl warm` — resolution wiring, failure aggregation, exit codes."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import opencrl.backends.docker as dk
from opencrl import warm
from opencrl.cli import main
from opencrl.task import Caps, Task


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clear_registry():
    """Clear the task registry so tests don't leak state."""
    sys.modules["opencrl.task"]._REGISTRY.clear()


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Fresh registry + empty OPENCRL_HOME + tmp cwd for every test."""
    _clear_registry()
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    yield
    _clear_registry()


def _register(name, world, backend="docker"):
    """Register an in-memory task with an inline world dict."""
    sys.modules["opencrl.task"]._REGISTRY[name] = lambda: Task(
        goal="g", reward=lambda s: 0.0, world=world, backend=backend,
        name=name, dir="/tmp")


def _write_task_dir(root: Path, name: str) -> None:
    """Write a task.py registering a task named ``name`` under ``root``."""
    d = root / name
    d.mkdir(parents=True)
    (d / "task.py").write_text(
        "from opencrl import task, Task, Caps\n"
        f"@task\ndef {name}() -> Task:\n"
        "    return Task(goal='g', reward=lambda s: 0.0, backend='docker')\n"
    )


class FakeBackend:
    """Records prebuild calls; raises for tasks whose x-opencrl.task marker matches."""

    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)
        self.calls = []

    def prebuild(self, spec, caps):
        self.calls.append(spec)
        marker = (spec.get("x-opencrl") or {}).get("task")
        if marker in self.fail_for:
            raise RuntimeError(f"docker compose build failed:\nboom for {marker}")


def _index(*entries):
    """Build a valid index from (name, module, project, level) tuples."""
    return [{"name": n, "module": m, "project": p, "level": lv}
            for n, m, p, lv in entries]


def _ok_cp():
    import subprocess
    return subprocess.CompletedProcess([], 0, stdout="", stderr="")


# ---------------------------------------------------------------------------
# build_index: task-index derivation
# ---------------------------------------------------------------------------

def test_build_index_infers_module_from_modules_dir(tmp_path):
    from opencrl import modules
    md = modules.modules_dir() / "cybergym"
    _write_task_dir(md, "srv_enum")
    _write_task_dir(md, "ssh_brute")
    modules.activate_module("cybergym")
    _write_task_dir(tmp_path / "tasks", "local_one")

    index = warm.build_index()
    by_name = {e["name"]: e for e in index}
    assert by_name["srv_enum"]["module"] == "cybergym"
    assert by_name["ssh_brute"]["module"] == "cybergym"
    assert by_name["local_one"]["module"] == "local"


def test_build_index_defaults_project_none_level_zero():
    _write_task_dir(Path("tasks"), "bare")
    (entry,) = warm.build_index()
    assert entry["project"] is None
    assert entry["level"] == 0
    assert entry["name"] == "bare"


def test_build_index_derives_index_usable_by_resolve_group():
    _write_task_dir(Path("tasks"), "solo")
    index = warm.build_index()
    assert warm.resolve_group("local/level0", index) == ["solo"]


# ---------------------------------------------------------------------------
# warm_group: group resolution wiring
# ---------------------------------------------------------------------------

def test_warm_group_wires_filter_resolution_to_prebuild():
    _register("a", {"x-opencrl": {"task": "a"}})
    _register("b", {"x-opencrl": {"task": "b"}})
    index = _index(("a", "m", "p", 1), ("b", "m", None, 2))
    be = FakeBackend()

    results = warm.warm_group("m/level1", index, be)

    assert [r.name for r in results] == ["a"]
    assert len(be.calls) == 1
    assert results[0].status == "built"


def test_warm_group_explicit_list_preserves_order():
    _register("a", {"x-opencrl": {"task": "a"}})
    _register("b", {"x-opencrl": {"task": "b"}})
    index = _index(("a", "m", None, 0), ("b", "m", None, 0))
    be = FakeBackend()

    results = warm.warm_group(["b", "a"], index, be)

    assert [r.name for r in results] == ["b", "a"]
    assert [c["x-opencrl"]["task"] for c in be.calls] == ["b", "a"]


def test_warm_group_propagates_resolution_errors():
    _register("a", {"x-opencrl": {"task": "a"}})
    index = _index(("a", "m", None, 0))
    with pytest.raises(ValueError, match="unknown task"):
        warm.warm_group("nope", index, FakeBackend())


def test_warm_group_reports_each_task_via_callback():
    _register("a", {"x-opencrl": {"task": "a"}})
    _register("b", {"x-opencrl": {"task": "b"}})
    index = _index(("a", "m", None, 0), ("b", "m", None, 0))
    seen = []

    warm.warm_group(["a", "b"], index, FakeBackend(), report=seen.append)

    assert [r.name for r in seen] == ["a", "b"]


def test_warm_group_skips_non_docker_backend_task():
    _register("q", {"services": {}}, backend="qemu")
    index = _index(("q", "m", None, 0))
    be = FakeBackend()

    (result,) = warm.warm_group("q", index, be)

    assert result.status == "skipped"
    assert be.calls == []


# ---------------------------------------------------------------------------
# failure aggregation: failures never abort the rest
# ---------------------------------------------------------------------------

def test_failed_build_does_not_abort_remaining_tasks():
    _register("a", {"x-opencrl": {"task": "a"}})
    _register("b", {"x-opencrl": {"task": "b"}})
    index = _index(("a", "m", None, 0), ("b", "m", None, 0))
    be = FakeBackend(fail_for={"a"})

    results = warm.warm_group(["a", "b"], index, be)

    assert [r.status for r in results] == ["failed", "built"]
    assert len(be.calls) == 2          # 'b' still prebuilt after 'a' failed
    assert "boom for a" in results[0].error


def test_failed_reason_is_collapsed_and_bounded():
    _register("a", {"x-opencrl": {"task": "a"}})
    index = _index(("a", "m", None, 0))
    be = FakeBackend(fail_for={"a"})

    (result,) = warm.warm_group("a", index, be)

    assert "\n" not in result.error   # multi-line stderr collapsed to one line
    assert len(result.error) <= 200


def test_summarize_counts_each_status():
    results = [warm.WarmResult("a", "failed", "boom"),
               warm.WarmResult("b", "built"),
               warm.WarmResult("c", "cached"),
               warm.WarmResult("d", "no-build")]
    assert warm.summarize(results) == "1 built, 1 cached, 1 no-build, 1 failed"


def test_summarize_empty_results():
    assert warm.summarize([]) == "no tasks warmed"


# ---------------------------------------------------------------------------
# built / cached / no-build classification (real pin machinery)
# ---------------------------------------------------------------------------

def _build_spec():
    return {"x-opencrl": {"basedir": "/abs/task"},
            "services": {"a": {"build": "/abs/task/ctx"}}}


def test_warm_reports_no_build_for_image_only_world(monkeypatch):
    calls = []

    def fake_run(args, timeout=None):
        calls.append(args)
        return _ok_cp()

    monkeypatch.setattr(dk, "_run", fake_run)
    _register("t", {"services": {"a": {"image": "alpine"}}})
    index = _index(("t", "m", None, 0))

    (result,) = warm.warm_group("t", index, dk.Docker())

    assert result.status == "no-build"
    assert not any("build" in c for c in calls)   # no docker compose build ran


def test_warm_reports_cached_when_layer_chain_unchanged(monkeypatch):
    monkeypatch.setattr(dk, "_run", lambda args, timeout=None: _ok_cp())
    monkeypatch.setattr(warm, "_layer_chain", lambda tag: '["sha256:a"]')
    _register("t", _build_spec())
    index = _index(("t", "m", None, 0))

    (result,) = warm.warm_group("t", index, dk.Docker())

    assert result.status == "cached"


def test_warm_reports_built_when_tag_absent(monkeypatch):
    # Tag missing before prebuild (no image yet) → a real build happened.
    monkeypatch.setattr(dk, "_run", lambda args, timeout=None: _ok_cp())
    seen = []

    def fake_layer_chain(tag):
        seen.append(tag)
        return '["sha256:new"]' if len(seen) > 1 else None

    monkeypatch.setattr(warm, "_layer_chain", fake_layer_chain)
    _register("t", _build_spec())
    index = _index(("t", "m", None, 0))

    (result,) = warm.warm_group("t", index, dk.Docker())

    assert result.status == "built"


def test_warm_reports_built_when_layers_changed(monkeypatch):
    # Tag exists but the rebuild produced different layers (stale tag) → built.
    monkeypatch.setattr(dk, "_run", lambda args, timeout=None: _ok_cp())
    seen = []

    def fake_layer_chain(tag):
        seen.append(tag)
        return '["sha256:old"]' if len(seen) == 1 else '["sha256:new"]'

    monkeypatch.setattr(warm, "_layer_chain", fake_layer_chain)
    _register("t", _build_spec())
    index = _index(("t", "m", None, 0))

    (result,) = warm.warm_group("t", index, dk.Docker())

    assert result.status == "built"

def test_pinned_tags_are_content_addressed():
    # warm delegates to the backend's public pinned_tags seam.
    be = dk.Docker()
    tags1 = warm._pinned_tags(be, _build_spec(), Caps())
    tags2 = be.pinned_tags(_build_spec(), Caps())
    assert tags1 == tags2
    assert tags1[0].startswith("opencrl-build-")


def test_pinned_tags_none_when_backend_lacks_the_seam():
    assert warm._pinned_tags(FakeBackend(), _build_spec(), Caps()) is None


# ---------------------------------------------------------------------------
# CLI: exit codes and summary output (mocked prebuild seam)
# ---------------------------------------------------------------------------

def test_cli_warm_success_exit_zero(monkeypatch, capsys):
    _register("ok", {"x-opencrl": {"task": "ok"}})
    monkeypatch.setattr("opencrl.backends.docker.Docker", FakeBackend)

    rc = main(["warm", "ok"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "ok" in out and "built" in out


def test_cli_warm_failure_exit_nonzero_and_summary_names_failures(
        monkeypatch, capsys):
    _register("good", {"x-opencrl": {"task": "good"}})
    _register("bad", {"x-opencrl": {"task": "bad"}})

    class FailingBackend(FakeBackend):
        def __init__(self):
            super().__init__(fail_for={"bad"})

    monkeypatch.setattr("opencrl.backends.docker.Docker", FailingBackend)

    rc = main(["warm", "good", "bad"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "bad" in out            # failure named
    assert "boom for bad" in out   # ...with its reason
    assert "good" in out           # non-failure still reported


def test_cli_warm_resolves_each_argument_independently(monkeypatch, capsys):
    # A filter form and a bare task name mix freely; each resolves on its own.
    _register("a", {"x-opencrl": {"task": "a"}})
    _register("b", {"x-opencrl": {"task": "b"}})

    made = []

    class RecordingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            made.append(self)

    monkeypatch.setattr("opencrl.backends.docker.Docker", RecordingBackend)

    rc = main(["warm", "local/level0", "a"])

    out = capsys.readouterr().out
    assert rc == 0
    calls = [c for be in made for c in be.calls]
    assert [c["x-opencrl"]["task"] for c in calls] == ["a", "b", "a"]
    assert "3 built" in out


def test_cli_warm_unknown_group_expression_errors(monkeypatch, capsys):
    _register("ok", {"x-opencrl": {"task": "ok"}})
    monkeypatch.setattr("opencrl.backends.docker.Docker", FakeBackend)

    rc = main(["warm", "nope/level1"])

    assert rc == 2
    assert "unknown module" in capsys.readouterr().out
