"""Tests for opencrl.groupeval — the group-aware run/eval entry point."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

import opencrl.cli as cli
from opencrl import groupeval
from opencrl.backends.mock import MockBackend, MockWorld
from opencrl.reward import chain, flag, stage
from opencrl.task import Caps, Task
from opencrl.tools import shell


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


def _register(name, *, reward=None, reset="RESET", world=None):
    """Register an in-memory task with an inline world dict."""
    sys.modules["opencrl.task"]._REGISTRY[name] = lambda: Task(
        goal=f"goal-{name}", reward=reward or flag("CTF{win}"),
        tools=(shell,), world=world or {"x-opencrl": {"task": name}},
        name=name, dir="/tmp", max_steps=2, reset=reset)


def _win(messages, tools):
    return {"role": "assistant", "content": "CTF{win}", "tool_calls": None}


class _World(MockWorld):
    """MockWorld that records reset-hook invocations."""

    def __init__(self, resets):
        super().__init__()
        self._resets = resets

    def exec(self, command, host=None):
        if command == "RESET":
            self._resets.append(command)
        return ""


class CountingBackend(MockBackend):
    """MockBackend recording every up/down; `fail_for` names break on bring-up."""

    def __init__(self, fail_for=()):
        super().__init__()
        self.fail_for = set(fail_for)
        self.ups = []
        self.downs = []
        self.resets = []

    def up(self, spec, caps):
        marker = (spec.get("x-opencrl") or {}).get("task")
        if marker in self.fail_for:
            raise RuntimeError(f"docker compose up failed: boom for {marker}")
        w = _World(self.resets)
        self.ups.append(w)
        return w

    def down(self, world):
        self.downs.append(world)


def _index(*entries):
    """Build a valid index from (name, module, project, level) tuples."""
    return [{"name": n, "module": m, "project": p, "level": lv}
            for n, m, p, lv in entries]


def _write_task(root: Path, name: str, *, reset="true"):
    """Write a mock-backend task.py under ``root/tasks/<name>/``."""
    d = root / "tasks" / name
    d.mkdir(parents=True)
    reset_arg = f", reset={reset!r}" if reset else ""
    (d / "task.py").write_text(
        "from opencrl import task, Task, flag\n"
        f"@task\ndef {name}() -> Task:\n"
        "    return Task(goal='find the flag', reward=flag('CTF{win}'),\n"
        f"                backend='mock', max_steps=2{reset_arg})\n"
    )


# ---------------------------------------------------------------------------
# resolution wiring
# ---------------------------------------------------------------------------

def test_resolves_group_expression_against_index():
    _register("a")
    _register("b")
    result = groupeval.run_group_eval(
        "m/level0", _win, index=_index(("a", "m", None, 0), ("b", "m", None, 1)),
        backend=MockBackend())
    assert [o.task for o in result.outcomes] == ["a"]


def test_explicit_list_preserves_submission_order():
    _register("a")
    _register("b")
    result = groupeval.run_group_eval(
        ["b", "a"], _win, index=_index(("a", "m", None, 0), ("b", "m", None, 0)),
        backend=MockBackend())
    assert [o.task for o in result.outcomes] == ["b", "a"]


def test_index_defaults_to_discovered_tasks():
    _write_task(Path("."), "solo")
    result = groupeval.run_group_eval("local/level0", _win)
    assert [o.task for o in result.outcomes] == ["solo"]
    assert result.outcomes[0].error is None
    assert result.outcomes[0].rollouts[0].reward == 1.0


def test_cli_run_rejects_zero_episodes(monkeypatch, capsys, tmp_path):
    _write_task(Path("."), "solo")
    _fake_openai(monkeypatch)
    rc = cli.main(["run", "solo", "--model", "m", "-n", "0"])
    assert rc == 1
    assert "episodes" in capsys.readouterr().out


def test_unresolvable_group_expression_raises():
    _register("a")
    with pytest.raises(ValueError, match="unknown"):
        groupeval.run_group_eval("nope/level0", _win,
                                 index=_index(("a", "m", None, 0)),
                                 backend=MockBackend())


def test_rejects_nonpositive_episodes():
    _register("a")
    with pytest.raises(ValueError, match="episodes"):
        groupeval.run_group_eval("a", _win, episodes=0,
                                 index=_index(("a", "m", None, 0)),
                                 backend=MockBackend())


def test_index_name_missing_from_registry_is_reported_not_raised():
    _register("live")
    result = groupeval.run_group_eval(
        ["ghost", "live"], _win,
        index=_index(("ghost", "m", None, 0), ("live", "m", None, 0)),
        backend=MockBackend())
    assert [o.task for o in result.outcomes] == ["ghost", "live"]
    assert "ghost" in (result.outcomes[0].error or "")
    assert result.outcomes[0].rollouts == []
    assert len(result.outcomes[1].rollouts) == 1


# ---------------------------------------------------------------------------
# artifacts: one Rollout per episode, stage breakdown, JSONL
# ---------------------------------------------------------------------------

def test_jsonl_carries_one_rollout_per_episode_with_stage_breakdown(tmp_path):
    _register("staged", reward=chain(
        stage("flag", lambda s: 1.0),
        stage("proof", lambda s: 0.5)))
    out = tmp_path / "rollouts.jsonl"
    result = groupeval.run_group_eval(
        "staged", _win, episodes=2, index=_index(("staged", "m", None, 0)),
        backend=MockBackend(), output=out)

    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 2                       # one Rollout per episode
    assert [l["task"] for l in lines] == ["staged", "staged"]
    assert all(l["stages"] == {"flag": 1.0, "proof": 0.5} for l in lines)
    assert all(l["reward"] == pytest.approx(0.75) for l in lines)
    assert all("transcript" in l and "caps" in l for l in lines)
    assert len(result.rollouts) == 2             # the result mirrors the file


def test_jsonl_submission_order_survives_out_of_order_completion(tmp_path):
    _register("t0")
    _register("t1")
    out = tmp_path / "rollouts.jsonl"

    def slow_first_task(messages, tools):
        if "goal-t0" in messages[1]["content"]:
            time.sleep(0.10)                     # t0's episodes finish last
        return _win(messages, tools)

    groupeval.run_group_eval(
        ["t0", "t1"], slow_first_task, episodes=2,
        index=_index(("t0", "m", None, 0), ("t1", "m", None, 0)),
        backend=CountingBackend(), concurrency=4, output=out)

    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [l["task"] for l in lines] == ["t0", "t0", "t1", "t1"]


# ---------------------------------------------------------------------------
# pool policy: warm reuse, cold bring-up
# ---------------------------------------------------------------------------

def test_warm_worlds_reset_and_reused_across_episodes():
    _register("t0")
    _register("t1")
    _register("t2")
    be = CountingBackend()
    result = groupeval.run_group_eval(
        ["t0", "t1", "t2"], _win, episodes=3,
        index=_index(*[(f"t{i}", "m", None, 0) for i in range(3)]),
        backend=be, concurrency=1, capacity=2)

    assert len(result.rollouts) == 9
    assert all(r.reward == 1.0 for r in result.rollouts)
    assert len(be.ups) == 3                       # one bring-up per task
    assert len(be.resets) == 6                    # 9 episodes minus 3 cold firsts
    assert set(map(id, be.downs)) == set(map(id, be.ups))
    assert len(be.downs) == 3                     # every world downed exactly once


# ---------------------------------------------------------------------------
# failure tolerance: per-task failures never abort the group
# ---------------------------------------------------------------------------

def test_failed_bringup_reported_once_and_group_runs_on():
    _register("bad")
    _register("good")
    be = CountingBackend(fail_for={"bad"})
    result = groupeval.run_group_eval(
        ["bad", "good"], _win, episodes=3,
        index=_index(("bad", "m", None, 0), ("good", "m", None, 0)),
        backend=be, concurrency=1)

    assert [o.task for o in result.outcomes] == ["bad", "good"]
    assert "boom" in (result.outcomes[0].error or "")
    assert result.outcomes[0].rollouts == []
    assert len(result.outcomes[1].rollouts) == 3  # the group ran on
    assert len(be.ups) == 1                       # only good's world (reused ×3)


def test_task_without_reset_hook_is_reported_group_runs_on():
    _register("noreset", reset=None)
    _register("good")
    be = CountingBackend()
    result = groupeval.run_group_eval(
        ["noreset", "good"], _win, episodes=2,
        index=_index(("noreset", "m", None, 0), ("good", "m", None, 0)),
        backend=be)

    assert "reset" in (result.outcomes[0].error or "")
    assert result.outcomes[0].rollouts == []
    assert len(result.outcomes[1].rollouts) == 2
    assert len(be.ups) == 1                       # noreset never came up


def test_model_failure_fails_only_that_task():
    _register("a")
    _register("b")

    def flaky(messages, tools):
        if "goal-a" in messages[1]["content"]:
            raise RuntimeError("model exploded")
        return _win(messages, tools)

    result = groupeval.run_group_eval(
        ["a", "b"], flaky, episodes=2,
        index=_index(("a", "m", None, 0), ("b", "m", None, 0)),
        backend=CountingBackend(), concurrency=1)

    assert "model exploded" in (result.outcomes[0].error or "")
    assert result.outcomes[0].rollouts == []
    assert len(result.outcomes[1].rollouts) == 2


# ---------------------------------------------------------------------------
# GroupEvalResult helpers
# ---------------------------------------------------------------------------

def test_rollouts_flatten_in_submission_order():
    r0 = groupeval.TaskOutcome("a", ["r0", "r1"])
    r1 = groupeval.TaskOutcome("b", ["r2"])
    result = groupeval.GroupEvalResult("a, b", [r0, r1])
    assert result.rollouts == ["r0", "r1", "r2"]


def test_summarize_tally():
    ok = groupeval.TaskOutcome("a", ["r0", "r1"])
    bad = groupeval.TaskOutcome("b", [], "boom")
    result = groupeval.GroupEvalResult(["a", "b"], [ok, bad])
    assert result.summarize() == "2 tasks, 2 episodes, 1 failed"
    assert groupeval.GroupEvalResult("a", [ok]).summarize() == "1 tasks, 2 episodes"


# ---------------------------------------------------------------------------
# CLI: opencrl run
# ---------------------------------------------------------------------------

def _fake_openai(monkeypatch, seen=None):
    """Patch OpenAIModel to a no-network callable; record the model id."""
    import opencrl.models as models

    def fake(model, **kw):
        if seen is not None:
            seen["model"] = model
        return _win

    monkeypatch.setattr(models, "OpenAIModel", fake)


def test_cli_run_writes_jsonl_and_exits_zero(monkeypatch, capsys, tmp_path):
    _write_task(Path("."), "solo")
    _fake_openai(monkeypatch)
    out = tmp_path / "r.jsonl"

    rc = cli.main(["run", "solo", "--model", "test-model",
                   "-n", "2", "-o", str(out)])

    assert rc == 0
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [l["task"] for l in lines] == ["solo", "solo"]
    assert all(l["reward"] == 1.0 for l in lines)
    assert "solo" in capsys.readouterr().out


def test_cli_run_model_id_defaults_to_env(monkeypatch, tmp_path):
    _write_task(Path("."), "solo")
    seen = {}
    _fake_openai(monkeypatch, seen)
    monkeypatch.setenv("OPENCRL_MODEL", "env-model")

    rc = cli.main(["run", "solo", "-o", str(tmp_path / "r.jsonl")])

    assert rc == 0
    assert seen["model"] == "env-model"


def test_cli_run_without_model_errors(monkeypatch, capsys, tmp_path):
    _write_task(Path("."), "solo")
    _fake_openai(monkeypatch)
    monkeypatch.delenv("OPENCRL_MODEL", raising=False)

    rc = cli.main(["run", "solo"])

    assert rc == 1
    assert "model" in capsys.readouterr().out.lower()


def test_cli_run_task_failure_exits_one(monkeypatch, capsys, tmp_path):
    _write_task(Path("."), "broken", reset=None)
    _write_task(Path("."), "good")
    _fake_openai(monkeypatch)

    rc = cli.main(["run", "local/level0", "--model", "m",
                   "-o", str(tmp_path / "r.jsonl")])

    assert rc == 1
    out = capsys.readouterr().out
    assert "broken" in out and "reset" in out


def test_cli_run_unknown_group_exits_two(monkeypatch, capsys, tmp_path):
    _write_task(Path("."), "solo")
    _fake_openai(monkeypatch)

    rc = cli.main(["run", "nope/level9", "--model", "m"])

    assert rc == 2
    assert "unknown" in capsys.readouterr().out
