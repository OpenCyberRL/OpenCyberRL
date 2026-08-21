"""Docker integration tests for opencrl.groupeval — real daemon, tiny builds."""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from opencrl import groupeval
from opencrl.backends.docker import Docker

pytestmark = [pytest.mark.docker,
              pytest.mark.skipif(shutil.which("docker") is None,
                                 reason="docker CLI not installed")]


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Clear the task registry so tests don't leak state."""
    sys.modules["opencrl.task"]._REGISTRY.clear()
    yield
    sys.modules["opencrl.task"]._REGISTRY.clear()


def _scratch_task(root: Path, name: str) -> None:
    """A real docker task: buildable image, agent service, reset marker file."""
    d = root / "tasks" / name
    (d / "build").mkdir(parents=True)
    nonce = uuid.uuid4().hex
    (d / "build" / "Dockerfile").write_text(
        f"FROM alpine:3.20\nRUN sleep 2 && echo {nonce} > /opt/nonce.txt\n"
    )
    # The reset hook appends one line per reset; the reward's `reset_ran`
    # stage reads that count, proving warm worlds really are reset between
    # episodes (episode 1 is cold: 0 lines, episode 2: 1 line).
    (d / "world.yml").write_text(
        "x-opencrl:\n"
        "  agent: box\n"
        "  reset: echo r >> /tmp/resets.txt\n"
        "services:\n"
        "  box:\n"
        "    build: build\n"
        "    command: sleep 120\n"
    )
    (d / "task.py").write_text(
        "from opencrl import task, Task, goals, stage\n"
        f"@task\ndef {name}() -> Task:\n"
        "    return Task(goal='reply with CTF{win}', world='world.yml', max_steps=2,\n"
        "                reward=goals(\n"
        "                    stage('flag', lambda s: 1.0 if 'CTF{win}' in (s.answer or '') else 0.0),\n"
        "                    stage('reset_ran', lambda s: 1.0 if (s.file('/tmp/resets.txt') or '').count('r') == 1 else 0.0)))\n"
    )


def _win(messages, tools):
    return {"role": "assistant", "content": "CTF{win}", "tool_calls": None}


class _CountingDocker(Docker):
    """Docker backend recording every up/down world identity."""

    def __init__(self):
        super().__init__()
        self.ups = []
        self.downs = []

    def up(self, spec, caps):
        w = super().up(spec, caps)
        self.ups.append(w)
        return w

    def down(self, world):
        super().down(world)
        self.downs.append(world)


def test_group_eval_end_to_end_docker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path / "home"))
    _scratch_task(tmp_path, "ge_alpha")
    _scratch_task(tmp_path, "ge_beta")

    backend = _CountingDocker()
    out = tmp_path / "rollouts.jsonl"
    result = groupeval.run_group_eval(
        "local/level0", _win, episodes=2, backend=backend, output=out,
        concurrency=1, capacity=2)

    # One outcome per task, submission order, every episode scored.
    assert [o.task for o in result.outcomes] == ["ge_alpha", "ge_beta"]
    assert all(o.error is None for o in result.outcomes)
    assert all(len(o.rollouts) == 2 for o in result.outcomes)

    # Pool policy: one bring-up per task (warm reuse between episodes),
    # every world torn down exactly once.
    assert len(backend.ups) == 2
    assert set(map(id, backend.ups)) == set(map(id, backend.downs))

    # JSONL artifact: one Rollout per episode, stage breakdown included.
    # Episode 1 is cold (no reset ran yet): reset_ran=0.0, reward 0.5;
    # episode 2 ran on the reset warm world: both stages 1.0, reward 1.0.
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [l["task"] for l in lines] == ["ge_alpha"] * 2 + ["ge_beta"] * 2
    assert [l["reward"] for l in lines] == [0.5, 1.0, 0.5, 1.0]
    assert all(l["stages"]["flag"] == 1.0 for l in lines)
    assert [l["stages"]["reset_ran"] for l in lines] == [0.0, 1.0, 0.0, 1.0]


def test_group_eval_failed_task_does_not_abort_group_docker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path / "home"))
    _scratch_task(tmp_path, "ge_good")
    # A broken Dockerfile fails this task's build/bring-up; the group runs on.
    d = tmp_path / "tasks" / "ge_bad"
    _scratch_task(tmp_path, "ge_bad")
    (d / "build" / "Dockerfile").write_text("FROM alpine:3.20\nRUN exit 7\n")

    backend = _CountingDocker()
    result = groupeval.run_group_eval(
        "local/level0", _win, episodes=2, backend=backend,
        concurrency=1, capacity=2)

    assert [o.task for o in result.outcomes] == ["ge_bad", "ge_good"]
    assert "build failed" in (result.outcomes[0].error or "")
    assert result.outcomes[0].rollouts == []
    assert [r.reward for r in result.outcomes[1].rollouts] == [0.5, 1.0]
    # episode 2 ran on the reset warm world
    assert len(backend.ups) == 1                     # only ge_good's world
    assert set(map(id, backend.ups)) == set(map(id, backend.downs))
