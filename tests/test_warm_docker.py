"""Docker integration tests for `opencrl warm` — real daemon, tiny real builds."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from opencrl import warm
from opencrl.backends.docker import Docker
from opencrl.task import Caps, get_task, load_world

pytestmark = [pytest.mark.docker,
              pytest.mark.skipif(shutil.which("docker") is None,
                                 reason="docker CLI not installed")]


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Clear the task registry so tests don't leak state."""
    sys.modules["opencrl.task"]._REGISTRY.clear()
    yield
    sys.modules["opencrl.task"]._REGISTRY.clear()


def _scratch_task(root: Path, name: str, dockerfile: str | None = None) -> None:
    d = root / "tasks" / name
    (d / "build").mkdir(parents=True)
    if dockerfile is None:
        # Unique content forces a genuinely cold first build (BuildKit's
        # layer cache persists across test runs); the sleep guarantees the
        # cold build is measurably slower than a cache-hit rebuild.
        nonce = uuid.uuid4().hex
        dockerfile = f"FROM alpine:3.20\nRUN sleep 3 && echo {nonce} > /opt/warm.txt\n"
    (d / "world.yml").write_text(
        "services:\n"
        "  box:\n"
        "    build: build\n"
        "    command: sleep 1\n"
    )
    (d / "build" / "Dockerfile").write_text(dockerfile)
    (d / "task.py").write_text(
        "from opencrl import task, Task, Caps\n"
        f"@task\ndef {name}() -> Task:\n"
        "    return Task(goal='warm me', reward=lambda s: 0.0,\n"
        "                world='world.yml', backend='docker', caps=Caps())\n"
    )


def test_warm_builds_then_rewarms_from_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path / "home"))
    _scratch_task(tmp_path, "warm_scratch")

    index = warm.build_index()
    assert [e["name"] for e in index] == ["warm_scratch"]
    backend = Docker()

    t0 = time.perf_counter()
    first = warm.warm_group("warm_scratch", index, backend)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    second = warm.warm_group("warm_scratch", index, backend)
    t_second = time.perf_counter() - t0

    assert first[0].status == "built"
    assert second[0].status == "cached"
    assert t_second < t_first          # re-warm hits the build cache
    assert t_second < 15.0             # a cache-hit re-warm stays quick

    # The content-addressed image really is in the local docker store.
    spec = load_world(get_task("warm_scratch"))
    (tag,) = warm._pinned_tags(backend, spec, Caps())
    cp = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    assert cp.returncode == 0


def test_warm_failure_does_not_abort_remaining_tasks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENCRL_HOME", str(tmp_path / "home"))
    _scratch_task(tmp_path, "warm_good")
    _scratch_task(tmp_path, "warm_bad",
                  dockerfile="FROM alpine:3.20\nRUN exit 7\n")

    index = warm.build_index()
    results = warm.warm_group(["warm_bad", "warm_good"], index, Docker())

    assert [r.status for r in results] == ["failed", "built"]
    assert results[0].error            # failure carries the build's reason
    assert "warm_bad" in results[0].name
