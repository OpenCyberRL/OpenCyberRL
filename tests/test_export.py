import pytest
datasets = pytest.importorskip("datasets")

import json
from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag, chain, stage, contains
from opencrl.models import ScriptedModel
from opencrl.backends.mock import MockBackend
from opencrl.adapters.export import export_rollouts


def _solved_model():
    return ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "cat /flag"})}}]},
        {"role": "assistant", "content": "The flag is CTF{win}", "tool_calls": None},
    ])


def _failed_model():
    return ScriptedModel([
        {"role": "assistant", "content": "I couldn't find it", "tool_calls": None},
    ])


def make_task():
    return Task(goal="Read the flag.", reward=flag("CTF{win}"), tools=(shell,),
                caps=Caps(offensive=True), max_steps=5, name="demo")


def test_export_dpo_creates_preference_pairs():
    from opencrl.rollout import rollout
    rollouts = [
        rollout(make_task(), _solved_model(), backend=MockBackend(exec_map={"cat /flag": "CTF{win}"})),
        rollout(make_task(), _failed_model(), backend=MockBackend(exec_map={"cat /flag": "nope"})),
    ]
    from opencrl.adapters.export import _to_dpo
    ds = _to_dpo(make_task(), rollouts)
    assert "prompt" in ds.column_names
    assert "chosen" in ds.column_names
    assert "rejected" in ds.column_names
    assert len(ds) == 1
    assert "CTF{win}" in str(ds[0]["chosen"])
    assert "couldn't find it" in str(ds[0]["rejected"])


def test_export_sft_keeps_only_reward_one():
    from opencrl.rollout import rollout
    rollouts = [
        rollout(make_task(), _solved_model(), backend=MockBackend(exec_map={"cat /flag": "CTF{win}"})),
        rollout(make_task(), _failed_model(), backend=MockBackend(exec_map={"cat /flag": "nope"})),
    ]
    from opencrl.adapters.export import _to_sft
    ds = _to_sft(make_task(), rollouts)
    assert len(ds) == 1
    assert "prompt" in ds.column_names
    assert "completion" in ds.column_names
    assert "CTF{win}" in str(ds[0]["completion"])


def test_export_kto_has_label():
    from opencrl.rollout import rollout
    rollouts = [
        rollout(make_task(), _solved_model(), backend=MockBackend(exec_map={"cat /flag": "CTF{win}"})),
        rollout(make_task(), _failed_model(), backend=MockBackend(exec_map={"cat /flag": "nope"})),
    ]
    from opencrl.adapters.export import _to_kto
    ds = _to_kto(make_task(), rollouts)
    assert len(ds) == 2
    assert "label" in ds.column_names
    assert ds[0]["label"] is True
    assert ds[1]["label"] is False


def test_export_dpo_with_staged_reward_uses_aggregate():
    from opencrl.rollout import rollout
    staged = Task(
        goal="Get root then read flag.", name="staged",
        reward=chain(stage("root", contains("uid=0")), stage("flag", flag("CTF{x}"))),
        tools=(shell,), max_steps=5,
    )
    full = ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "id"})}}]},
        {"role": "assistant", "content": "CTF{x}", "tool_calls": None},
    ])
    partial = ScriptedModel([
        {"role": "assistant", "content": "CTF{x}", "tool_calls": None},
    ])
    rollouts = [
        rollout(staged, full, backend=MockBackend(exec_map={"id": "uid=0(root)"})),
        rollout(staged, partial, backend=MockBackend(exec_map={})),
    ]
    from opencrl.adapters.export import _to_dpo
    ds = _to_dpo(staged, rollouts)
    assert len(ds) == 1
    assert "uid=0" in str(ds[0]["chosen"])


def test_export_rollouts_runs_n_rollouts():
    ds = export_rollouts(
        make_task(),
        lambda m, t: {"role": "assistant", "content": "CTF{win}", "tool_calls": None},
        n=4, backend=MockBackend(),
        fmt="kto",
    )
    assert len(ds) == 4
    assert "label" in ds.column_names


def test_export_invalid_fmt_raises():
    with pytest.raises(ValueError, match="unknown format"):
        export_rollouts(
            make_task(),
            lambda m, t: {"role": "assistant", "content": "x", "tool_calls": None},
            n=1, backend=MockBackend(),
            fmt="invalid",
        )
