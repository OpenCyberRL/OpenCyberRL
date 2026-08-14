import json
import pytest
from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag
from opencrl.models import ScriptedModel
from opencrl.adapters.eval import evaluate

pytestmark = pytest.mark.docker


def _task():
    return Task(
        goal="Read /flag and state it.", name="conc",
        reward=flag("CTF{hi}"), tools=(shell,), max_steps=3,
        caps=Caps(offensive=True),
        world={"x-opencrl": {"agent": "box"},
               "services": {"box": {"image": "alpine:3.20",
                                    "command": "sh -c 'echo CTF{hi}>/flag; sleep 600'"}}},
    )


def _script():
    # a fresh instance per rollout (model_factory) since ScriptedModel is stateful
    return ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "cat /flag"})}}]},
        {"role": "assistant", "content": "CTF{hi}", "tool_calls": None},
    ])


def test_concurrent_evaluate_runs():
    stats = evaluate(_task(), model_factory=_script, n=4, concurrency=4)
    assert stats["n"] == 4
    assert stats["mean_reward"] == 1.0
