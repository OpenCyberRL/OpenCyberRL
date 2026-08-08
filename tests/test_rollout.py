import json
from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag
from opencrl.models import ScriptedModel
from opencrl.backends.mock import MockBackend
from opencrl.rollout import rollout, Rollout

def make_task():
    return Task(
        goal="Read the flag with `cat`.",
        reward=flag("CTF{win}"),
        tools=(shell,),
        caps=Caps(offensive=True),
        max_steps=5,
    )

def solved_model():
    return ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "cat /flag"})}}]},
        {"role": "assistant", "content": "The flag is CTF{win}", "tool_calls": None},
    ])

def test_rollout_solved_scores_one():
    backend = MockBackend(exec_map={"cat /flag": "CTF{win}"})
    r = rollout(make_task(), solved_model(), backend=backend)
    assert isinstance(r, Rollout)
    assert r.reward == 1.0
    # transcript records the tool call and its result
    roles = [m["role"] for m in r.transcript]
    assert "assistant" in roles and "tool" in roles

def test_rollout_unsolved_scores_zero():
    backend = MockBackend(exec_map={"cat /flag": "nothing here"})
    model = ScriptedModel([{"role": "assistant", "content": "no idea", "tool_calls": None}])
    r = rollout(make_task(), model, backend=backend)
    assert r.reward == 0.0

def test_rollout_respects_max_steps():
    # A model that always makes a tool call would loop forever without the guard.
    def loop_call(_m, _t):
        return {"role": "assistant", "content": None,
                "tool_calls": [{"id": "x", "type": "function",
                                "function": {"name": "shell",
                                             "arguments": json.dumps({"command": "noop"})}}]}
    backend = MockBackend(exec_map={"noop": ""})
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), max_steps=3)
    r = rollout(task, loop_call, backend=backend)
    # 3 assistant turns + 3 tool results
    assert sum(1 for m in r.transcript if m["role"] == "assistant") == 3
    assert r.reward == 0.0

def test_rollout_to_dict_has_caps():
    backend = MockBackend(exec_map={"cat /flag": "CTF{win}"})
    d = rollout(make_task(), solved_model(), backend=backend).to_dict()
    assert d["reward"] == 1.0
    assert d["caps"]["offensive"] is True
