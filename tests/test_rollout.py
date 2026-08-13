import json
from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag, contains, stage, chain
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

def staged_task():
    return Task(
        goal="Get root, then read the flag.",
        reward=chain(
            stage("root", contains("uid=0")),
            stage("flag", flag("CTF{win}")),
        ),
        tools=(shell,),
        max_steps=5,
    )


def staged_model():
    return ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "id"})}}]},
        {"role": "assistant", "content": "the flag is CTF{win}", "tool_calls": None},
    ])


def test_rollout_carries_stage_breakdown():
    backend = MockBackend(exec_map={"id": "uid=0(root) gid=0(root)"})
    r = rollout(staged_task(), staged_model(), backend=backend)
    assert r.reward == 1.0
    assert r.stages == {"root": 1.0, "flag": 1.0}
    assert r.to_dict()["stages"] == {"root": 1.0, "flag": 1.0}


def test_rollout_scalar_reward_has_no_stages():
    backend = MockBackend(exec_map={"cat /flag": "CTF{win}"})
    r = rollout(make_task(), solved_model(), backend=backend)   # existing scalar task
    assert r.stages is None
    assert r.to_dict()["stages"] is None


def test_rollout_survives_unknown_tool_name():
    """A model that hallucinates a tool name should not crash the rollout."""
    backend = MockBackend(exec_map={})
    model = ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "nonexistent",
                                      "arguments": "{}"}}]},
        {"role": "assistant", "content": "I tried but failed", "tool_calls": None},
    ])
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), max_steps=5)
    r = rollout(task, model, backend=backend)
    assert r.reward == 0.0
    # The error string should appear in the tool result message
    tool_msgs = [m for m in r.transcript if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "tool call failed" in tool_msgs[0]["content"]


def test_rollout_survives_malformed_json_args():
    """A model that sends invalid JSON arguments should not crash the rollout."""
    backend = MockBackend(exec_map={})
    model = ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": "{not valid json"}}]},
        {"role": "assistant", "content": "I tried but failed", "tool_calls": None},
    ])
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), max_steps=5)
    r = rollout(task, model, backend=backend)
    assert r.reward == 0.0
    tool_msgs = [m for m in r.transcript if m["role"] == "tool"]
    assert "tool call failed" in tool_msgs[0]["content"]


def test_rollout_survives_tool_run_exception():
    """A backend that raises during exec should not crash the rollout."""
    class ExplodingBackend(MockBackend):
        def up(self, spec, caps):
            class W:
                agent = "agent"
                def exec(self, command, host=None):
                    raise RuntimeError("kaboom")
                def read_file(self, path, host=None):
                    return None
            return W()
    backend = ExplodingBackend()
    model = ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "id"})}}]},
        {"role": "assistant", "content": "I tried but failed", "tool_calls": None},
    ])
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), max_steps=5)
    r = rollout(task, model, backend=backend)
    assert r.reward == 0.0
    tool_msgs = [m for m in r.transcript if m["role"] == "tool"]
    assert "kaboom" in tool_msgs[0]["content"]
