import json
import pytest
gym = pytest.importorskip("gymnasium")

from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag, file_exists, contains, stage, chain
from opencrl.backends.mock import MockBackend
from opencrl.adapters.gym import to_gym

def test_gym_env_has_valid_spaces():
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), max_steps=5)
    env = to_gym(task, backend=MockBackend())
    assert isinstance(env.action_space, gym.spaces.Space)
    assert isinstance(env.observation_space, gym.spaces.Space)

def test_gym_env_spaces_contain_real_messages():
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), max_steps=5)
    env = to_gym(task, backend=MockBackend())
    message = {"role": "assistant", "content": "hi", "tool_calls": None}
    assert env.action_space.contains(message)
    assert env.observation_space.contains([message])
    # A str (what the old Text space would produce) must NOT satisfy the
    # action space, since step() calls action.get(...) on it.
    assert not env.action_space.contains("hi")

def test_gym_env_action_space_sample_is_a_message_dict():
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), max_steps=5)
    env = to_gym(task, backend=MockBackend())
    sample = env.action_space.sample()
    assert isinstance(sample, dict)
    assert sample.get("role") is not None
    obs_sample = env.observation_space.sample()
    assert isinstance(obs_sample, list)

def test_gym_env_reset_observation_is_a_stable_snapshot():
    # obs0 must not mutate when step() later appends to the live transcript
    # (Fix 6: retained observations must not corrupt retroactively).
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), max_steps=5)
    env = to_gym(task, backend=MockBackend(exec_map={"cat /flag": "CTF{win}"}))
    obs0, info = env.reset()
    before = len(obs0)
    env.step({"role": "assistant", "content": None,
              "tool_calls": [{"id": "1", "type": "function",
                              "function": {"name": "shell",
                                           "arguments": json.dumps({"command": "cat /flag"})}}]})
    assert len(obs0) == before

def test_gym_env_scores_verifier_on_tool_call_truncation():
    # max_steps=1: the only allowed action is a tool call, so the episode
    # truncates on that step. rollout() would score the verifier at that
    # point (answer=""); the gym env's step() must match, not report 0.0.
    task = Task(goal="g", reward=file_exists("/flag"), tools=(shell,), max_steps=1)
    env = to_gym(task, backend=MockBackend(files={"/flag": "CTF{win}"}))
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "id"})}}]})
    assert truncated and not terminated
    assert reward == 1.0

def test_gym_env_step_cycle():
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), max_steps=5)
    env = to_gym(task, backend=MockBackend(exec_map={"cat /flag": "CTF{win}"}))
    obs, info = env.reset()
    assert any(m["role"] == "user" for m in obs)
    # one tool-calling turn
    obs, reward, terminated, truncated, info = env.step(
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "cat /flag"})}}]})
    assert not terminated
    # final answer turn
    obs, reward, terminated, truncated, info = env.step(
        {"role": "assistant", "content": "CTF{win}", "tool_calls": None})
    assert terminated and reward == 1.0

def test_gym_env_scores_staged_reward_as_aggregate_float():
    # A Score-returning (staged) reward must flow through step() as the aggregate
    # float — to_gym scores by calling task.reward directly, not via Rollout.reward.
    task = Task(
        goal="g",
        reward=chain(stage("root", contains("uid=0")),
                     stage("flag", flag("CTF{win}"))),
        tools=(shell,), max_steps=5,
    )
    env = to_gym(task, backend=MockBackend(exec_map={"id": "uid=0(root)"}))
    env.reset()
    # tool-calling turn: puts "uid=0(root)" in the transcript so `root` scores 1.0
    env.step({"role": "assistant", "content": None,
              "tool_calls": [{"id": "1", "type": "function",
                              "function": {"name": "shell",
                                           "arguments": json.dumps({"command": "id"})}}]})
    # final answer lacks the flag: chain gives root=1.0 (0.5 share), flag=0.0 -> 0.5
    obs, reward, terminated, truncated, info = env.step(
        {"role": "assistant", "content": "no flag found", "tool_calls": None})
    assert terminated and not truncated
    assert reward == 0.5
    assert isinstance(reward, float)

def test_gym_reset_nulls_world_before_up_so_no_double_teardown():
    """If up() raises, close() must not call down() on the stale world."""
    down_count = [0]

    class RecordingBackend(MockBackend):
        def up(self, spec, caps):
            if down_count[0] > 0:
                raise RuntimeError("up failed")
            return super().up(spec, caps)
        def down(self, world):
            down_count[0] += 1

    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), max_steps=5)
    env = to_gym(task, backend=RecordingBackend())
    env.reset()
    try:
        env.reset()
    except RuntimeError:
        pass
    env.close()
    assert down_count[0] == 1
