import json
import pytest
gym = pytest.importorskip("gymnasium")

from cyberl.task import Task, Caps
from cyberl.tools import shell
from cyberl.reward import flag, file_exists
from cyberl.backends.mock import MockBackend
from cyberl.adapters.gym import to_gym

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
