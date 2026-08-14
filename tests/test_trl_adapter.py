import json
import pytest

from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag, chain, stage, contains
from opencrl.models import ScriptedModel
from opencrl.backends.mock import MockBackend
from opencrl.adapters.trl_adapter import to_trl


def make_task():
    return Task(goal="Read the flag.", reward=flag("CTF{win}"), tools=(shell,),
                caps=Caps(offensive=True), max_steps=5, name="demo")


def solved_model():
    return ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "cat /flag"})}}]},
        {"role": "assistant", "content": "The flag is CTF{win}", "tool_calls": None},
    ])


def test_to_trl_returns_env_class_and_reward_func():
    env_cls, reward_func = to_trl(make_task(), backend=MockBackend())
    assert callable(reward_func)
    assert hasattr(env_cls, "reset")
    assert hasattr(env_cls, "get_reward")
    assert hasattr(env_cls, "_close")


def test_trl_env_reset_calls_backend_up():
    up_count = [0]
    class CountingBackend(MockBackend):
        def up(self, spec, caps):
            up_count[0] += 1
            return super().up(spec, caps)
    env_cls, _ = to_trl(make_task(), backend=CountingBackend())
    env = env_cls()
    prompt = env.reset()
    assert up_count[0] == 1
    assert prompt == "Read the flag."
    env._close()


def test_trl_env_get_reward_returns_scalar():
    env_cls, _ = to_trl(
        make_task(),
        backend=MockBackend(exec_map={"cat /flag": "CTF{win}"}),
    )
    env = env_cls()
    env.reset()
    env._episode.transcript.append(
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "cat /flag"})}}]})
    env._episode.transcript.extend(
        env._episode.run_tool_calls(env._episode.transcript[-1]["tool_calls"]))
    env._episode.transcript.append(
        {"role": "assistant", "content": "The flag is CTF{win}", "tool_calls": None})
    reward = env.get_reward()
    assert reward == 1.0
    env._close()


def test_trl_env_get_reward_staged():
    staged = Task(
        goal="Get root then read flag.", name="staged",
        reward=chain(stage("root", contains("uid=0")), stage("flag", flag("CTF{x}"))),
        tools=(shell,), max_steps=5,
    )
    env_cls, _ = to_trl(staged, backend=MockBackend(exec_map={"id": "uid=0(root)"}))
    env = env_cls()
    env.reset()
    env._episode.transcript.append(
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "a", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "id"})}}]})
    env._episode.transcript.extend(
        env._episode.run_tool_calls(env._episode.transcript[-1]["tool_calls"]))
    env._episode.transcript.append(
        {"role": "assistant", "content": "CTF{x}", "tool_calls": None})
    reward = env.get_reward()
    assert reward == 1.0
    env._close()


def test_trl_env_close_calls_backend_down():
    down_count = [0]
    class CountingBackend(MockBackend):
        def down(self, world):
            down_count[0] += 1
    env_cls, _ = to_trl(make_task(), backend=CountingBackend())
    env = env_cls()
    env.reset()
    env._close()
    assert down_count[0] == 1


def test_trl_reward_func_returns_list_of_floats():
    _, reward_func = to_trl(make_task(), backend=MockBackend())
    prompts = ["Read the flag."]
    completions = ["The flag is CTF{win}"]
    results = reward_func(prompts=prompts, completions=completions,
                         completion_ids=None)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0] == 1.0


def test_trl_reward_func_staged_gated():
    staged = Task(
        goal="Get root then read flag.", name="staged",
        reward=chain(stage("root", contains("uid=0")), stage("flag", flag("CTF{x}"))),
        tools=(shell,), max_steps=5,
    )
    _, reward_func = to_trl(staged, backend=MockBackend())
    results = reward_func(prompts=["g"], completions=["CTF{x}"],
                         completion_ids=None)


def test_trl_env_reset_accepts_kwargs():
    env_cls, _ = to_trl(make_task(), backend=MockBackend())
    env = env_cls()
    # TRL passes reset_kwargs from dataset extra_info; reset must accept them
    prompt = env.reset(seed=42, extra_info={"index": 0})
    assert prompt == "Read the flag."
    env._close()


def test_trl_env_exposes_shell_tool():
    env_cls, _ = to_trl(
        make_task(),
        backend=MockBackend(exec_map={"cat /flag": "CTF{win}"}),
    )
    env = env_cls()
    env.reset()
    output = env.shell("cat /flag")
    assert output == "CTF{win}"
    env._close()


def test_trl_env_exposes_read_file_tool():
    class FileBackend(MockBackend):
        def up(self, spec, caps):
            from opencrl.backends.mock import MockWorld
            return MockWorld(files={"/flag": "CTF{win}"})
    env_cls, _ = to_trl(make_task(), backend=FileBackend())
    env = env_cls()
    env.reset()
    content = env.read_file("/flag")
    assert content == "CTF{win}"
    env._close()


def test_trl_env_close_not_exposed_as_tool():
    """close() should be prefixed so TRL doesn't expose it as a callable tool."""
    env_cls, _ = to_trl(make_task(), backend=MockBackend())
    # The class should have _close but not close
    assert hasattr(env_cls, "_close")
    assert not hasattr(env_cls, "close")


def test_trl_reward_func_handles_conversational_prompts():
    _, reward_func = to_trl(make_task(), backend=MockBackend())
    prompts = [[{"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Read the flag."}]]
    completions = ["The flag is CTF{win}"]
    results = reward_func(prompts=prompts, completions=completions,
                         completion_ids=None)
    assert results[0] == 1.0


def test_to_trl_prebuilds_shared_backend_once():
    calls = []

    class PrebuildBackend(MockBackend):
        def prebuild(self, spec, caps):
            calls.append(1)

    env_cls, _ = to_trl(make_task(), backend=PrebuildBackend())
    assert calls == [1]                  # prebuilt once at adapter construction
    env = env_cls()
    env.reset()
    env._close()
