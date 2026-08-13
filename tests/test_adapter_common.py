from opencrl.adapters._common import _score_with_stages, _extract_prompt, _extract_completion
from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag, chain, stage, contains
from opencrl.state import State


class FakeWorld:
    agent = "agent"
    def exec(self, command, host=None): return ""
    def read_file(self, path, host=None): return None


def test_score_with_stages_scalar():
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,))
    state = State(world=FakeWorld(), transcript=[], answer="CTF{win}")
    reward, stages = _score_with_stages(task, state)
    assert reward == 1.0
    assert stages is None


def test_score_with_stages_chain():
    task = Task(goal="g", reward=chain(
        stage("root", contains("uid=0")),
        stage("flag", flag("CTF{x}")),
    ), tools=(shell,))
    state = State(world=FakeWorld(),
                  transcript=[{"role": "tool", "content": "uid=0(root)"}],
                  answer="CTF{x}")
    reward, stages = _score_with_stages(task, state)
    assert reward == 1.0
    assert stages == {"root": 1.0, "flag": 1.0}


def test_score_with_stages_chain_gated():
    task = Task(goal="g", reward=chain(
        stage("root", contains("uid=0")),
        stage("flag", flag("CTF{x}")),
    ), tools=(shell,))
    state = State(world=FakeWorld(), transcript=[], answer="CTF{x}")
    reward, stages = _score_with_stages(task, state)
    assert stages == {"root": 0.0, "flag": 0.0}
    assert reward == 0.0


def test_extract_prompt():
    task = Task(goal="read the flag", reward=flag("CTF{win}"), tools=(shell,))
    prompt = _extract_prompt(task)
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert prompt[1]["content"] == "read the flag"


def test_extract_completion():
    transcript = [
        {"role": "system", "content": "You are a security agent..."},
        {"role": "user", "content": "read the flag"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": '{"command": "cat /flag"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "CTF{win}"},
        {"role": "assistant", "content": "The flag is CTF{win}", "tool_calls": None},
    ]
    completion = _extract_completion(transcript)
    assert isinstance(completion, list)
    roles = [m["role"] for m in completion]
    assert "assistant" in roles
    assert "tool" in roles
    assert "system" not in roles
    assert "user" not in roles
