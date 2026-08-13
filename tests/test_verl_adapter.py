import json
import os
import pytest

pandas = pytest.importorskip("pandas")

from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag, chain, stage, contains, file_exists
from opencrl.backends.mock import MockBackend
from opencrl.adapters.verl_adapter import to_verl


def make_task():
    return Task(goal="Read the flag.", reward=flag("CTF{win}"), tools=(shell,),
                caps=Caps(offensive=True), max_steps=5, name="demo")


def make_notool_task():
    return Task(goal="Read the flag.", reward=flag("CTF{win}"), tools=(),
                caps=Caps(), max_steps=5, name="notool")


def make_file_exists_task():
    return Task(goal="Create /pwned.", reward=file_exists("/pwned"),
                tools=(shell,), caps=Caps(offensive=True), max_steps=5,
                name="fileex")


def make_contains_task():
    return Task(goal="Run id.", reward=contains("uid=0"), tools=(shell,),
                caps=Caps(offensive=True), max_steps=5, name="cont")


def test_to_verl_writes_parquet(tmp_path):
    parquet_path, reward_func = to_verl(make_task(), backend=MockBackend(),
                                        out_path=str(tmp_path))
    assert os.path.exists(parquet_path)
    assert parquet_path.endswith(".parquet")
    df = pandas.read_parquet(parquet_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["data_source"] == "demo"
    # pandas returns ndarray for list-typed parquet columns; normalize to list
    prompt = list(row["prompt"]) if not isinstance(row["prompt"], list) else row["prompt"]
    assert isinstance(prompt, list)
    # Should include system + user messages (M2 fix)
    roles = [m["role"] for m in prompt]
    assert "system" in roles
    assert "user" in roles
    assert row["ability"] == "cyber"
    assert "reward_model" in row
    assert "extra_info" in row


def test_to_verl_sets_agent_name_when_tools_present(tmp_path):
    parquet_path, _ = to_verl(make_task(), backend=MockBackend(),
                              out_path=str(tmp_path))
    df = pandas.read_parquet(parquet_path)
    assert df.iloc[0]["agent_name"] == "tool_agent_loop"


def test_to_verl_no_agent_name_when_no_tools(tmp_path):
    parquet_path, _ = to_verl(make_notool_task(), backend=MockBackend(),
                             out_path=str(tmp_path))
    df = pandas.read_parquet(parquet_path)
    assert "agent_name" not in df.columns or pandas.isna(df.iloc[0].get("agent_name"))


def test_to_verl_returns_callable_reward_func():
    _, reward_func = to_verl(make_task(), backend=MockBackend())
    assert callable(reward_func)


def test_to_verl_ground_truth_extracted_from_flag(tmp_path):
    parquet_path, _ = to_verl(make_task(), backend=MockBackend(),
                             out_path=str(tmp_path))
    df = pandas.read_parquet(parquet_path)
    gt = df.iloc[0]["reward_model"]["ground_truth"]
    assert gt == "CTF{win}"


def test_to_verl_ground_truth_empty_for_file_exists(tmp_path):
    """file_exists() closures should NOT produce a ground_truth string."""
    parquet_path, _ = to_verl(make_file_exists_task(), backend=MockBackend(),
                             out_path=str(tmp_path))
    df = pandas.read_parquet(parquet_path)
    gt = df.iloc[0]["reward_model"]["ground_truth"]
    assert gt == ""  # not "/pwned"


def test_to_verl_ground_truth_empty_for_contains(tmp_path):
    """contains() closures should NOT produce a ground_truth string."""
    parquet_path, _ = to_verl(make_contains_task(), backend=MockBackend(),
                             out_path=str(tmp_path))
    df = pandas.read_parquet(parquet_path)
    gt = df.iloc[0]["reward_model"]["ground_truth"]
    assert gt == ""  # not "uid=0"


def test_to_verl_ground_truth_empty_for_staged(tmp_path):
    """chain() closures should NOT produce a ground_truth string."""
    staged = Task(
        goal="Get root then read flag.", name="staged",
        reward=chain(stage("root", contains("uid=0")), stage("flag", flag("CTF{x}"))),
        tools=(shell,), max_steps=5,
    )
    parquet_path, _ = to_verl(staged, backend=MockBackend(), out_path=str(tmp_path))
    df = pandas.read_parquet(parquet_path)
    gt = df.iloc[0]["reward_model"]["ground_truth"]
    assert gt == ""


def test_to_verl_reward_func_compute_score_signature():
    """Verl's compute_score API: fn(data_source, solution_str, ground_truth, extra_info=None) -> float."""
    _, reward_func = to_verl(make_task(), backend=MockBackend())
    # Call with Verl's positional API
    score = reward_func("demo", "The flag is CTF{win}", "CTF{win}")
    assert isinstance(score, float)
    assert score == 1.0


def test_to_verl_reward_func_wrong_answer_scores_zero():
    _, reward_func = to_verl(make_task(), backend=MockBackend())
    score = reward_func("demo", "I couldn't find it", "CTF{win}")
    assert score == 0.0


def test_to_verl_reward_func_staged_gated():
    staged = Task(
        goal="Get root then read flag.", name="staged",
        reward=chain(stage("root", contains("uid=0")), stage("flag", flag("CTF{x}"))),
        tools=(shell,), max_steps=5,
    )
    _, reward_func = to_verl(staged, backend=MockBackend())
    # Answer has flag but not uid=0 -> gated at root stage
    score = reward_func("staged", "CTF{x}", "")
    assert score == 0.0


def test_to_verl_reward_func_staged_full_pass():
    staged = Task(
        goal="Get root then read flag.", name="staged",
        reward=chain(stage("root", contains("uid=0")), stage("flag", flag("CTF{x}"))),
        tools=(shell,), max_steps=5,
    )
    _, reward_func = to_verl(staged, backend=MockBackend())
    # Answer contains both uid=0 and the flag
    score = reward_func("staged", "ran id -> uid=0; the flag is CTF{x}", "")
    assert score == 1.0


def test_to_verl_reward_func_with_extra_info():
    _, reward_func = to_verl(make_task(), backend=MockBackend())
    score = reward_func("demo", "The flag is CTF{win}", "CTF{win}",
                        extra_info={"index": 0})
    assert score == 1.0


def test_to_verl_empty_task_name_raises():
    task = Task(goal="g", reward=flag("CTF{x}"), tools=(shell,), name="")
    with pytest.raises(ValueError, match="task.name"):
        to_verl(task, backend=MockBackend())


def test_to_verl_decode_callback_param():
    """decode_callback is accepted but not required for the compute_score API."""
    def fake_decode(ids, mask):
        return "The flag is CTF{win}"
    _, reward_func = to_verl(make_task(), backend=MockBackend(),
                             decode_callback=fake_decode)
    score = reward_func("demo", "The flag is CTF{win}", "CTF{win}")
    assert score == 1.0
