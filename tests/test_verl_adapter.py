import json
import os
import pytest

pandas = pytest.importorskip("pandas")

from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag, chain, stage, contains
from opencrl.backends.mock import MockBackend
from opencrl.adapters.verl_adapter import to_verl


def make_task():
    return Task(goal="Read the flag.", reward=flag("CTF{win}"), tools=(shell,),
                caps=Caps(offensive=True), max_steps=5, name="demo")


def make_notool_task():
    return Task(goal="Read the flag.", reward=flag("CTF{win}"), tools=(),
                caps=Caps(), max_steps=5, name="notool")


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
    assert prompt[0]["role"] == "user"
    assert prompt[0]["content"] == "Read the flag."
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


def test_to_verl_staged_reward_returns_callable(tmp_path):
    staged = Task(
        goal="Get root then read flag.", name="staged",
        reward=chain(stage("root", contains("uid=0")), stage("flag", flag("CTF{x}"))),
        tools=(shell,), max_steps=5,
    )
    _, reward_func = to_verl(staged, backend=MockBackend())
    assert callable(reward_func)
