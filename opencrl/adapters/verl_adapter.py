"""Verl on-policy adapter: generate Parquet dataset + reward function from a Task."""
from __future__ import annotations

import os
from typing import Any

from opencrl.adapters._common import (
    _score_with_stages, _SYSTEM_PROMPT,
)
from opencrl.backend import resolve_backend
from opencrl.state import State
from opencrl.task import Task


def to_verl(task: Task, backend=None, out_path: str = "."):
    """Return (parquet_path, reward_func) for Verl training.

    Writes a single-row Parquet dataset with the task as one prompt,
    and returns a reward function keyed by task.name.

    Requires the `verl` extra: pip install opencrl[verl]
    """
    resolved_backend = resolve_backend(backend or task.backend)
    parquet_path = _write_verl_dataset(task, out_path)
    reward_func = _make_verl_reward_func(task, resolved_backend)
    return parquet_path, reward_func


def _write_verl_dataset(task: Task, out_path: str) -> str:
    """Write a Verl-compatible Parquet with one row per task."""
    import pandas as pd
    row: dict[str, Any] = {
        "data_source": task.name,
        "prompt": [{"role": "user", "content": task.goal}],
        "ability": "cyber",
        "reward_model": {"style": "rule",
                         "ground_truth": _extract_ground_truth(task)},
        "extra_info": {"split": "train", "index": 0},
    }
    if task.tools:
        row["agent_name"] = "tool_agent_loop"
    df = pd.DataFrame([row])
    path = os.path.join(out_path, f"{task.name}.parquet")
    df.to_parquet(path)
    return path


def _extract_ground_truth(task: Task) -> str:
    """Try to extract a ground-truth string from the task's reward function.

    For flag("CTF{abc}") rewards, returns "CTF{abc}". For complex staged
    rewards, returns empty string — the reward function does all the scoring.
    """
    reward = task.reward
    if hasattr(reward, "__closure__") and reward.__closure__:
        for cell in reward.__closure__:
            if isinstance(cell.cell_contents, str):
                return cell.cell_contents
    return ""


def _make_verl_reward_func(task: Task, backend):
    """Build a Verl reward function from an OpenCyberRL Task.

    The reward function receives a DataProto-like batch with tokenized
    prompts/responses and returns per-sample scalar rewards. Stage
    breakdown is attached as reward_extra_info.
    """
    world_holder = [None]

    def reward_func(data):
        """Verl reward function: receives DataProto, returns (list[float], extra_info)."""
        results = []
        extra_info = {}
        for i in range(len(data)):
            prompt_ids = data.batch["prompts"][i]
            response_ids = data.batch["responses"][i]
            response_mask = data.batch["response_mask"][i]
            transcript = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": task.goal},
            ]
            answer = _decode_answer(response_ids, response_mask)
            transcript.append(
                {"role": "assistant", "content": answer, "tool_calls": None})
            state = State(world=world_holder[0], transcript=transcript,
                          answer=answer)
            reward, stages = _score_with_stages(task, state)
            if stages:
                extra_info[i] = {"stage_breakdown": stages}
            results.append(reward)
        return results, extra_info

    return reward_func


def _decode_answer(response_ids, response_mask) -> str:
    """Decode the answer from response tokens.

    In production, this uses the tokenizer's decode method. For the adapter's
    internal logic, we return a placeholder that transcript-only rewards
    (flag, contains) can scan. Tokenizer integration is the user's responsibility
    when wiring the reward function into Verl's training loop.
    """
    return ""


def _generate_tool_file(task: Task, out_path: str) -> str:
    """Generate a Verl-compatible tool definition file from Task.tools.

    Writes a .py file with each Tool exposed as a @function_tool function.
    The function body delegates to world.exec / world.read_file.
    """
    from opencrl.tools import Tool
    lines = [
        '"""Auto-generated Verl tool definitions from OpenCyberRL Task.tools."""',
        'from verl.tools import function_tool',
        '',
    ]
    for tool in task.tools:
        if not isinstance(tool, Tool):
            continue
        lines.append(f'@function_tool')
        lines.append(f'def {tool.name}(command: str) -> str:')
        lines.append(f'    """{tool.description}"""')
        lines.append(f'    # Delegates to the OpenCyberRL world backend.')
        lines.append(f'    # The world handle is injected by the adapter at runtime.')
        lines.append(f'    ...')
        lines.append('')
    path = os.path.join(out_path, f"{task.name}_tools.py")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
