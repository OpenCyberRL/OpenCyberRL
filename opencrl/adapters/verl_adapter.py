"""Verl on-policy adapter: generate Parquet dataset + reward function from a Task."""
from __future__ import annotations

import os
from typing import Any, Callable

from opencrl.adapters._common import (
    _score_with_stages, _extract_answer, _SYSTEM_PROMPT, _extract_prompt,
)
from opencrl.backend import resolve_backend
from opencrl.state import State
from opencrl.task import Task


def to_verl(task: Task, backend=None, out_path: str = ".",
           decode_callback: Callable[[list, list], str] | None = None):
    """Return (parquet_path, reward_func) for Verl training.

    Writes a single-row Parquet dataset with the task as one prompt,
    and returns a reward function matching Verl's compute_score API:
    ``fn(data_source, solution_str, ground_truth, extra_info=None) -> float``.

    The reward function only supports transcript-only rewards (flag, contains)
    since Verl's compute_score API does not provide access to the running world.
    For world-dependent rewards (file_exists, exec), write a custom reward
    function that accesses the world through Verl's tool_agent_loop.

    Args:
        task: The OpenCyberRL Task to adapt.
        backend: Backend name or instance (unused by the reward function,
            but kept for API symmetry with to_trl).
        out_path: Directory to write the Parquet file.
        decode_callback: Optional ``fn(response_ids, response_mask) -> str``
            to decode token IDs into text for transcript-only rewards. If
            None, solution_str from Verl's API is used directly as the answer.

    Requires the `verl` extra: pip install opencrl[verl]
    """
    if not task.name:
        raise ValueError(
            "opencrl: to_verl requires task.name to be set "
            "(use the @task decorator or pass name=... to Task)")
    resolved_backend = resolve_backend(backend or task.backend)
    parquet_path = _write_verl_dataset(task, out_path)
    reward_func = _make_verl_reward_func(task, decode_callback)
    return parquet_path, reward_func


def _write_verl_dataset(task: Task, out_path: str) -> str:
    """Write a Verl-compatible Parquet with one row per task."""
    import pandas as pd
    row: dict[str, Any] = {
        "data_source": task.name,
        "prompt": _extract_prompt(task),
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
    """Extract a ground-truth string from a flag() reward function.

    Only flag() closures are introspected (identified by a single freevar
    named 'expected'). For file_exists(), contains(), chain(), goals(),
    and custom reward functions, returns empty string — the reward function
    does all the scoring.
    """
    reward = task.reward
    if not hasattr(reward, "__code__"):
        return ""
    freevars = reward.__code__.co_freevars
    if freevars == ("expected",) and reward.__closure__:
        cell = reward.__closure__[0]
        if isinstance(cell.cell_contents, str):
            return cell.cell_contents
    return ""


def _make_verl_reward_func(task: Task,
                          decode_callback: Callable | None = None):
    """Build a Verl reward function from an OpenCyberRL Task.

    The reward function matches Verl's compute_score API:
    ``fn(data_source, solution_str, ground_truth, extra_info=None) -> float``

    Only transcript-only rewards (flag, contains) are supported. The world
    is not available through this API — Verl's compute_score doesn't pass it.
    For world-dependent rewards, use a custom reward function.
    """
    def reward_func(data_source, solution_str, ground_truth,
                   extra_info=None) -> float:
        """Verl compute_score reward function."""
        transcript = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": task.goal},
        ]
        if decode_callback is not None:
            # When wired into Verl's training loop, the caller can pass
            # a decode_callback that converts token IDs to text. In the
            # compute_score API, solution_str is already decoded text.
            # decode_callback is used when the caller wraps this function
            # and has access to raw token IDs.
            pass
        answer = solution_str if solution_str else ""
        transcript.append(
            {"role": "assistant", "content": answer, "tool_calls": None})
        state = State(world=None, transcript=transcript, answer=answer)
        reward, _ = _score_with_stages(task, state)
        return reward

    return reward_func
