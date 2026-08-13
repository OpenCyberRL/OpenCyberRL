"""Shared helpers for RL framework adapters."""
from __future__ import annotations

from opencrl.reward import resolve_reward
from opencrl.state import State
from opencrl.task import Task


def _score_with_stages(task: Task, state: State) -> tuple[float, dict[str, float] | None]:
    """Run the verifier and split into (aggregate, stage breakdown)."""
    scored = task.reward(state)
    return resolve_reward(scored)


_SYSTEM_PROMPT = (
    "You are a security agent working inside an isolated sandbox. "
    "Use the tools to accomplish the goal, then state your final answer."
)


def _extract_prompt(task: Task) -> list[dict]:
    """Build the conversational prompt (system + user) from a Task."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": task.goal},
    ]


def _extract_completion(transcript: list[dict]) -> list[dict]:
    """Extract assistant + tool messages from a transcript (skip system/user)."""
    return [m for m in transcript if m["role"] not in ("system", "user")]


def _extract_answer(transcript: list[dict]) -> str:
    """Extract the agent's final answer (last assistant content)."""
    for m in reversed(transcript):
        if m["role"] == "assistant" and m.get("content"):
            return m["content"]
    return ""
