"""Eval adapter: run N rollouts, write JSONL, report reward stats."""
from __future__ import annotations

import json

from cyberl.rollout import rollout
from cyberl.task import Task


def evaluate(task: Task, model, n: int = 1, out: str | None = None,
             backend=None) -> dict:
    rewards = []
    handle = open(out, "w") if out else None
    try:
        for _ in range(n):
            r = rollout(task, model, backend=backend)
            rewards.append(r.reward)
            if handle:
                handle.write(json.dumps(r.to_dict()) + "\n")
    finally:
        if handle:
            handle.close()
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    return {"n": n, "mean_reward": mean, "rewards": rewards}
